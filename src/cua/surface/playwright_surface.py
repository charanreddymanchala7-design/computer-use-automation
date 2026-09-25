"""A Surface backed by a real Chromium driven through Playwright (sync API).

Single-threaded on purpose: every Playwright call happens on the caller's thread. Anything that
needs to run beside it (an operator page, a terminal prompt) only touches thread-safe state such
as the control lease, never the browser, so a human takes over by using the same headed window.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

from playwright.sync_api import (
    BrowserContext,
    Dialog,
    ElementHandle,
    Page,
    Playwright,
    Route,
    sync_playwright,
)
from playwright.sync_api import Error as PlaywrightError

from cua.surface.base import (
    MAX_WAIT_MS,
    Action,
    ActionResult,
    DialogEvent,
    ElementInfo,
    Observation,
    RequestGuard,
    RequestInfo,
    StaleRefError,
    SurfaceError,
    SurfaceUnavailable,
    UnknownRefError,
    UnknownSecretError,
)
from cua.surface.observe import TEXT_JS, RefTable, observe_frames


def _accept_all_but_prompts(kind: str, message: str) -> bool:
    return kind != "prompt"


@dataclass(frozen=True)
class SurfaceConfig:
    headless: bool = True
    debug_port: int | None = None  # a second client can attach here (loopback only)
    user_data_dir: Path | None = None  # default: a fresh private directory, removed on close
    viewport: tuple[int, int] = (1280, 800)
    default_timeout_ms: int = 8000
    # Playwright dismisses dialogs nobody handles, so a `confirm` would silently return false.
    accept_dialog: Callable[[str, str], bool] = _accept_all_but_prompts


def _first_line(exc: BaseException) -> str:
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0]


def _describe(info: ElementInfo) -> str:
    label = info.text or info.label_hint or info.attrs.get("name") or ""
    return f"{info.ref} <{info.tag}> {label[:60]!r}".rstrip()


class PlaywrightSurface:
    def __init__(
        self, config: SurfaceConfig | None = None, *, secrets: Mapping[str, str] | None = None
    ) -> None:
        self._config = config or SurfaceConfig()
        self._secrets = dict(secrets or {})  # provided by the caller; never read from the env here
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._profile: Path | None = None
        self._owns_profile = False
        self._refs = RefTable()
        self._dialog_log: list[DialogEvent] = []
        self._act_dialogs: list[DialogEvent] | None = None
        self._route_handler: Callable[[Route], None] | None = None

    # --- lifecycle ----------------------------------------------------------------------------

    def open(self) -> None:
        if self._context is not None:
            return
        profile = self._config.user_data_dir
        self._owns_profile = profile is None
        profile = profile or Path(tempfile.mkdtemp(prefix="cua-profile-"))
        profile.mkdir(parents=True, exist_ok=True)
        os.chmod(profile, 0o700)  # cookies live here
        self._profile = profile
        args: list[str] = []
        if self._config.debug_port is not None:
            args += [
                f"--remote-debugging-port={self._config.debug_port}",
                "--remote-debugging-address=127.0.0.1",
            ]
        width, height = self._config.viewport
        try:
            self._playwright = sync_playwright().start()
            self._context = self._playwright.chromium.launch_persistent_context(
                str(profile),
                headless=self._config.headless,
                viewport={"width": width, "height": height},
                args=args,
            )
        except PlaywrightError as exc:
            self.close()
            if "Executable doesn't exist" in str(exc):
                raise SurfaceUnavailable(
                    "Chromium is not installed for Playwright: "
                    "run `uv run playwright install chromium`"
                ) from exc
            raise SurfaceError(f"could not start the browser: {_first_line(exc)}") from exc
        self._context.set_default_timeout(self._config.default_timeout_ms)
        self._context.on("dialog", self._on_dialog)
        pages = self._context.pages
        self._page = pages[0] if pages else self._context.new_page()

    def close(self) -> None:
        context, playwright = self._context, self._playwright
        self._context = self._page = self._playwright = None
        try:
            if context is not None:
                context.close()
        except PlaywrightError:
            pass
        try:
            if playwright is not None:
                playwright.stop()
        except PlaywrightError:
            pass
        if self._owns_profile and self._profile is not None:
            shutil.rmtree(self._profile, ignore_errors=True)

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def reset(self) -> None:
        """Forget refs and dialogs and sign out (cookies), leaving one blank page."""
        context = self._require_context()
        self._refs = RefTable()
        self._dialog_log = []
        self.set_request_guard(None)  # a guard belongs to one run and must not leak into the next
        context.clear_cookies()
        for extra in context.pages[1:]:
            extra.close()
        self.page.goto("about:blank")

    @property
    def page(self) -> Page:
        if self._page is None:
            raise SurfaceError("the surface is not open")
        return self._page

    @property
    def user_data_dir(self) -> Path:
        if self._profile is None:
            raise SurfaceError("the surface is not open")
        return self._profile

    @property
    def debug_endpoint(self) -> str | None:
        port = self._config.debug_port
        return None if port is None else f"http://127.0.0.1:{port}"

    def _require_context(self) -> BrowserContext:
        if self._context is None:
            raise SurfaceError("the surface is not open")
        return self._context

    # --- the request guard ----------------------------------------------------------------------

    def set_request_guard(self, guard: RequestGuard | None) -> None:
        """Every request, including frame navigations, form posts and meta refreshes, passes here.

        Blocking at the network layer is what makes an allowlist real: an in-page link or script
        can navigate without any action ever being issued. A blocked request is answered with a
        short 403 page (so the model can read that it was blocked) and never reaches the server.
        """
        context = self._require_context()
        if self._route_handler is not None:
            context.unroute("**/*", self._route_handler)
            self._route_handler = None
        if guard is None:
            return

        def handler(route: Route) -> None:
            request = route.request
            info = RequestInfo(
                request.url, request.method, request.resource_type, request.is_navigation_request()
            )
            try:
                allowed = guard(info)
            except Exception:
                allowed = False
            if allowed:
                route.continue_()
            else:
                route.fulfill(status=403, content_type="text/plain", body="BLOCKED BY POLICY")

        self._route_handler = handler
        context.route("**/*", handler)

    # --- dialogs ------------------------------------------------------------------------------

    def _on_dialog(self, dialog: Dialog) -> None:
        accept = self._config.accept_dialog(dialog.type, dialog.message)
        event = DialogEvent(dialog.type, dialog.message, accept)
        self._dialog_log.append(event)
        if self._act_dialogs is not None:
            self._act_dialogs.append(event)
        if accept:
            dialog.accept()
        else:
            dialog.dismiss()

    # --- observing ----------------------------------------------------------------------------

    def observe(self) -> Observation:
        self._refs = RefTable()  # refs from an older observation are no longer valid
        frames = observe_frames(self.page, self._refs, self._secrets.values())
        dialogs, self._dialog_log = self._dialog_log, []
        viewport = self.page.viewport_size or {"width": 0, "height": 0}
        return Observation(
            url=self.page.url,
            title=self.page.title(),
            viewport=(viewport["width"], viewport["height"]),
            frames=frames,
            screenshot=self.page.screenshot(type="png"),
            dialogs=dialogs,
        )

    def element_info(self, ref: str) -> ElementInfo:
        info = self._refs.infos.get(ref)
        if info is None:
            raise UnknownRefError(f"unknown ref {ref!r}; refs come from the latest observe")
        return info

    def wait_for_text(self, text: str, *, timeout_ms: int = 5000) -> bool:
        """Poll every frame for visible text; a timeout is an answer (False), not an error."""
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            for frame in self.page.frames:
                try:
                    if text in frame.evaluate(TEXT_JS):
                        return True
                except PlaywrightError:
                    continue  # mid-navigation; look again
            if time.monotonic() >= deadline:
                return False
            self.page.wait_for_timeout(100)

    # --- acting -------------------------------------------------------------------------------

    def act(self, action: Action) -> ActionResult:
        self._act_dialogs = []
        try:
            detail = self._perform(action)
            self._settle()
            return ActionResult(True, detail, self.page.url, list(self._act_dialogs))
        except PlaywrightError as exc:
            # a timeout or an obscured element is something a model can recover from
            error = _first_line(exc)
            return ActionResult(
                False, f"{action.kind} failed", self.page.url, list(self._act_dialogs), error
            )
        finally:
            self._act_dialogs = None

    def _perform(self, action: Action) -> str:
        page = self.page
        if action.kind == "navigate":
            assert action.url is not None
            page.goto(action.url)
            return f"navigated to {action.url}"
        if action.kind == "press":
            assert action.key is not None
            page.keyboard.press(action.key)
            return f"pressed {action.key}"
        if action.kind == "wait":
            ms = min(max(action.ms or 0, 0), MAX_WAIT_MS)
            page.wait_for_timeout(ms)
            return f"waited {ms} ms"
        if action.kind == "click" and action.ref is None:
            assert action.x is not None
            assert action.y is not None
            page.mouse.click(action.x, action.y)
            return f"clicked at ({action.x:.0f}, {action.y:.0f})"
        assert action.ref is not None
        handle, info = self._resolve(action.ref)
        if action.kind == "click":
            handle.click()
            return f"clicked {_describe(info)}"
        if action.kind == "fill":
            return self._fill(handle, info, action)
        assert action.option is not None
        return self._select(handle, info, action.option)

    def _fill(self, handle: ElementHandle, info: ElementInfo, action: Action) -> str:
        if action.secret is not None:
            if action.secret not in self._secrets:
                raise UnknownSecretError(f"no secret named {action.secret!r} was provided")
            handle.fill(self._secrets[action.secret])
            return f"typed secret {action.secret} into {_describe(info)}"
        assert action.text is not None
        handle.fill(action.text)
        return f"typed {action.text!r} into {_describe(info)}"

    def _select(self, handle: ElementHandle, info: ElementInfo, option: str) -> str:
        index = handle.evaluate(
            "(el, v) => Array.from(el.options).findIndex("
            "(o) => o.value === v || o.text.trim() === v)",
            option,
        )
        if index < 0:
            raise PlaywrightError(f"no option {option!r} in {_describe(info)}")
        handle.select_option(index=index)
        return f"selected {option!r} in {_describe(info)}"

    def _resolve(self, ref: str) -> tuple[ElementHandle, ElementInfo]:
        handle = self._refs.handles.get(ref)
        if handle is None:
            raise UnknownRefError(f"unknown ref {ref!r}; refs come from the latest observe")
        try:
            alive = bool(handle.evaluate("(el) => el.isConnected"))
        except PlaywrightError:
            alive = False  # its execution context was destroyed by a navigation
        if not alive:
            raise StaleRefError(f"{ref} is no longer on the page; observe again")
        return handle, self._refs.infos[ref]

    def _settle(self) -> None:
        """Give navigations the action started a moment to finish loading."""
        for frame in self.page.frames:
            try:
                frame.wait_for_load_state("load", timeout=3000)
            except PlaywrightError:
                continue
        self.page.wait_for_timeout(50)
