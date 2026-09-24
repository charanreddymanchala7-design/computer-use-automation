"""Browser lifecycle: private profile, loopback-only debug port, a second client attaching.

Kept apart from the other surface tests because the synchronous Playwright API allows only one
instance per thread: these tests own their surface, and the shared one belongs to other modules.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from tests.conftest import _free_port
from tests.mockbank_support import MockHandle

from cua.surface import Action, PlaywrightSurface, SurfaceConfig

pytestmark = pytest.mark.browser

# The second client is a separate process, exactly as a real second client would be.
ATTACH = """
import sys
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    browser = pw.chromium.connect_over_cdp(sys.argv[1])
    for context in browser.contexts:
        for page in context.pages:
            print(page.url)
    browser.close()
"""


def test_the_profile_is_private_not_the_default_and_removed_on_close() -> None:
    surface = PlaywrightSurface(SurfaceConfig(headless=True))
    surface.open()
    profile = surface.user_data_dir
    try:
        assert profile.name.startswith("cua-profile-")
        assert stat.S_IMODE(profile.stat().st_mode) == 0o700
        assert Path.home() not in profile.parents  # never the user's own browser profile
    finally:
        surface.close()
    assert not profile.exists()


def test_a_caller_supplied_profile_is_kept_after_close(tmp_path: Path) -> None:
    surface = PlaywrightSurface(SurfaceConfig(headless=True, user_data_dir=tmp_path / "profile"))
    surface.open()
    surface.close()
    assert (tmp_path / "profile").is_dir()  # we only delete what we created


def test_closing_twice_is_harmless() -> None:
    surface = PlaywrightSurface(SurfaceConfig(headless=True))
    surface.open()
    surface.open()  # already open: no second browser
    surface.close()
    surface.close()


def test_a_second_client_can_attach_to_the_same_live_session(mock: MockHandle) -> None:
    surface = PlaywrightSurface(SurfaceConfig(headless=True, debug_port=_free_port()))
    surface.open()
    try:
        surface.act(Action.navigate(f"{mock.base}/msv/login.cgi"))
        endpoint = surface.debug_endpoint
        assert endpoint is not None
        assert endpoint.startswith("http://127.0.0.1:")  # loopback only
        done = subprocess.run(
            [sys.executable, "-c", ATTACH, endpoint],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert f"{mock.base}/msv/login.cgi" in done.stdout.splitlines()
    finally:
        surface.close()


@pytest.mark.skipif(
    os.environ.get("CUA_TEST_HEADED") != "1" or sys.platform not in ("darwin", "win32"),
    reason="opens a real window; run with CUA_TEST_HEADED=1 on a desktop",
)
def test_headed_mode_launches_too(mock: MockHandle) -> None:
    surface = PlaywrightSurface(SurfaceConfig(headless=False, debug_port=_free_port()))
    surface.open()
    try:
        assert surface.act(Action.navigate(f"{mock.base}/msv/login.cgi")).ok
        assert surface.observe().title == "MemberServ 3.1"
    finally:
        surface.close()
