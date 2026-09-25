"""The discovery loop: observe, decide, act, until the goal is met or the model is stuck.

Every action goes through the gateway, so policy and audit apply here exactly as in replay. The
element's locator bundle is harvested *before* the action (the page changes after it), values
the caller supplies are bound by name (a parameter or a secret) so the recording is
parameterized by construction, and only successful actions are recorded. The result is a
``RecordedRun`` that owes nothing to the conversation that produced it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from cua.agent.prompts import NUDGE, SYSTEM_PROMPT, first_message
from cua.agent.recording import (
    ObservationDigest,
    Outcome,
    RecordedDialog,
    RecordedRun,
    RecordedStep,
    StepKind,
    state_key,
)
from cua.agent.render import render_observation
from cua.agent.tools import ACT_KINDS, TOOLS, DiscoveryTask, Limits
from cua.artifact import (
    CoordinatesLocator,
    LiteralValue,
    LocatorBundle,
    ParamRef,
    RiskClass,
    SecretRef,
    ValueRef,
    parameterize,
)
from cua.evlog import EventLog
from cua.gateway import ActionGateway, Decision, Reason
from cua.llm import (
    LLM,
    ImagePart,
    LLMError,
    Message,
    Part,
    TextPart,
    ToolResult,
    ToolUse,
)
from cua.surface import Action, LocatingSurface, Observation, SurfaceError

_STATE_CHANGING = frozenset({"navigate", "click", "fill", "select", "press"})
_OMITTED = "[earlier screenshot omitted]"
_OUTSIDE_POLICY = "this is outside what you are permitted to do."


@dataclass
class _Outcome:
    """The result of one tool call, before it becomes a message to the model."""

    text: str
    is_error: bool = False
    observation: Observation | None = None
    finished: bool = False
    action_key: tuple[Any, ...] | None = None  # set for act attempts, to catch repetition


def _error(text: str) -> _Outcome:
    return _Outcome(f"ERROR: {text}", is_error=True)


def _blocked(decision: Decision) -> _Outcome:
    if decision.reason is Reason.CONFIRMATION_REQUIRED:
        text = (
            "BLOCKED: this action is irreversible and needs a human's confirmation. Do not "
            "retry it. If the goal requires it, call finish with success=false and explain."
        )
    elif decision.reason is Reason.IRREVERSIBLE_BLOCKED:
        text = (
            "BLOCKED: irreversible actions are not permitted here. Do not retry it. If the goal "
            "requires it, call finish with success=false and explain."
        )
    else:
        detail = f" ({decision.detail})" if decision.detail else ""
        text = f"BLOCKED: {decision.reason.value}{detail}: {_OUTSIDE_POLICY}"
    return _Outcome(text, is_error=True)


@dataclass
class _Prepared:
    action: Action
    value: ValueRef | None = None
    url: str | None = None
    ref: str | None = None
    point: tuple[float, float] | None = None


@dataclass
class _Run:
    task: DiscoveryTask
    started: float
    values: dict[str, str]
    steps: list[RecordedStep] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    secrets_used: list[str] = field(default_factory=list)
    last_obs: Observation | None = None
    last_state: str = ""
    last_key: tuple[Any, ...] | None = None
    repeats: int = 0
    errors: int = 0
    stalled: int = 0
    nudges: int = 0
    llm_calls: int = 0
    outcome: Outcome | None = None
    reason: str = ""
    summary: str = ""


def _squash(text: str) -> str:
    return " ".join(text.split())


def _strip_images(message: Message) -> Message:
    def strip(part: Part) -> Part:
        if isinstance(part, ImagePart):
            return TextPart(_OMITTED)
        if isinstance(part, ToolResult):
            content = tuple(
                TextPart(_OMITTED) if isinstance(p, ImagePart) else p for p in part.content
            )
            return ToolResult(part.tool_use_id, content, part.is_error)
        return part

    return Message(message.role, tuple(strip(p) for p in message.parts))


def _has_image(message: Message) -> bool:
    return any(
        isinstance(p, ImagePart)
        or (isinstance(p, ToolResult) and any(isinstance(c, ImagePart) for c in p.content))
        for p in message.parts
    )


def compact(messages: Sequence[Message]) -> list[Message]:
    """Keep only the latest screenshot: old ones cost tokens and describe a page that is gone."""
    last = max((i for i, m in enumerate(messages) if _has_image(m)), default=-1)
    return [m if i >= last else _strip_images(m) for i, m in enumerate(messages)]


class DiscoveryLoop:
    def __init__(
        self,
        surface: LocatingSurface,
        gateway: ActionGateway,
        llm: LLM,
        log: EventLog,
        *,
        limits: Limits | None = None,
        clock: Callable[[], float] = time.monotonic,
        run_id: str = "run",
        on_screenshot: Callable[[str, bytes], None] | None = None,
    ) -> None:
        self._on_screenshot = on_screenshot
        self._shots = 0
        self._surface = surface
        self._gateway = gateway
        self._llm = llm
        self._log = log
        self._limits = limits or Limits()
        self._clock = clock
        self._run_id = run_id

    # --- the run ----------------------------------------------------------------------------

    def run(self, task: DiscoveryTask) -> RecordedRun:
        run = _Run(task, self._clock(), {name: spec.value for name, spec in task.params.items()})
        self._log.emit("discovery_start", goal=task.goal, model=self._llm.model)
        if self._open_start_page(run):
            self._converse(run)
        return self._conclude(run)

    def _open_start_page(self, run: _Run) -> bool:
        task = run.task
        gated = self._gateway.act(Action.navigate(task.start_url), step="s1")
        if not gated.executed:
            detail = gated.decision.detail or gated.decision.reason.value
            run.outcome, run.reason = "failed", f"the start page is not permitted ({detail})"
            return False
        assert gated.result is not None
        if not gated.result.ok:
            run.outcome = "failed"
            run.reason = (
                f"could not open the start page: {gated.result.error or gated.result.detail}"
            )
            return False
        obs = self._observe(run)
        run.steps.append(
            RecordedStep(
                id="s1",
                kind="navigate",
                description="open the start page",
                url=task.start_url,
                risk=gated.decision.risk,
                after=ObservationDigest.from_observation(obs),
            )
        )
        return True

    def _keep(self, name: str, png: bytes) -> None:
        """Hand a page the model was shown to whoever keeps evidence, numbered in order."""
        if self._on_screenshot is not None:
            self._on_screenshot(f"{self._shots:02d}-{name}", png)
            self._shots += 1

    def _observe(self, run: _Run) -> Observation:
        obs = self._surface.observe()
        run.last_obs = obs
        return obs

    def _converse(self, run: _Run) -> None:
        assert run.last_obs is not None
        obs = run.last_obs
        run.last_state = state_key(obs)
        self._keep("start", obs.screenshot)
        messages = [
            Message.user(
                TextPart(first_message(run.task)),
                TextPart(render_observation(obs)),
                ImagePart(obs.screenshot),
            )
        ]
        limits = self._limits
        while run.outcome is None:
            if run.llm_calls >= limits.max_steps:
                run.outcome, run.reason = (
                    "max_steps",
                    f"stopped after {limits.max_steps} model calls",
                )
                break
            if self._clock() - run.started > limits.timeout_s:
                run.outcome, run.reason = "timeout", f"stopped after {limits.timeout_s:.0f} seconds"
                break
            try:
                response = self._llm.step(
                    system=SYSTEM_PROMPT, messages=compact(messages), tools=TOOLS
                )
            except LLMError as exc:
                run.outcome, run.reason = "failed", f"the model call failed: {exc}"
                break
            run.llm_calls += 1
            messages.append(response.as_message())
            if not response.tool_calls:
                self._nudge(run, messages)
                continue
            run.nudges = 0
            results: list[Part] = []
            for call in response.tool_calls:
                outcome = self._dispatch(run, call)
                results.append(self._to_result(call, outcome))
                if outcome.finished:
                    break
            if run.outcome is not None:
                break
            messages.append(Message.user(*results))
            self._check_dead_end(run)

    def _nudge(self, run: _Run, messages: list[Message]) -> None:
        run.nudges += 1
        if run.nudges > self._limits.max_nudges:
            run.outcome, run.reason = "dead_end", "the model stopped calling tools"
        else:
            messages.append(Message.user(TextPart(NUDGE)))

    def _check_dead_end(self, run: _Run) -> None:
        limits = self._limits
        if run.repeats >= limits.max_repeats:
            reason = f"repeated the same action {run.repeats} times in a row"
        elif run.errors >= limits.max_errors:
            reason = f"{run.errors} errors in a row"
        elif run.stalled >= limits.max_stalled:
            reason = f"no progress: the page did not change after {run.stalled} actions"
        else:
            return
        run.outcome, run.reason = "dead_end", reason

    def _to_result(self, call: ToolUse, outcome: _Outcome) -> ToolResult:
        first_line = outcome.text.splitlines()[0] if outcome.text else ""
        self._log.emit(
            "tool",
            action=call.name,
            outcome="error" if outcome.is_error else "ok",
            reason=first_line[:160],
        )
        if outcome.observation is None:
            return ToolResult(call.id, (TextPart(outcome.text),), outcome.is_error)
        obs = outcome.observation
        self._keep(call.name, obs.screenshot)
        return ToolResult(
            call.id,
            (TextPart(f"{outcome.text}\n\n{render_observation(obs)}"), ImagePart(obs.screenshot)),
            outcome.is_error,
        )

    def _conclude(self, run: _Run) -> RecordedRun:
        task, meter = run.task, self._llm.meter
        final = ObservationDigest.from_observation(run.last_obs) if run.last_obs else None
        outcome: Outcome = run.outcome or "failed"
        recorded = RecordedRun(
            run_id=self._run_id,
            goal=task.goal,
            start_url=task.start_url,
            model=self._llm.model,
            params={
                name: "<withheld>" if spec.sensitive else spec.value
                for name, spec in task.params.items()
            },
            secrets_used=run.secrets_used,
            steps=run.steps,
            outputs=run.outputs,
            missing_outputs=[name for name in task.outputs if name not in run.outputs],
            outcome=outcome,
            reason=run.reason,
            summary=run.summary,
            llm_steps=meter.steps,
            tokens=meter.usage.total_tokens,
            cost_usd=meter.cost_usd,
            duration_s=self._clock() - run.started,
            final=final,
        )
        self._log.emit(
            "discovery_end",
            outcome=outcome,
            reason=run.reason,
            steps=len(run.steps),
            llm_steps=meter.steps,
            tokens=meter.usage.total_tokens,
        )
        return recorded

    # --- tools ------------------------------------------------------------------------------

    def _dispatch(self, run: _Run, call: ToolUse) -> _Outcome:
        if call.name == "observe":
            outcome = _Outcome("OK: the page as it is now", observation=self._observe(run))
        elif call.name == "act":
            outcome = self._act(run, call.input)
        elif call.name == "extract":
            outcome = self._extract(run, call.input)
        elif call.name == "finish":
            return self._finish(run, call.input)
        else:
            outcome = _error(f"unknown tool {call.name!r}; use observe, act, extract or finish")
        self._account(run, outcome)
        return outcome

    def _account(self, run: _Run, outcome: _Outcome) -> None:
        """Bookkeeping that feeds dead-end detection."""
        run.errors = run.errors + 1 if outcome.is_error else 0
        if outcome.action_key is not None:
            run.repeats = run.repeats + 1 if outcome.action_key == run.last_key else 1
            run.last_key = outcome.action_key
        if outcome.observation is None or outcome.is_error:
            return
        state = state_key(outcome.observation)
        if outcome.action_key is not None and outcome.action_key[0] in _STATE_CHANGING:
            run.stalled = run.stalled + 1 if state == run.last_state else 0
        run.last_state = state

    def _finish(self, run: _Run, args: dict[str, Any]) -> _Outcome:
        success = args.get("success") is True
        run.summary = str(args.get("summary") or "")
        run.outcome = "finished" if success else "failed"
        run.reason = (
            "the model reported that the goal was achieved"
            if success
            else "the model reported that the goal was not achieved"
        )
        return _Outcome("OK: run ended", finished=True)

    # --- act --------------------------------------------------------------------------------

    def _act(self, run: _Run, args: dict[str, Any]) -> _Outcome:
        kind = args.get("kind")
        reason = (str(args.get("reason") or "").strip() or str(kind))[:200]
        if kind not in ACT_KINDS:
            return _error(f"unknown kind {kind!r}; use one of: {', '.join(ACT_KINDS)}")
        if kind == "wait":
            return self._wait(run, args)
        if kind == "wait_for":
            return self._wait_for(run, args, reason)
        prepared = self._prepare(run, str(kind), args)
        if isinstance(prepared, _Outcome):
            return prepared
        key = (
            kind,
            prepared.ref,
            prepared.point,
            prepared.url,
            args.get("key"),
            args.get("option"),
        )
        step_id = f"s{len(run.steps) + 1}"
        already_blocked = len(self._gateway.blocked_navigations)
        try:
            bundle = self._locator_for(run, str(kind), prepared)
            gated = self._gateway.act(prepared.action, step=step_id)
        except SurfaceError as exc:
            outcome = _error(str(exc))
            outcome.action_key = key
            return outcome
        if not gated.executed:
            blocked = _blocked(gated.decision)
            blocked.action_key = key
            return blocked
        assert gated.result is not None
        result = gated.result
        if not result.ok:
            failed = _error(result.error or result.detail)
            failed.action_key = key
            return failed
        provoked = self._gateway.blocked_navigations[already_blocked:]
        if provoked:
            refused = _Outcome(
                f"BLOCKED: the page tried to go to {provoked[0]}, which policy does not allow; "
                "the action was not recorded. Try a different element, or finish with "
                "success=false if the goal needs it.",
                is_error=True,
                observation=self._observe(run),
                action_key=key,
            )
            return refused
        obs = self._observe(run)
        run.steps.append(
            RecordedStep(
                id=step_id,
                kind=kind,
                description=reason,
                url=prepared.url,
                locator=bundle,
                value=prepared.value,
                risk=gated.decision.risk,
                dialogs=[
                    RecordedDialog(kind=d.kind, message=d.message, accepted=d.accepted)
                    for d in result.dialogs
                ],
                after=ObservationDigest.from_observation(obs),
            )
        )
        if isinstance(prepared.value, SecretRef) and prepared.value.name not in run.secrets_used:
            run.secrets_used.append(prepared.value.name)
        return _Outcome(f"OK: {result.detail}", observation=obs, action_key=key)

    def _locator_for(self, run: _Run, kind: str, prepared: _Prepared) -> LocatorBundle | None:
        """Harvested before acting: the element is gone or changed once the action has run."""
        if prepared.ref is not None:
            return self._surface.harvest(prepared.ref, params=run.values, for_click=kind == "click")
        if prepared.point is not None:
            width, height = run.last_obs.viewport if run.last_obs else (1280, 800)
            x, y = prepared.point
            return LocatorBundle(
                description=f"point ({x:.0f}, {y:.0f})",
                strategies=[
                    CoordinatesLocator(
                        x=x,
                        y=y,
                        viewport_width=width,
                        viewport_height=height,
                        rationale="The model had no ref for this element: tied to this viewport",
                    )
                ],
            )
        return None

    def _bind_literal(self, run: _Run, text: str) -> ValueRef:
        """A constant that equals a parameter's value is that parameter, or replay would be
        stuck on today's value."""
        for name, value in run.values.items():
            if text == value:
                return ParamRef(name=name)
        return LiteralValue(value=text)

    def _prepare(self, run: _Run, kind: str, args: dict[str, Any]) -> _Prepared | _Outcome:
        ref, x, y = args.get("ref"), args.get("x"), args.get("y")
        if kind == "navigate":
            url = args.get("url")
            if not url:
                return _error("navigate needs a url")
            return _Prepared(Action.navigate(str(url)), url=parameterize(str(url), run.values))
        if kind == "press":
            key = args.get("key")
            if not key:
                return _error("press needs a key")
            return _Prepared(Action.press(str(key)), value=LiteralValue(value=str(key)))
        if kind == "click":
            if ref:
                return _Prepared(Action.click(str(ref)), ref=str(ref))
            if x is not None and y is not None:
                return _Prepared(Action.click_at(float(x), float(y)), point=(float(x), float(y)))
            return _error("click needs a ref (or x and y for a coordinate click)")
        if not ref:
            return _error(f"{kind} needs a ref")
        if kind == "select":
            option = args.get("option")
            if not option:
                return _error("select needs an option")
            return _Prepared(
                Action.select(str(ref), str(option)),
                value=self._bind_literal(run, str(option)),
                ref=str(ref),
            )
        return self._prepare_fill(run, str(ref), args)

    def _prepare_fill(self, run: _Run, ref: str, args: dict[str, Any]) -> _Prepared | _Outcome:
        text, secret, param = args.get("text"), args.get("secret"), args.get("param")
        if sum(v is not None for v in (text, secret, param)) != 1:
            return _error("fill needs exactly one of text, secret or param")
        if param is not None:
            if param not in run.values:
                return _error(
                    f"unknown param {param!r} (available: {', '.join(run.values) or 'none'})"
                )
            return _Prepared(
                Action.fill(ref, text=run.values[str(param)]),
                value=ParamRef(name=str(param)),
                ref=ref,
            )
        if secret is not None:
            if secret not in run.task.secrets:
                names = ", ".join(run.task.secrets) or "none"
                return _error(f"unknown secret {secret!r} (available: {names})")
            return _Prepared(
                Action.fill(ref, secret=str(secret)), value=SecretRef(name=str(secret)), ref=ref
            )
        assert text is not None
        return _Prepared(
            Action.fill(ref, text=str(text)), value=self._bind_literal(run, str(text)), ref=ref
        )

    def _wait(self, run: _Run, args: dict[str, Any]) -> _Outcome:
        gated = self._gateway.act(Action.wait(int(args.get("ms") or 500)))
        if not gated.executed:
            return _blocked(gated.decision)
        assert gated.result is not None
        return _Outcome(f"OK: {gated.result.detail}", observation=self._observe(run))

    def _wait_for(self, run: _Run, args: dict[str, Any], reason: str) -> _Outcome:
        text = args.get("text")
        if not text:
            return _error("wait_for needs text")
        timeout_ms = int(args.get("ms") or 5000)
        found = self._surface.wait_for_text(str(text), timeout_ms=timeout_ms)
        self._log.emit("wait_for", outcome="ok" if found else "timeout", reason=str(text)[:120])
        if not found:
            return _error(f"{str(text)!r} did not appear within {timeout_ms} ms")
        obs = self._observe(run)
        run.steps.append(
            RecordedStep(
                id=f"s{len(run.steps) + 1}",
                kind="wait_for",
                description=reason,
                expect_text=str(text),
                risk=RiskClass.READ,
                after=ObservationDigest.from_observation(obs),
            )
        )
        return _Outcome(f"OK: {str(text)!r} is on the page", observation=obs)

    # --- extract ----------------------------------------------------------------------------

    def _extract(self, run: _Run, args: dict[str, Any]) -> _Outcome:
        name, value = args.get("name"), args.get("value")
        requested = ", ".join(run.task.outputs) or "none"
        if not name:
            return _error(f"extract needs a name (requested outputs: {requested})")
        if name not in run.task.outputs:
            return _error(f"unknown output {name!r} (requested outputs: {requested})")
        if not value:
            return _error("extract needs a value: the exact text as displayed")
        anchor = str(args["anchor_text"]) if args.get("anchor_text") else None
        try:
            bundle = self._surface.harvest_value(str(value), anchor_text=anchor, params=run.values)
            resolved = self._surface.resolve(bundle, run.values)
            read = self._surface.read_text(resolved.ref) if resolved.ref else ""
        except SurfaceError as exc:
            return _error(str(exc))
        if _squash(read) != _squash(str(value)):
            return _error(
                f"read back {read!r} instead of {str(value)!r}: this locator is not reliable, "
                "add or change anchor_text"
            )
        run.steps.append(
            RecordedStep(
                id=f"s{len(run.steps) + 1}",
                kind="extract",
                description=(str(args.get("reason") or f"read {name}"))[:200],
                locator=bundle,
                output=str(name),
                risk=RiskClass.READ,
            )
        )
        run.outputs[str(name)] = str(value)
        return _Outcome(f"OK: recorded output {name} = {value}")


__all__ = ["DiscoveryLoop", "StepKind", "compact"]
