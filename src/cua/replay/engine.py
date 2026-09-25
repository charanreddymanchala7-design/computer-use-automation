"""Deterministic replay: run a capability with typed inputs and no model in the loop.

The steps, the way each element is found, what to wait for and what counts as success all come
from the artifact. Every action passes the same gateway as discovery, so policy and audit
cannot be skipped. What the engine adds is judgement about *runtime* conditions: inputs and
secrets are checked before the app is touched, elements are waited for (never slept for),
each expectation and the final checkpoint are verified, and every failure names the step, what
was expected and what was seen. A lower-ranked locator strategy matching is reported as drift.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import jsonschema

from cua.artifact import (
    ActionKind,
    Capability,
    Detector,
    ErrorClass,
    ErrorRule,
    Expectation,
    LiteralValue,
    ParamRef,
    SecretRef,
    Step,
    fill_placeholders,
)
from cua.control import build_request
from cua.control.handoff import Handoff
from cua.evlog import EventLog, write_redacted_json
from cua.gateway import ActionGateway, Decision, Reason
from cua.replay.match import url_matches
from cua.result import (
    BusinessOutcome,
    CuaError,
    DegradedLocator,
    EscalationRequired,
    EvidenceRefs,
    HardFailure,
    InterventionRecord,
    RecoverableCondition,
    RecoveryRecord,
    ReplayResult,
    Status,
)
from cua.surface import Action, LocatingSurface, LocatorNotFound, Resolved, SurfaceError

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SEEN_LIMIT = 200
# A step that is stuck for one of these reasons is worth a person's time. Everything else is an
# answer (business outcome), a known crash or a broken input, and asking someone would not help.
_HANDOFF_FAILURES = frozenset({"locator_not_found", "expectation_failed", "checkpoint_failed"})


@dataclass(frozen=True)
class ReplayLimits:
    step_timeout_s: float = 8.0  # how long to keep looking for an element
    poll_ms: int = 100  # how often
    run_timeout_s: float = 120.0
    slow_threshold_ms: int = 1000  # waiting at least this long is worth reporting
    max_backoff_ms: int = 10_000
    max_interventions: int = 2  # how many times one run may ask a person, then it just fails


@dataclass
class _Ctx:
    capability: Capability
    inputs: dict[str, Any]
    values: dict[str, str]  # non-sensitive inputs, for placeholders
    sensitive: dict[str, str]  # sensitive input name -> the secret name it is typed under
    started: float
    outputs: dict[str, str] = field(default_factory=dict)
    degraded: list[DegradedLocator] = field(default_factory=list)
    recoveries: list[RecoveryRecord] = field(default_factory=list)
    declared: list[tuple[str, str]] = field(default_factory=list)  # (kind, message) of this step
    attempts: dict[tuple[str, str], int] = field(default_factory=dict)  # (rule, step) -> tries
    escalations: int = 0
    interventions: list[InterventionRecord] = field(default_factory=list)


def _describe_schema_error(error: jsonschema.ValidationError, schema: Mapping[str, Any]) -> str:
    """What was wrong, by name and rule, never by value (values may be sensitive)."""
    instance = error.instance if isinstance(error.instance, dict) else {}
    if error.validator == "required":
        missing = [n for n in schema.get("required", []) if n not in instance]
        return "missing " + ", ".join(missing or ["a required value"])
    if error.validator == "additionalProperties":
        extra = sorted(set(instance) - set(schema.get("properties", {})))
        return "unexpected property " + ", ".join(extra or ["(unknown)"])
    path = ".".join(str(p) for p in error.absolute_path) or "value"
    return f"{path}: failed {error.validator}"


def _first_error(schema: Mapping[str, Any], instance: object) -> jsonschema.ValidationError | None:
    errors = sorted(
        jsonschema.Draft202012Validator(dict(schema)).iter_errors(instance),
        key=lambda e: [str(p) for p in e.absolute_path],
    )
    return errors[0] if errors else None


class ReplayEngine:
    def __init__(
        self,
        surface: LocatingSurface,
        gateway: ActionGateway,
        log: EventLog,
        *,
        base_url: str,
        secrets: Mapping[str, str] | None = None,
        evidence_dir: Path | None = None,
        limits: ReplayLimits | None = None,
        clock: Any = time.monotonic,
        run_id: str = "run",
        handoff: Handoff | None = None,
    ) -> None:
        self._surface = surface
        self._gateway = gateway
        self._log = log
        self._base = base_url
        self._secrets = dict(secrets or {})
        self._evidence_dir = evidence_dir
        self._limits = limits or ReplayLimits()
        self._clock = clock
        self._run_id = run_id
        self._handoff = handoff
        self._redact = log.redactor.redact_text

    # --- the run -------------------------------------------------------------------------------

    def run(self, capability: Capability, inputs: Mapping[str, Any]) -> ReplayResult:
        started = self._clock()
        sensitive = capability.sensitive_params()
        ctx = _Ctx(
            capability=capability,
            inputs=dict(inputs),
            values={k: str(v) for k, v in inputs.items() if k not in sensitive},
            sensitive={k: f"param:{k}" for k in inputs if k in sensitive},
            started=started,
        )
        self._log.emit(
            "replay_start", capability=f"{capability.id}@{capability.capability_version}"
        )
        if self._handoff is not None:
            self._handoff.begin()
        try:
            self._validate_inputs(ctx)
            self._prepare_secrets(ctx)
            for step in capability.steps:
                self._check_run_timeout(ctx, step)
                self._run_step(ctx, step)
            self._await(ctx, "checkpoint", self._checkpoint_expectation(ctx), "checkpoint_failed")
            outputs = self._validated_outputs(ctx)
            result = ReplayResult(
                status=Status.SUCCESS,
                outcome_code="completed",
                capability_id=capability.id,
                capability_version=capability.capability_version,
                run_id=self._run_id,
                outputs=outputs,
                recoveries=ctx.recoveries,
                degraded=ctx.degraded,
                interventions=ctx.interventions,
                duration_ms=self._elapsed_ms(ctx),
            )
        except CuaError as exc:
            result = ReplayResult.from_error(
                exc,
                capability_id=capability.id,
                capability_version=capability.capability_version,
                run_id=self._run_id,
                duration_ms=self._elapsed_ms(ctx),
                recoveries=ctx.recoveries,
                degraded=ctx.degraded,
                interventions=ctx.interventions,
                evidence=self._capture_evidence(exc),
            )
        finally:
            if self._handoff is not None:
                self._handoff.end()
        self._log.emit(
            "replay_end",
            status=result.status.value,
            outcome=result.outcome_code,
            duration_ms=result.duration_ms,
        )
        return result

    def _elapsed_ms(self, ctx: _Ctx) -> int:
        return max(0, int((self._clock() - ctx.started) * 1000))

    def _check_run_timeout(self, ctx: _Ctx, step: Step) -> None:
        elapsed = self._clock() - ctx.started
        if elapsed > self._limits.run_timeout_s:
            raise HardFailure(
                "timeout",
                step_id=step.id,
                expected=f"the run to finish within {self._limits.run_timeout_s:.0f} s",
                observed=f"{elapsed:.0f} s elapsed",
            )

    # --- before the app is touched -------------------------------------------------------------

    def _validate_inputs(self, ctx: _Ctx) -> None:
        error = _first_error(ctx.capability.inputs, ctx.inputs)
        if error is not None:
            raise HardFailure(
                "invalid_input",
                step_id="inputs",
                expected="inputs that satisfy the capability's input schema",
                observed=_describe_schema_error(error, ctx.capability.inputs),
            )

    def _prepare_secrets(self, ctx: _Ctx) -> None:
        needed = ctx.capability.required_secrets()
        missing = [name for name in needed if name not in self._secrets]
        if missing:
            raise HardFailure(
                "missing_secret",
                step_id="setup",
                expected="secrets configured: " + ", ".join(needed),
                observed="missing: " + ", ".join(missing),
            )
        typed_under = {name: str(ctx.inputs[param]) for param, name in ctx.sensitive.items()}
        self._surface.add_secrets({**self._secrets, **typed_under})

    def _validated_outputs(self, ctx: _Ctx) -> dict[str, Any]:
        error = _first_error(ctx.capability.outputs, ctx.outputs)
        if error is not None:
            raise HardFailure(
                "output_invalid",
                step_id="outputs",
                expected="outputs that satisfy the capability's output schema",
                observed=_describe_schema_error(error, ctx.capability.outputs),
            )
        return dict(ctx.outputs)

    # --- one step ------------------------------------------------------------------------------

    def _run_step(self, ctx: _Ctx, step: Step) -> None:
        """Run one step. If it gets stuck and a person can help, ask, and carry on afterwards.

        What a person's help may repeat depends on where the step stopped. Before its action ran
        (an element that was missing, a session that had expired) the whole step is tried again.
        Once the action has run, only the *check* is: pressing Submit a second time because the
        confirmation page was slow could submit twice. Nothing that can be handed to a person is
        raised between an action running and the step moving on to its check."""
        strategy: str | None = None
        performed = False
        while True:
            try:
                if not performed:
                    strategy = self._perform(ctx, step)
                    performed = True
                self._verify(ctx, step)
                extra: dict[str, Any] = {"strategy": strategy} if strategy else {}
                self._log.emit("step_ok", step=step.id, **extra)
                return
            except CuaError as exc:
                if not self._should_hand_off(ctx, exc):
                    raise
                self._hand_off(ctx, step, exc)

    def _perform(self, ctx: _Ctx, step: Step) -> str | None:
        self._log.emit("step_start", step=step.id, action=step.action.value)
        self._install_dialog_policy(ctx, step)
        strategy: str | None = None
        if step.action is ActionKind.NAVIGATE:
            acting = self._clock()
            self._act(ctx, step, Action.navigate(self._url(ctx, step)))
            if (self._clock() - acting) * 1000 >= self._limits.slow_threshold_ms:
                self._record_slow(ctx, step.id, 1)
        elif step.action is ActionKind.WAIT_FOR:
            assert step.expect is not None
            self._await(ctx, step.id, step.expect, "expectation_failed", report_slow=False)
        elif step.action is ActionKind.EXTRACT:
            found = self._locate(ctx, step)
            strategy = found.strategy_kind
            assert found.ref is not None
            assert step.output is not None
            ctx.outputs[step.output] = self._surface.read_text(found.ref)
        else:
            target = self._locate(ctx, step) if step.locator else None
            acting = self._clock()
            self._act(ctx, step, self._action_for(ctx, step, target))
            # the browser may absorb a slow page while an action settles: no element was
            # "missing", so the action's own duration is what shows it
            if (self._clock() - acting) * 1000 >= self._limits.slow_threshold_ms:
                self._record_slow(ctx, step.id, 1)
            strategy = target.strategy_kind if target else None
        return strategy

    def _verify(self, ctx: _Ctx, step: Step) -> None:
        if step.expect is not None and step.action is not ActionKind.WAIT_FOR:
            self._await(ctx, step.id, step.expect, "expectation_failed")

    # --- asking a person for help ---------------------------------------------------------------

    def _should_hand_off(self, ctx: _Ctx, exc: CuaError) -> bool:
        if self._handoff is None or len(ctx.interventions) >= self._limits.max_interventions:
            return False
        if isinstance(exc, EscalationRequired):
            # a held irreversible action is approved through the gateway, not by taking the page
            return exc.reason != "confirmation_required"
        return isinstance(exc, HardFailure) and exc.code in _HANDOFF_FAILURES

    def _hand_off(self, ctx: _Ctx, step: Step, exc: CuaError) -> None:
        """Give the live session to a person. Returns once they hand it back; otherwise the run
        ends as escalated, with the reason it was stuck."""
        assert self._handoff is not None
        if isinstance(exc, EscalationRequired):
            code, request_id = exc.reason, exc.request_id
            expected, observed = exc.expected or "the step to proceed", exc.observed or self._seen()
        else:
            assert isinstance(exc, HardFailure)
            ctx.escalations += 1
            code, request_id = exc.code, f"ir_{self._run_id}_{ctx.escalations}"
            expected, observed = exc.expected, exc.observed
        number = len(ctx.interventions) + 1
        request = build_request(
            request_id=request_id,
            capability_id=ctx.capability.id,
            goal=ctx.capability.title,
            step_id=step.id,
            outcome_code=code,
            expected=expected,
            observed=observed,
            url=self._surface.current_url(),
            screenshot=self._snapshot(exc, f"intervention-{number}"),
            redactor=self._log.redactor,
            now=self._clock(),
        )
        handed = self._handoff.escalate(request)
        ctx.interventions.append(
            InterventionRecord(
                request_id=request_id,
                step_id=step.id,
                reason_code=request.reason_code,
                outcome=handed.outcome,
                taken_by=handed.taken_by,
                duration_ms=handed.duration_ms,
            )
        )
        if handed.outcome != "handed_back":
            raise EscalationRequired(code, request_id=request_id, step_id=step.id)

    def _url(self, ctx: _Ctx, step: Step) -> str:
        template = step.url_template or ""

        def encoded(match: re.Match[str]) -> str:
            return quote(ctx.values[match.group(1)], safe="")

        path = _PLACEHOLDER.sub(encoded, template)
        if urlsplit(path).scheme in ("http", "https"):
            return path
        return self._base.rstrip("/") + "/" + path.lstrip("/")

    def _text_of(self, ctx: _Ctx, step: Step) -> str:
        value = step.value
        if isinstance(value, ParamRef):
            return str(ctx.inputs[value.name])
        if isinstance(value, LiteralValue):
            return str(value.value)
        raise HardFailure(
            "invalid_step",
            step_id=step.id,
            expected="a value to type",
            observed="the step has no usable value",
        )

    def _action_for(self, ctx: _Ctx, step: Step, resolved: Resolved | None) -> Action:
        kind = step.action
        if kind is ActionKind.PRESS:
            return Action.press(self._text_of(ctx, step))
        assert resolved is not None
        if kind is ActionKind.CLICK:
            if resolved.ref is not None:
                return Action.click(resolved.ref)
            assert resolved.coordinates is not None
            return Action.click_at(*resolved.coordinates)
        if resolved.ref is None:
            raise HardFailure(
                "invalid_step",
                step_id=step.id,
                expected=f"an element to {kind.value}",
                observed="only a screen coordinate was available",
            )
        if kind is ActionKind.FILL:
            value = step.value
            if isinstance(value, SecretRef):
                return Action.fill(resolved.ref, secret=value.name)
            if isinstance(value, ParamRef) and value.name in ctx.sensitive:
                return Action.fill(resolved.ref, secret=ctx.sensitive[value.name])
            return Action.fill(resolved.ref, text=self._text_of(ctx, step))
        return Action.select(resolved.ref, self._text_of(ctx, step))

    # --- acting --------------------------------------------------------------------------------

    def _act(self, ctx: _Ctx, step: Step, action: Action) -> None:
        gated = self._gateway.act(action, declared_risk=step.risk_class, step=step.id)
        if not gated.executed:
            raise self._blocked(step, gated.decision)
        assert gated.result is not None
        result = gated.result
        for event in result.dialogs:
            if (event.kind, event.message) not in ctx.declared:
                raise self._fail(
                    step.id,
                    "unexpected_dialog",
                    expected="no dialog other than those the capability declares",
                    observed=f"{event.kind}: {event.message}",
                )
        if not result.ok:
            raise self._fail(
                step.id,
                "action_failed",
                expected=f"{step.action.value} succeeds",
                observed=result.error or result.detail,
            )

    def _blocked(self, step: Step, decision: Decision) -> CuaError:
        if decision.reason is Reason.CONFIRMATION_REQUIRED:
            return EscalationRequired(
                "confirmation_required",
                request_id=decision.confirmation_id or "cf_unknown",
                step_id=step.id,
            )
        detail = f" ({decision.detail})" if decision.detail else ""
        return HardFailure(
            "blocked_by_policy",
            step_id=step.id,
            expected="the step is permitted by policy",
            observed=f"{decision.reason.value}{detail}",
        )

    def _install_dialog_policy(self, ctx: _Ctx, step: Step) -> None:
        """A declared dialog gets its declared answer; anything else is dismissed, never clicked
        through, and reported after the action."""
        answers: dict[tuple[str, str], bool] = {
            (d.kind, fill_placeholders(d.message, ctx.values)): d.action == "accept"
            for d in step.dialogs
        }
        ctx.declared = list(answers)

        def decide(kind: str, message: str) -> bool:
            return answers.get((kind, message), False)

        self._surface.set_dialog_policy(decide)

    # --- finding elements ----------------------------------------------------------------------

    def _locate(self, ctx: _Ctx, step: Step) -> Resolved:
        """Walk the ranked bundle, waiting for the page rather than sleeping, and recognising
        what the application is telling us while the element is missing."""
        assert step.locator is not None
        deadline = self._clock() + self._limits.step_timeout_s
        misses, first_miss, last = 0, 0.0, ""
        while True:
            try:
                resolved = self._surface.resolve(step.locator, ctx.values)
            except LocatorNotFound as exc:
                attempts = ", ".join(f"{a.kind}: {a.outcome}" for a in exc.attempts)
                last = attempts or str(exc)
            except SurfaceError as exc:
                raise self._fail(
                    step.id,
                    "locator_error",
                    expected=f"{step.locator.description}",
                    observed=str(exc),
                ) from exc
            else:
                self._note_slow(ctx, step.id, misses, first_miss)
                if resolved.strategy_index > 0:
                    ctx.degraded.append(
                        DegradedLocator(
                            step_id=step.id,
                            strategy_index=resolved.strategy_index,
                            strategy_kind=resolved.strategy_kind,
                        )
                    )
                return resolved
            if misses == 0:
                first_miss = self._clock()
            misses += 1
            wanted = f"element {step.locator.description}"
            if self._diagnose(ctx, step.id, wanted):
                continue  # a known condition was dealt with: look again straight away
            if self._clock() >= deadline:
                raise self._fail(
                    step.id,
                    "locator_not_found",
                    expected=wanted,
                    observed=f"no strategy matched ({last})",
                )
            self._surface.pause(self._limits.poll_ms)

    # --- expectations and the checkpoint -------------------------------------------------------

    def _checkpoint_expectation(self, ctx: _Ctx) -> Expectation:
        check = ctx.capability.checkpoint
        return Expectation(
            url_pattern=check.url_pattern,
            text_present=list(check.text_present),
            timeout_ms=check.timeout_ms,
        )

    def _unmet(self, ctx: _Ctx, expect: Expectation) -> list[str]:
        def fill(text: str) -> str:
            return fill_placeholders(text, ctx.values)

        unmet: list[str] = []
        for text in expect.text_present:
            if not self._surface.wait_for_text(fill(text), timeout_ms=0):
                unmet.append(f"text {fill(text)!r} present")
        for text in expect.text_absent:
            if self._surface.wait_for_text(fill(text), timeout_ms=0):
                unmet.append(f"text {fill(text)!r} absent")
        if expect.url_pattern and not url_matches(
            fill(expect.url_pattern), self._surface.current_url()
        ):
            unmet.append(f"url matching {fill(expect.url_pattern)}")
        if expect.element is not None:
            try:
                self._surface.resolve(expect.element, ctx.values)
            except LocatorNotFound:
                unmet.append(f"element {expect.element.description}")
        return unmet

    def _await(
        self,
        ctx: _Ctx,
        step_id: str,
        expect: Expectation,
        code: str,
        *,
        report_slow: bool = True,
    ) -> None:
        deadline = self._clock() + expect.timeout_ms / 1000
        misses, first_miss = 0, 0.0
        while True:
            unmet = self._unmet(ctx, expect)
            if not unmet:
                if report_slow:  # a wait_for is the capability asking to wait: not a surprise
                    self._note_slow(ctx, step_id, misses, first_miss)
                return
            if misses == 0:
                first_miss = self._clock()
            misses += 1
            wanted = " and ".join(unmet)
            if self._diagnose(ctx, step_id, wanted):
                continue
            if self._clock() >= deadline:
                raise self._fail(step_id, code, expected=wanted, observed=self._seen())
            self._surface.pause(self._limits.poll_ms)

    # --- runtime conditions ---------------------------------------------------------------------

    def _note_slow(self, ctx: _Ctx, step_id: str, misses: int, first_miss: float) -> None:
        """A page that made us wait is reported, not failed: transient slowness is expected."""
        if misses == 0:
            return
        waited_ms = (self._clock() - first_miss) * 1000
        if waited_ms >= self._limits.slow_threshold_ms:
            self._record_slow(ctx, step_id, misses)

    def _record_slow(self, ctx: _Ctx, step_id: str, attempts: int) -> None:
        if any(r.rule_id == "slow_response" and r.step_id == step_id for r in ctx.recoveries):
            return  # once per step is enough
        ctx.recoveries.append(
            RecoveryRecord(
                rule_id="slow_response", step_id=step_id, kind="wait_retry", attempts=attempts
            )
        )

    def _detected(self, ctx: _Ctx, detector: Detector) -> bool:
        """Any one signal is enough (dialog signals are not evaluated: see the report)."""
        for text in detector.text_present:
            if self._surface.wait_for_text(fill_placeholders(text, ctx.values), timeout_ms=0):
                return True
        if detector.url_pattern and url_matches(
            fill_placeholders(detector.url_pattern, ctx.values), self._surface.current_url()
        ):
            return True
        if detector.element is not None:
            try:
                self._surface.resolve(detector.element, ctx.values)
            except LocatorNotFound:
                return False
            return True
        return False

    def _diagnose(self, ctx: _Ctx, step_id: str, wanted: str) -> bool:
        """What is the application telling us? True means a recovery ran and the caller should
        look again; a business outcome, a hard failure or an escalation is raised."""
        rule = next((r for r in ctx.capability.error_map if self._detected(ctx, r.detect)), None)
        if rule is None:
            return False
        self._log.emit("condition", step=step_id, reason=rule.id, outcome=rule.classification.value)
        if rule.classification is ErrorClass.BUSINESS_OUTCOME:
            assert rule.outcome_code is not None
            raise BusinessOutcome(rule.outcome_code, message=rule.message)
        if rule.classification is ErrorClass.HARD_FAILURE:
            assert rule.code is not None
            if rule.escalate:
                ctx.escalations += 1
                raise EscalationRequired(
                    rule.code,
                    request_id=f"ir_{self._run_id}_{ctx.escalations}",
                    step_id=step_id,
                    expected=wanted,
                    observed=self._seen(),
                )
            raise self._fail(
                step_id,
                rule.code,
                expected=wanted,
                observed=f"condition '{rule.id}' detected: {self._seen()}",
                message=rule.message,
            )
        self._recover(ctx, step_id, rule)
        return True

    def _recover(self, ctx: _Ctx, step_id: str, rule: ErrorRule) -> None:
        """One bounded attempt to clear a known condition. Budgets are per rule and per step."""
        recovery = rule.recovery
        assert recovery is not None
        key = (rule.id, step_id)
        tried = ctx.attempts.get(key, 0)
        if tried >= recovery.max_attempts:
            raise RecoverableCondition(rule.id, step_id=step_id, kind=recovery.kind)
        ctx.attempts[key] = tried + 1
        if recovery.kind == "dismiss":
            assert recovery.locator is not None
            try:
                target = self._surface.resolve(recovery.locator, ctx.values)
            except LocatorNotFound as exc:
                raise RecoverableCondition(rule.id, step_id=step_id, kind=recovery.kind) from exc
            action = (
                Action.click(target.ref)
                if target.ref is not None
                else Action.click_at(*(target.coordinates or (0.0, 0.0)))
            )
            gated = self._gateway.act(action, step=step_id)
            if not gated.executed:
                raise RecoverableCondition(rule.id, step_id=step_id, kind=recovery.kind)
        else:
            backoff = min(recovery.backoff_ms * 2**tried, self._limits.max_backoff_ms)
            self._surface.pause(backoff)
        self._log.emit("recovery", step=step_id, reason=rule.id, outcome=recovery.kind)
        self._record_recovery(ctx, rule.id, step_id, recovery.kind)

    def _record_recovery(self, ctx: _Ctx, rule_id: str, step_id: str, kind: str) -> None:
        for index, record in enumerate(ctx.recoveries):
            if (record.rule_id, record.step_id, record.kind) == (rule_id, step_id, kind):
                ctx.recoveries[index] = record.model_copy(update={"attempts": record.attempts + 1})
                return
        ctx.recoveries.append(
            RecoveryRecord(
                rule_id=rule_id,
                step_id=step_id,
                kind="dismiss" if kind == "dismiss" else "wait_retry",
                attempts=1,
            )
        )

    # --- failures and evidence -----------------------------------------------------------------

    def _seen(self) -> str:
        """What the page shows right now, in a few words."""
        try:
            frames = self._surface.observe().frames
        except SurfaceError:
            return "the page could not be read"
        return " | ".join(f.text for f in frames if f.text)[:_SEEN_LIMIT] or "an empty page"

    def _fail(
        self,
        step_id: str,
        code: str,
        *,
        expected: str,
        observed: str,
        message: str | None = None,
    ) -> HardFailure:
        return HardFailure(
            code,
            step_id=step_id,
            expected=self._redact(expected),
            observed=self._redact(observed) or "nothing",
            message=message,
        )

    def _capture_evidence(self, exc: CuaError) -> EvidenceRefs:
        """A screenshot and a redacted page snapshot, for anything a person may have to debug."""
        if not isinstance(exc, HardFailure | EscalationRequired | RecoverableCondition):
            return EvidenceRefs()
        if self._write_snapshot(exc, "failure.png", "page.json") is None:
            return EvidenceRefs(log=self._log.path.name) if self._evidence_dir else EvidenceRefs()
        return EvidenceRefs(
            screenshot="failure.png", aria_snapshot="page.json", log=self._log.path.name
        )

    def _snapshot(self, exc: CuaError, stem: str) -> str | None:
        """What a person is shown when asked for help: the screenshot's path under the evidence
        directory, or ``None`` when no evidence is being kept."""
        written = self._write_snapshot(exc, f"{stem}.png", f"{stem}.json")
        return None if written is None else f"{self._run_id}/{stem}.png"

    def _write_snapshot(self, exc: CuaError, png: str, document: str) -> Path | None:
        if self._evidence_dir is None:
            return None
        try:
            obs = self._surface.observe()
        except SurfaceError:
            return None
        folder = self._evidence_dir / self._run_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / png).write_bytes(obs.screenshot)
        write_redacted_json(
            folder / document,
            {
                "failure": {"code": str(exc), "type": type(exc).__name__},
                "url": urlsplit(obs.url).path,
                "frames": [
                    {"name": f.name, "url": urlsplit(f.url).path, "text": f.text, "aria": f.aria}
                    for f in obs.frames
                ],
            },
            self._log.redactor,
        )
        return folder / png
