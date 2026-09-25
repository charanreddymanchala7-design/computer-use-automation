"""How a run reads on a terminal: one marker per line, then a RESULT block.

Every line starts with a text marker, so nothing depends on colour: ``[ok]`` went as planned,
``[!!]`` needs attention (a recovery, a person, drift), ``[xx]`` failed, ``[--]`` is information.
Colour, when allowed, is added on top of the marker and never replaces it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cua.artifact import Capability
from cua.redact import Redactor
from cua.result import ReplayResult, Status

_MARKERS = {"ok": "[ok]", "warn": "[!!]", "fail": "[xx]", "info": "[--]"}
_COLORS = {"ok": "32", "warn": "33", "fail": "31", "info": "2"}


def marker(kind: str, *, color: bool) -> str:
    text = _MARKERS[kind]
    return f"\x1b[{_COLORS[kind]}m{text}\x1b[0m" if color else text


def mark_up(kind: str, text: str, *, color: bool) -> str:
    return f"{marker(kind, color=color)} {text}"


def mask_inputs(inputs: Mapping[str, Any], sensitive: Iterable[str] = ()) -> dict[str, str]:
    """What may be shown of the caller's inputs: identifiers keep their last two characters and
    sensitive values are not shown at all."""
    hidden = set(sensitive)
    return {
        name: "<withheld>" if name in hidden else Redactor.mask_id(str(value))
        for name, value in inputs.items()
    }


def progress_lines(
    events: Iterable[Mapping[str, Any]], capability: Capability, *, color: bool
) -> list[str]:
    """The run as it happened, from its own event log."""
    steps = {s.id: s.description for s in capability.steps}
    lines: list[str] = []

    def add(kind: str, text: str) -> None:
        lines.append(mark_up(kind, text, color=color))

    for event in events:
        name = event.get("event")
        step = event.get("step", "")
        if name == "replay_start":
            add("info", f"replaying {event['capability']} (no model is involved)")
        elif name == "step_ok":
            add("ok", f"{step} {steps.get(step, '')}".rstrip())
        elif name == "recovery":
            add("warn", f"{step} recovered from '{event.get('reason')}' by {event.get('outcome')}")
        elif name == "intervention_raised":
            add("warn", f"{step} stuck ({event.get('reason')}): a person was asked to help")
        elif name == "control_taken":
            add("warn", f"a person took control ({event.get('actor')})")
        elif name == "control_returned":
            count = event.get("actions", 0)
            add(
                "info",
                f"the person did {count} action{'s' if count != 1 else ''} (kept in the run log)",
            )
        elif name == "control_resumed":
            add("ok", "control handed back; the page was re-checked and the run continues")
        elif name in ("intervention_aborted", "intervention_timed_out"):
            how = "aborted" if name == "intervention_aborted" else "timed out"
            add("fail", f"the person did not finish the handoff: {how}")
        elif name == "action_blocked":
            add("fail", f"{step} blocked by policy ({event.get('reason')})")
    return lines


def render_result(result: ReplayResult, *, color: bool = False) -> str:
    """The RESULT block: what happened, in the order a caller needs it."""
    kind = {
        Status.SUCCESS: "ok",
        Status.BUSINESS_OUTCOME: "warn",
        Status.ESCALATED: "warn",
        Status.HARD_FAILURE: "fail",
    }[result.status]
    rows: list[tuple[str, str]] = [("outcome", result.outcome_code)]
    if result.message:
        rows.append(("message", result.message))
    if result.outputs is not None:
        rows.extend(("output", f"{name} = {value}") for name, value in result.outputs.items())
    if result.failed_step:
        rows.append(("failed step", result.failed_step))
        rows.append(("expected", result.expected or ""))
        rows.append(("observed", result.observed or ""))
    if result.escalation:
        rows.append(
            ("needs a person", f"{result.escalation.reason} at {result.escalation.step_id}")
        )
    rows.extend(
        ("recovered", f"{r.rule_id} at {r.step_id} ({r.kind} x{r.attempts})")
        for r in result.recoveries
    )
    rows.extend(
        ("drift", f"{d.step_id} found by fallback strategy {d.strategy_index} ({d.strategy_kind})")
        for d in result.degraded
    )
    for i in result.interventions:
        who = f" by {i.taken_by}" if i.taken_by else ""
        rows.append(
            (
                "handoff",
                f"{i.request_id} at {i.step_id}: {i.outcome}{who} ({i.duration_ms / 1000:.1f} s)",
            )
        )
        rows.extend(("human did", action) for action in i.actions)
    if result.evidence.screenshot:
        rows.append(("evidence", f"{result.evidence.screenshot}, {result.evidence.aria_snapshot}"))
    rows.append(("duration", f"{result.duration_ms / 1000:.1f} s"))
    width = max(len(name) for name, _ in rows)
    head = f"{marker(kind, color=color)} RESULT  {result.status.value}  (exit {result.exit_code})"
    return "\n".join([head, *(f"    {name:<{width}}  {value}" for name, value in rows)])
