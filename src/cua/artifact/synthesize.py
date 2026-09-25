"""From a recorded discovery run to a reusable capability.

The recording is one run's worth of examples. The capability is what remains true for *any*
run: the inputs are the parameters that were actually used, navigation is a portable path (the
host belongs to the tenant, not the capability), text that came from the caller is turned back
into ``{placeholders}``, expectations and the checkpoint are built from labels, never from the
values a page happened to show, and a dialog the app always raises is declared so replay can
accept it and treat any other as unexpected.

Nothing specific to the discovery run may survive: no example member number, no value that was
read, no name or ID from one member's page, no host.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from urllib.parse import urlsplit

from pydantic import ValidationError

from cua.agent.recording import ObservationDigest, RecordedRun, RecordedStep
from cua.agent.tools import DiscoveryTask
from cua.artifact.schema import (
    ActionKind,
    AncestorAnchorLocator,
    Capability,
    Checkpoint,
    ErrorRule,
    Expectation,
    ExpectedDialog,
    ParamRef,
    Provenance,
    RiskClass,
    SecretRef,
    Step,
    Target,
    parameterize,
)

_LABEL_LINE = re.compile(r"^([A-Za-z][A-Za-z .#&/'-]{1,30}):\s*(.*)$")
_DIGIT_RUN = re.compile(r"\d{3,}")
_EXPECT_LIMIT = 2
_CHECKPOINT_LIMIT = 3
_WAIT_TIMEOUT_MS = 10_000


class SynthesisError(ValueError):
    """The run cannot become a trustworthy capability; the message says why."""


@dataclass(frozen=True)
class CapabilitySpec:
    """What a person decides about a capability that a recording cannot know."""

    id: str
    title: str
    description: str
    target: Target
    version: str = "1.0.0"
    error_map: Sequence[ErrorRule] = ()


@dataclass
class Synthesis:
    capability: Capability
    warnings: list[str] = field(default_factory=list)


def _path(url: str) -> str:
    """A portable address: the path and query only. The host belongs to the tenant."""
    parts = urlsplit(url)
    return (parts.path or "/") + (f"?{parts.query}" if parts.query else "")


def _lines(digest: ObservationDigest | None) -> Iterator[str]:
    if digest is None:
        return
    for frame in digest.frames:
        for raw in frame.text.split("\n"):
            line = " ".join(raw.split())
            if line:
                yield line


def _is_static(line: str, dynamic: Sequence[str]) -> bool:
    """A line that is a label or a heading, not data: no digits runs, money or example values."""
    if not 3 <= len(line) <= 60:
        return False
    if sum(ch.isalpha() for ch in line) < 3:
        return False
    if "$" in line or _DIGIT_RUN.search(line):
        return False
    return not any(value and value in line for value in dynamic)


def _scrub(text: str, values: Mapping[str, str], outputs: Mapping[str, str]) -> str:
    """Model-written prose can quote example data; replace it with its placeholder."""
    text = parameterize(text, values)
    for name, value in outputs.items():
        if len(value) >= 3:
            text = text.replace(value, f"<{name}>")
    return text


def _expectation(
    step: RecordedStep,
    previous: ObservationDigest | None,
    dynamic: Sequence[str],
    values: Mapping[str, str],
    fallback_url: str,
    warnings: list[str],
) -> Expectation | None:
    if step.kind == "wait_for":
        assert step.expect_text is not None
        return Expectation(
            text_present=[parameterize(step.expect_text, values)], timeout_ms=_WAIT_TIMEOUT_MS
        )
    if step.risk is RiskClass.READ:
        return None  # the next step's locator is the check; reads need no text expectation
    seen = set(_lines(previous))
    fresh = [line for line in _lines(step.after) if line not in seen and _is_static(line, dynamic)]
    if fresh:
        return Expectation(text_present=fresh[:_EXPECT_LIMIT])
    warnings.append(
        f"step {step.id} changes something but showed no new label to expect: its expectation "
        "is weak (the page URL); add a wait_for after it"
    )
    return Expectation(url_pattern=fallback_url)


def _dialogs(step: RecordedStep, values: Mapping[str, str]) -> list[ExpectedDialog]:
    return [
        ExpectedDialog.model_validate(
            {
                "kind": d.kind,
                "message": parameterize(d.message, values),
                "action": "accept" if d.accepted else "dismiss",
            }
        )
        for d in step.dialogs
    ]


def _checkpoint(run: RecordedRun, values: Mapping[str, str], dynamic: Sequence[str]) -> Checkpoint:
    final = run.final
    final_text = " ".join(_lines(final)).lower()
    param_lines: list[str] = []
    labels: list[str] = []
    for line in _lines(final):
        match = _LABEL_LINE.match(line)
        if not match:
            continue
        label, value = match.group(1).strip(), match.group(2).strip()
        named = [name for name, example in values.items() if value == example]
        if named:
            param_lines.append(f"{label}: {{{named[0]}}}")
        elif value and _is_static(label, dynamic):
            labels.append(f"{label}:")
    anchors = [
        strategy.anchor_text
        for step in run.steps
        if step.kind == "extract" and step.locator
        for strategy in step.locator.strategies
        if isinstance(strategy, AncestorAnchorLocator)
        and strategy.anchor_text.lower() in final_text
        and _is_static(strategy.anchor_text, dynamic)
    ]
    last = run.steps[-1] if run.steps else None
    waited = [last.expect_text] if last and last.kind == "wait_for" and last.expect_text else []
    ordered = [*param_lines, *waited, *anchors, *labels]
    unique = list(dict.fromkeys(parameterize(text, values) for text in ordered))
    return Checkpoint(
        url_pattern=_path(final.url) if final else None,
        text_present=unique[:_CHECKPOINT_LIMIT],
    )


def _placeholders_used(cap_parts: object) -> set[str]:
    text = json.dumps(cap_parts, default=lambda o: o.model_dump(mode="json"))
    return set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", text))


def synthesize(
    run: RecordedRun,
    task: DiscoveryTask,
    spec: CapabilitySpec,
    *,
    now: datetime | None = None,
) -> Synthesis:
    if run.outcome != "finished":
        raise SynthesisError(
            f"only a finished run can be synthesized (this one is {run.outcome}: {run.reason})"
        )
    if run.missing_outputs:
        raise SynthesisError(
            "outputs were requested but never extracted: " + ", ".join(run.missing_outputs)
        )

    values = {name: p.value for name, p in task.params.items() if not p.sensitive}
    sensitive_values = [p.value for p in task.params.values() if p.sensitive]
    dynamic = [*values.values(), *sensitive_values, *run.outputs.values()]
    warnings: list[str] = []

    steps: list[Step] = []
    previous: ObservationDigest | None = None
    last_url = _path(run.start_url)
    for recorded in run.steps:
        if recorded.after is not None:
            last_url = _path(recorded.after.url)
        expect = _expectation(recorded, previous, dynamic, values, last_url, warnings)
        steps.append(
            Step(
                id=recorded.id,
                action=ActionKind(recorded.kind),
                description=_scrub(recorded.description, values, run.outputs) or recorded.kind,
                url_template=_path(recorded.url) if recorded.url else None,
                locator=recorded.locator,
                value=recorded.value,
                output=recorded.output,
                risk_class=recorded.risk,
                sensitive=isinstance(recorded.value, SecretRef),
                expect=expect,
                dialogs=_dialogs(recorded, values),
            )
        )
        if recorded.after is not None:
            previous = recorded.after

    # An input is a parameter the *steps* use. The checkpoint may only reuse those: a page that
    # happens to show an example value must not turn an unused parameter into a requirement.
    used = {s.value.name for s in steps if isinstance(s.value, ParamRef)}
    used |= _placeholders_used(steps)
    checkpoint = _checkpoint(run, {n: v for n, v in values.items() if n in used}, dynamic)
    for name in task.params:
        if name not in used:
            warnings.append(f"parameter '{name}' was never used, so it is not an input")

    inputs = {
        "type": "object",
        "properties": {
            name: {
                "type": "string",
                "description": p.description,
                **({"pattern": p.pattern} if p.pattern else {}),
                **({"x-sensitive": True} if p.sensitive else {}),
            }
            for name, p in task.params.items()
            if name in used
        },
        "required": [name for name in task.params if name in used],
        "additionalProperties": False,
    }
    outputs = {
        "type": "object",
        "properties": {
            name: {
                "type": "string",
                "description": o.description,
                **({"x-sensitive": True} if o.sensitive else {}),
            }
            for name, o in task.outputs.items()
        },
        "required": list(task.outputs),
        "additionalProperties": False,
    }
    try:
        capability = Capability(
            id=spec.id,
            capability_version=spec.version,
            title=spec.title,
            description=spec.description,
            target=spec.target,
            inputs=inputs,
            outputs=outputs,
            steps=steps,
            checkpoint=checkpoint,
            error_map=list(spec.error_map),
            provenance=Provenance(
                run_id=run.run_id,
                discovered_at=now or datetime.now(UTC),
                model=run.model,
                steps_observed=len(run.steps),
            ),
        )
    except ValidationError as exc:
        raise SynthesisError(f"the recording does not make a valid capability: {exc}") from exc
    return Synthesis(capability, warnings)
