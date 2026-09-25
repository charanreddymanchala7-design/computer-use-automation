"""The review view of a capability: what it does, needs, returns and risks, in plain text.

A capability is only trustworthy if a person can read it and understand it. This is that
reading: the same document a calling agent consumes, laid out for a human reviewer, with every
state-changing step impossible to miss.
"""

from __future__ import annotations

import textwrap

from cua.artifact.schema import (
    Capability,
    ErrorClass,
    ErrorRule,
    ExpectedDialog,
    LiteralValue,
    LocatorBundle,
    ParamRef,
    RiskClass,
    SecretRef,
    Step,
)

_RISK_TAG = {
    RiskClass.READ: "read",
    RiskClass.REVERSIBLE_WRITE: "WRITE",
    RiskClass.IRREVERSIBLE_WRITE: "IRREVERSIBLE WRITE",
}
_CLASS_LABEL = {
    ErrorClass.BUSINESS_OUTCOME: "business outcome",
    ErrorClass.RECOVERABLE: "recoverable",
    ErrorClass.HARD_FAILURE: "hard failure",
}


def _value(step: Step) -> str:
    value = step.value
    if isinstance(value, ParamRef):
        return f"input {value.name}"
    if isinstance(value, SecretRef):
        return f"secret {value.name}"
    if isinstance(value, LiteralValue):
        return f"constant {value.value!r}"
    return ""


def _finds(locator: LocatorBundle | None) -> str:
    if locator is None:
        return ""
    frames = " > ".join(hop.name or f"#{hop.index}" for hop in locator.frame_path) or "top"
    kinds = " > ".join(s.kind for s in locator.strategies)
    return f"finds: {locator.description} in frame {frames} by {kinds}"


def _dialog(dialog: ExpectedDialog) -> str:
    return f"{dialog.action}s {dialog.kind} {dialog.message!r}"


def _step_lines(step: Step) -> list[str]:
    head = f"  {step.id} [{_RISK_TAG[step.risk_class]}] {step.action.value}: {step.description}"
    lines = [head]
    if step.url_template:
        lines.append(f"      goes to {step.url_template}")
    if step.value:
        lines.append(f"      types {_value(step)}")
    if step.output:
        lines.append(f"      returns {step.output}")
    if step.locator:
        lines.append(f"      {_finds(step.locator)}")
    if step.expect:
        checks = [*step.expect.text_present]
        if step.expect.url_pattern:
            checks.append(f"url {step.expect.url_pattern}")
        lines.append(
            "      expects " + "; ".join(repr(c) if not c.startswith("url ") else c for c in checks)
        )
    lines.extend(f"      {_dialog(d)}" for d in step.dialogs)
    return lines


def _rule_line(rule: ErrorRule) -> str:
    detect = [*rule.detect.text_present]
    if rule.detect.url_pattern:
        detect.append(f"url {rule.detect.url_pattern}")
    if rule.detect.dialog_message:
        detect.append(f"dialog {rule.detect.dialog_message!r}")
    seen = "; ".join(repr(d) if not d.startswith(("url ", "dialog ")) else d for d in detect)
    if rule.classification is ErrorClass.BUSINESS_OUTCOME:
        result = f"returns outcome {rule.outcome_code}"
    elif rule.classification is ErrorClass.RECOVERABLE and rule.recovery:
        result = f"recovers by {rule.recovery.kind} (up to {rule.recovery.max_attempts} tries)"
    else:
        result = f"fails with {rule.code}" + (" and brings in a human" if rule.escalate else "")
    return f"  {rule.id}: {_CLASS_LABEL[rule.classification]}; when {seen or 'seen'}, {result}"


def render_review(cap: Capability) -> str:
    props = cap.inputs.get("properties", {})
    outs = cap.outputs.get("properties", {})
    secrets = cap.required_secrets()
    writes = [s for s in cap.steps if s.risk_class is not RiskClass.READ]
    out: list[str] = [
        cap.title,
        f"{cap.id} {cap.capability_version}  [{cap.status}]  {cap.content_digest()[:19]}",
        f"Target: {cap.target.vendor} {cap.target.product} {cap.target.version_range}"
        f" (tenant profile {cap.target.tenant_profile}, surface {cap.target.surface})",
        "",
        *textwrap.wrap(cap.description, 96),
        "",
        "Needs",
    ]
    for name, schema in props.items():
        extra = f", pattern {schema['pattern']}" if "pattern" in schema else ""
        flag = ", sensitive" if schema.get("x-sensitive") else ""
        kind, note = schema.get("type", "any"), schema.get("description", "")
        out.append(f"  input {name} ({kind}{extra}{flag}): {note}")
    out += [f"  secret {name} (read from the environment, never stored)" for name in secrets]
    if not props and not secrets:
        out.append("  nothing")
    out += ["", "Returns"]
    for name, schema in outs.items():
        out.append(f"  {name} ({schema.get('type', 'any')}): {schema.get('description', '')}")
    out += ["", "Steps"]
    for step in cap.steps:
        out.extend(_step_lines(step))
    check = cap.checkpoint
    expected = [repr(t) for t in check.text_present] + (
        [f"url {check.url_pattern}"] if check.url_pattern else []
    )
    out += ["", "Checkpoint (the run succeeded if all of these hold)", "  " + "; ".join(expected)]
    out += ["", "Handles"]
    out += [_rule_line(rule) for rule in cap.error_map] or ["  nothing declared"]
    out += ["", "Safety"]
    if not writes:
        out.append("  read-only: no step changes anything")
    else:
        out.append(
            "  changes state at " + ", ".join(f"{s.id} ({_RISK_TAG[s.risk_class]})" for s in writes)
        )
    if cap.provenance:
        out += [
            "",
            f"Discovered by {cap.provenance.model} in run {cap.provenance.run_id}"
            f" ({cap.provenance.steps_observed} steps)",
        ]
    return "\n".join(out)
