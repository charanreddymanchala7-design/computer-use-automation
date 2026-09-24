"""The capability artifact: a typed, versioned, reviewable description of one automated flow.

A capability has two halves that different readers care about:

* the **contract** (id, versions, target, inputs, outputs, error map, risk) is what a calling
  agent or a human reviewer reads to learn what the capability does, needs and returns;
* the **recipe** (steps, locator bundles, checkpoint) is what deterministic replay executes.

Design rules enforced here rather than by convention:

* inputs and outputs are plain JSON Schema, so they map straight onto tool definitions;
* a step value is a literal, a reference to a declared input, or a reference to a secret, and
  sensitive steps can never carry a literal, so raw sensitive data cannot reach an artifact;
* every locator strategy states why it should survive a UI change, and the least robust one
  (coordinates) can only be a last resort;
* write steps must declare what success looks like, so a click is verified, not assumed;
* runtime conditions are declared up front as business outcome, recoverable, or hard failure.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

import jsonschema
from pydantic import Field, model_validator

from cua.common import CODE_PATTERN, StrictModel

SCHEMA_VERSION: Literal["1"] = "1"

_SLUG = r"^[a-z][a-z0-9_]{2,63}$"
_SEMVER = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(-[0-9A-Za-z.-]+)?$"
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SENSITIVE_KEYWORD = "x-sensitive"
_FORBIDDEN_ON_SENSITIVE = ("default", "examples", "enum", "const")


_Model = StrictModel


class RiskClass(StrEnum):
    READ = "read"
    REVERSIBLE_WRITE = "reversible_write"
    IRREVERSIBLE_WRITE = "irreversible_write"


class ActionKind(StrEnum):
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    PRESS = "press"
    WAIT_FOR = "wait_for"
    EXTRACT = "extract"


class ErrorClass(StrEnum):
    BUSINESS_OUTCOME = "business_outcome"
    RECOVERABLE = "recoverable"
    HARD_FAILURE = "hard_failure"


# --- locators ---------------------------------------------------------------------------------


class FrameSelector(_Model):
    """One hop into a frame. Legacy framesets often have no names, so index and URL work too."""

    name: str | None = None
    url_pattern: str | None = None
    index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _identifies_a_frame(self) -> Self:
        if self.name is None and self.url_pattern is None and self.index is None:
            raise ValueError("frame selector needs name, url_pattern or index")
        return self


class _Strategy(_Model):
    rationale: str = Field(min_length=3, description="Why this strategy should survive UI change")


class RoleNameLocator(_Strategy):
    kind: Literal["role_name"] = "role_name"
    role: str
    name: str
    exact: bool = False


class LabelLocator(_Strategy):
    kind: Literal["label"] = "label"
    text: str
    exact: bool = False


class TextLocator(_Strategy):
    kind: Literal["text"] = "text"
    text: str
    exact: bool = False


class AncestorAnchorLocator(_Strategy):
    """Find the element by a nearby stable landmark, e.g. the row that says 'Savings'."""

    kind: Literal["ancestor_anchor"] = "ancestor_anchor"
    anchor_text: str
    container: Literal["row", "cell", "form", "fieldset", "frame_body", "any"] = "any"
    target_role: str | None = None
    target_tag: str | None = None
    nth: int = Field(default=0, ge=0)


class AttributeFingerprintLocator(_Strategy):
    """Tag plus stable attributes (name, type, alt, title). Never dynamic ids."""

    kind: Literal["attribute_fingerprint"] = "attribute_fingerprint"
    tag: str
    attributes: dict[str, str] = Field(min_length=1)


class CssLocator(_Strategy):
    kind: Literal["css"] = "css"
    selector: str


class XPathLocator(_Strategy):
    kind: Literal["xpath"] = "xpath"
    expression: str


class CoordinatesLocator(_Strategy):
    """Last resort for surfaces with no DOM or accessibility tree."""

    kind: Literal["coordinates"] = "coordinates"
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    viewport_width: int = Field(gt=0)
    viewport_height: int = Field(gt=0)


LocatorStrategy = Annotated[
    RoleNameLocator
    | LabelLocator
    | TextLocator
    | AncestorAnchorLocator
    | AttributeFingerprintLocator
    | CssLocator
    | XPathLocator
    | CoordinatesLocator,
    Field(discriminator="kind"),
]


class LocatorBundle(_Model):
    """A ranked list of ways to find one element. Replay tries them in order."""

    description: str = Field(min_length=1, description="Human name of the element")
    frame_path: list[FrameSelector] = Field(default_factory=list)
    strategies: list[LocatorStrategy] = Field(min_length=1)

    @model_validator(mode="after")
    def _coordinates_last(self) -> Self:
        if any(s.kind == "coordinates" for s in self.strategies[:-1]):
            raise ValueError("coordinates must be the last strategy in a bundle (least robust)")
        return self


# --- values -----------------------------------------------------------------------------------


class LiteralValue(_Model):
    source: Literal["literal"] = "literal"
    value: str | int | float | bool


class ParamRef(_Model):
    source: Literal["param"] = "param"
    name: str


class SecretRef(_Model):
    source: Literal["secret"] = "secret"
    name: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$", description="Environment variable name")


ValueRef = Annotated[LiteralValue | ParamRef | SecretRef, Field(discriminator="source")]


# --- expectations, checkpoint and error handling ----------------------------------------------


class Expectation(_Model):
    """What must be true after a step for it to count as done."""

    url_pattern: str | None = None
    text_present: list[str] = Field(default_factory=list)
    text_absent: list[str] = Field(default_factory=list)
    element: LocatorBundle | None = None
    timeout_ms: int = Field(default=5000, ge=100, le=120_000)

    @model_validator(mode="after")
    def _checks_something(self) -> Self:
        if not (self.url_pattern or self.text_present or self.text_absent or self.element):
            raise ValueError(
                "expect must check at least one of url_pattern, text_present, "
                "text_absent or element"
            )
        return self


class Checkpoint(_Model):
    """The success condition of the whole capability."""

    url_pattern: str | None = None
    text_present: list[str] = Field(default_factory=list)
    aria_contains: list[str] = Field(default_factory=list)
    timeout_ms: int = Field(default=10_000, ge=100, le=120_000)

    @model_validator(mode="after")
    def _checks_something(self) -> Self:
        if not (self.url_pattern or self.text_present or self.aria_contains):
            raise ValueError(
                "checkpoint must check at least one of url_pattern, text_present or aria_contains"
            )
        return self


class Detector(_Model):
    """How replay recognises a runtime condition. Any one signal is enough."""

    url_pattern: str | None = None
    text_present: list[str] = Field(default_factory=list)
    dialog_message: str | None = None
    element: LocatorBundle | None = None

    @model_validator(mode="after")
    def _has_a_signal(self) -> Self:
        if not (self.url_pattern or self.text_present or self.dialog_message or self.element):
            raise ValueError(
                "detect must specify at least one of url_pattern, text_present, "
                "dialog_message or element"
            )
        return self


class Recovery(_Model):
    kind: Literal["dismiss", "wait_retry"]
    locator: LocatorBundle | None = None
    max_attempts: int = Field(default=3, ge=1, le=5)
    backoff_ms: int = Field(default=500, ge=0, le=10_000)

    @model_validator(mode="after")
    def _dismiss_needs_a_target(self) -> Self:
        if self.kind == "dismiss" and self.locator is None:
            raise ValueError("dismiss recovery requires locator")
        return self


_CODE = CODE_PATTERN

# class -> (required field, fields that belong to other classes)
_ERROR_FIELDS: dict[ErrorClass, tuple[str, tuple[str, ...]]] = {
    ErrorClass.BUSINESS_OUTCOME: ("outcome_code", ("recovery", "code")),
    ErrorClass.RECOVERABLE: ("recovery", ("outcome_code", "code")),
    ErrorClass.HARD_FAILURE: ("code", ("outcome_code", "recovery")),
}


class ErrorRule(_Model):
    """One declared runtime condition and how it is classified."""

    id: str = Field(pattern=_CODE)
    detect: Detector
    classification: ErrorClass
    message: str | None = None
    outcome_code: str | None = Field(default=None, pattern=_CODE)
    recovery: Recovery | None = None
    code: str | None = Field(default=None, pattern=_CODE)

    @model_validator(mode="after")
    def _fields_match_classification(self) -> Self:
        required, forbidden = _ERROR_FIELDS[self.classification]
        cls = self.classification.value
        if getattr(self, required) is None:
            raise ValueError(f"{cls} requires {required}")
        for name in forbidden:
            if getattr(self, name) is not None:
                raise ValueError(f"{cls} must not set {name}")
        return self


# --- steps ------------------------------------------------------------------------------------

_OPERANDS = ("url_template", "locator", "value", "output")

# action -> (required operands, optional operands)
_ACTION_OPERANDS: dict[ActionKind, tuple[tuple[str, ...], tuple[str, ...]]] = {
    ActionKind.NAVIGATE: (("url_template",), ()),
    ActionKind.CLICK: (("locator",), ()),
    ActionKind.FILL: (("locator", "value"), ()),
    ActionKind.SELECT: (("locator", "value"), ()),
    ActionKind.PRESS: (("value",), ("locator",)),
    ActionKind.WAIT_FOR: ((), ()),
    ActionKind.EXTRACT: (("locator", "output"), ()),
}


class Step(_Model):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    action: ActionKind
    description: str = Field(min_length=1)
    url_template: str | None = Field(
        default=None, description="Absolute URL; {input_name} placeholders are substituted"
    )
    locator: LocatorBundle | None = None
    value: ValueRef | None = None
    output: str | None = Field(default=None, description="Output property this step extracts")
    risk_class: RiskClass
    sensitive: bool = False
    expect: Expectation | None = None

    @model_validator(mode="after")
    def _operands_match_action(self) -> Self:
        required, optional = _ACTION_OPERANDS[self.action]
        action = self.action.value
        for name in required:
            if getattr(self, name) is None:
                raise ValueError(f"{action} requires {name}")
        for name in _OPERANDS:
            if name not in required + optional and getattr(self, name) is not None:
                raise ValueError(f"{action} must not carry {name}")
        if self.action is ActionKind.WAIT_FOR and self.expect is None:
            raise ValueError("wait_for requires expect")
        return self

    @model_validator(mode="after")
    def _writes_are_verified(self) -> Self:
        if self.risk_class is not RiskClass.READ and self.expect is None:
            raise ValueError(
                f"write steps must declare expect (step {self.id} is {self.risk_class.value})"
            )
        return self

    @model_validator(mode="after")
    def _sensitive_steps_carry_no_raw_values(self) -> Self:
        if self.sensitive and isinstance(self.value, LiteralValue):
            raise ValueError("sensitive step must use a param or secret, never a literal value")
        return self


# --- the capability ---------------------------------------------------------------------------


class Target(_Model):
    """What software this capability was recorded against; the seam for multi-tenant reuse."""

    vendor: str
    product: str
    version_range: str
    tenant_profile: str = "default"
    surface: Literal["web", "desktop"] = "web"


class Provenance(_Model):
    """How it was discovered. Deliberately not the model transcript."""

    run_id: str | None = None
    discovered_at: datetime | None = None
    model: str | None = None
    steps_observed: int | None = Field(default=None, ge=1)


def _properties(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    props: dict[str, dict[str, Any]] = schema.get("properties", {})
    return props


def _check_object_schema(label: str, schema: dict[str, Any]) -> None:
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise ValueError(f"{label} is not a valid JSON Schema: {exc.message}") from exc
    if schema.get("type") != "object":
        raise ValueError(f"{label} must be an object schema (type: object)")


class Capability(_Model):
    schema_version: Literal["1"] = SCHEMA_VERSION
    id: str = Field(pattern=_SLUG, description="Tool-name style slug, stable across versions")
    capability_version: str = Field(pattern=_SEMVER)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    status: Literal["draft", "active", "deprecated"] = "draft"
    target: Target
    inputs: dict[str, Any] = Field(
        description="JSON Schema (object) of the parameters callers supply"
    )
    outputs: dict[str, Any] = Field(description="JSON Schema (object) of the data returned")
    steps: list[Step] = Field(min_length=1)
    checkpoint: Checkpoint
    error_map: list[ErrorRule] = Field(default_factory=list)
    provenance: Provenance | None = None

    def sensitive_params(self) -> set[str]:
        return {n for n, p in _properties(self.inputs).items() if p.get(_SENSITIVE_KEYWORD) is True}

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def content_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @model_validator(mode="after")
    def _cross_references(self) -> Self:
        _check_object_schema("inputs", self.inputs)
        _check_object_schema("outputs", self.outputs)
        inputs, outputs = _properties(self.inputs), _properties(self.outputs)
        sensitive = self.sensitive_params()
        for name in sensitive:
            for keyword in _FORBIDDEN_ON_SENSITIVE:
                if keyword in inputs[name]:
                    raise ValueError(
                        f"sensitive input '{name}' must not declare {keyword} "
                        "(raw values would be stored in the artifact)"
                    )

        seen_steps: set[str] = set()
        produced: set[str] = set()
        for step in self.steps:
            if step.id in seen_steps:
                raise ValueError(f"duplicate step id '{step.id}'")
            seen_steps.add(step.id)
            self._check_step_references(step, inputs, outputs, sensitive, produced)

        missing = [r for r in self.outputs.get("required", []) if r not in produced]
        if missing:
            raise ValueError(f"required output '{missing[0]}' is never extracted")

        seen_rules: set[str] = set()
        for rule in self.error_map:
            if rule.id in seen_rules:
                raise ValueError(f"duplicate error rule id '{rule.id}'")
            seen_rules.add(rule.id)
        return self

    @staticmethod
    def _check_step_references(
        step: Step,
        inputs: dict[str, dict[str, Any]],
        outputs: dict[str, dict[str, Any]],
        sensitive: set[str],
        produced: set[str],
    ) -> None:
        if isinstance(step.value, ParamRef):
            if step.value.name not in inputs:
                raise ValueError(f"step {step.id} references undeclared input '{step.value.name}'")
            if step.value.name in sensitive:
                step.sensitive = True
        if step.url_template is not None:
            for name in _PLACEHOLDER.findall(step.url_template):
                if name not in inputs:
                    raise ValueError(f"step {step.id} url_template uses undeclared input '{name}'")
                if name in sensitive:
                    raise ValueError(
                        f"sensitive input '{name}' must not appear in a url_template "
                        "(URLs are logged and stored)"
                    )
        if step.action is ActionKind.EXTRACT and step.output is not None:
            if step.output not in outputs:
                raise ValueError(f"step {step.id} extracts undeclared output '{step.output}'")
            if step.output in produced:
                raise ValueError(f"output '{step.output}' is extracted twice")
            produced.add(step.output)


def capability_json_schema() -> dict[str, Any]:
    """The JSON Schema of a capability file, for reviewers, editors and calling agents."""
    schema = Capability.model_json_schema()
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
