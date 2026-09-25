"""The artifact is the product's contract: a calling agent reads it, replay executes it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from cua.artifact import Capability, capability_json_schema, fill_placeholders, placeholders_in

ROOT = Path(__file__).resolve().parent.parent


def load(data: dict[str, Any]) -> Capability:
    return Capability.model_validate(data)


def step(data: dict[str, Any], step_id: str) -> dict[str, Any]:
    found: dict[str, Any] = next(s for s in data["steps"] if s["id"] == step_id)
    return found


# --- happy path and serialization -------------------------------------------------------------


def test_valid_capability_round_trips_through_json_losslessly(
    capability_dict: dict[str, Any],
) -> None:
    cap = load(capability_dict)
    again = Capability.model_validate_json(cap.model_dump_json())
    assert again == cap
    assert json.loads(again.model_dump_json()) == json.loads(cap.model_dump_json())


def test_status_defaults_to_draft(capability_dict: dict[str, Any]) -> None:
    del capability_dict["status"]
    assert load(capability_dict).status == "draft"


def test_unknown_fields_are_rejected_so_reviewers_see_everything(
    capability_dict: dict[str, Any],
) -> None:
    capability_dict["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        load(capability_dict)


def test_content_digest_ignores_key_order_but_tracks_content(
    capability_dict: dict[str, Any],
) -> None:
    a = load(capability_dict)
    b = load(dict(reversed(list(capability_dict.items()))))
    assert a.content_digest() == b.content_digest()
    capability_dict["title"] = "A different title"
    assert load(capability_dict).content_digest() != a.content_digest()


# --- identity and versioning ------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["Member Lookup", "1member", "a", "x" * 65])
def test_id_must_be_a_tool_name_style_slug(capability_dict: dict[str, Any], bad: str) -> None:
    capability_dict["id"] = bad
    with pytest.raises(ValidationError, match="id"):
        load(capability_dict)


@pytest.mark.parametrize("bad", ["1", "1.0", "v1.0.0", "1.0.0.0"])
def test_capability_version_must_be_semver(capability_dict: dict[str, Any], bad: str) -> None:
    capability_dict["capability_version"] = bad
    with pytest.raises(ValidationError, match="capability_version"):
        load(capability_dict)


def test_schema_version_is_pinned(capability_dict: dict[str, Any]) -> None:
    capability_dict["schema_version"] = "2"
    with pytest.raises(ValidationError, match="schema_version"):
        load(capability_dict)


# --- locator bundles --------------------------------------------------------------------------


def test_coordinates_are_only_allowed_as_the_last_strategy(
    capability_dict: dict[str, Any],
) -> None:
    coords = {
        "kind": "coordinates",
        "x": 10,
        "y": 20,
        "viewport_width": 1280,
        "viewport_height": 800,
        "rationale": "No DOM or accessibility tree here",
    }
    bundle = step(capability_dict, "s3")["locator"]
    bundle["strategies"].append(coords)
    load(capability_dict)  # last is fine
    bundle["strategies"].insert(0, bundle["strategies"].pop())  # now first
    with pytest.raises(ValidationError, match="coordinates must be the last"):
        load(capability_dict)


def test_every_strategy_needs_a_robustness_rationale(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s3")["locator"]["strategies"][0]["rationale"] = ""
    with pytest.raises(ValidationError, match="rationale"):
        load(capability_dict)


def test_a_bundle_needs_at_least_one_strategy(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s3")["locator"]["strategies"] = []
    with pytest.raises(ValidationError, match="at least 1"):
        load(capability_dict)


def test_unknown_strategy_kind_is_rejected(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s3")["locator"]["strategies"][0]["kind"] = "telepathy"
    with pytest.raises(ValidationError, match="telepathy"):
        load(capability_dict)


def test_a_frame_selector_must_identify_a_frame(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s3")["locator"]["frame_path"] = [{}]
    with pytest.raises(ValidationError, match="name, url_pattern or index"):
        load(capability_dict)


def test_frames_can_be_identified_by_index_or_url_pattern(
    capability_dict: dict[str, Any],
) -> None:
    step(capability_dict, "s3")["locator"]["frame_path"] = [
        {"index": 1},
        {"url_pattern": "/members/*/accounts"},
    ]
    assert len(load(capability_dict).steps[2].locator.frame_path) == 2  # type: ignore[union-attr]


# --- steps ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("step_id", "missing", "message"),
    [
        ("s1", "url_template", "navigate requires url_template"),
        ("s2", "locator", "fill requires locator"),
        ("s2", "value", "fill requires value"),
        ("s3", "locator", "click requires locator"),
        ("s4", "output", "extract requires output"),
    ],
)
def test_each_action_kind_requires_its_operands(
    capability_dict: dict[str, Any], step_id: str, missing: str, message: str
) -> None:
    del step(capability_dict, step_id)[missing]
    with pytest.raises(ValidationError, match=message):
        load(capability_dict)


def test_navigate_must_not_carry_a_locator(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s1")["locator"] = step(capability_dict, "s3")["locator"]
    with pytest.raises(ValidationError, match="navigate must not"):
        load(capability_dict)


def test_wait_for_requires_an_expectation(capability_dict: dict[str, Any]) -> None:
    capability_dict["steps"].append(
        {"id": "s5", "action": "wait_for", "description": "Wait", "risk_class": "read"}
    )
    with pytest.raises(ValidationError, match="wait_for requires expect"):
        load(capability_dict)


def test_writes_must_declare_what_success_looks_like(capability_dict: dict[str, Any]) -> None:
    s3 = step(capability_dict, "s3")
    s3["risk_class"] = "reversible_write"
    del s3["expect"]
    with pytest.raises(ValidationError, match="write steps must declare expect"):
        load(capability_dict)


def test_an_expectation_must_check_something(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s3")["expect"] = {"timeout_ms": 1000}
    with pytest.raises(ValidationError, match="expect must check at least one"):
        load(capability_dict)


def test_step_ids_are_unique(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s4")["id"] = "s3"
    with pytest.raises(ValidationError, match="duplicate step id"):
        load(capability_dict)


def test_navigate_placeholders_must_be_declared_inputs(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s1")["url_template"] = "http://127.0.0.1:4310/members/{nobody}"
    with pytest.raises(ValidationError, match="nobody"):
        load(capability_dict)
    step(capability_dict, "s1")["url_template"] = "http://127.0.0.1:4310/members/{member_id}"
    load(capability_dict)


# --- sensitive data ---------------------------------------------------------------------------


def test_sensitive_steps_cannot_carry_raw_values(capability_dict: dict[str, Any]) -> None:
    s2 = step(capability_dict, "s2")
    s2["sensitive"] = True
    s2["value"] = {"source": "literal", "value": "hunter2"}
    with pytest.raises(ValidationError, match="sensitive step must use a param or secret"):
        load(capability_dict)


def test_sensitive_steps_may_use_params_or_secrets(capability_dict: dict[str, Any]) -> None:
    s2 = step(capability_dict, "s2")
    s2["sensitive"] = True
    load(capability_dict)  # param ref is fine
    s2["value"] = {"source": "secret", "name": "BANK_PASSWORD"}
    load(capability_dict)


def test_a_step_using_a_sensitive_param_is_treated_as_sensitive(
    capability_dict: dict[str, Any],
) -> None:
    capability_dict["inputs"]["properties"]["member_id"]["x-sensitive"] = True
    cap = load(capability_dict)
    assert cap.sensitive_params() == {"member_id"}
    assert cap.steps[1].sensitive is True


@pytest.mark.parametrize("keyword", ["default", "examples", "enum", "const"])
def test_sensitive_inputs_cannot_embed_example_values(
    capability_dict: dict[str, Any], keyword: str
) -> None:
    prop = capability_dict["inputs"]["properties"]["member_id"]
    prop["x-sensitive"] = True
    prop[keyword] = ["12345"] if keyword in ("examples", "enum") else "12345"
    with pytest.raises(ValidationError, match=r"sensitive input .* must not declare"):
        load(capability_dict)


def test_sensitive_inputs_cannot_appear_in_urls(capability_dict: dict[str, Any]) -> None:
    capability_dict["inputs"]["properties"]["member_id"]["x-sensitive"] = True
    step(capability_dict, "s1")["url_template"] = "http://127.0.0.1:4310/members/{member_id}"
    with pytest.raises(ValidationError, match="must not appear in a url_template"):
        load(capability_dict)


# --- parameterized locators and expectations ---------------------------------------------------


def test_locator_text_may_reference_declared_inputs(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s3")["locator"]["strategies"][1]["text"] = "Row for {member_id}"
    step(capability_dict, "s3")["expect"] = {"text_present": ["Search results", "{member_id}"]}
    load(capability_dict)


@pytest.mark.parametrize(
    "where",
    ["locator", "expect", "checkpoint", "detector", "recovery"],
)
def test_placeholders_must_be_declared_inputs_wherever_they_appear(
    capability_dict: dict[str, Any], where: str
) -> None:
    if where == "locator":
        step(capability_dict, "s3")["locator"]["strategies"][1]["text"] = "{ghost}"
    elif where == "expect":
        step(capability_dict, "s3")["expect"] = {"text_present": ["{ghost}"]}
    elif where == "checkpoint":
        capability_dict["checkpoint"]["text_present"] = ["{ghost}"]
    elif where == "detector":
        capability_dict["error_map"][0]["detect"]["text_present"] = ["{ghost}"]
    else:
        strategy = capability_dict["error_map"][1]["recovery"]["locator"]["strategies"][0]
        strategy["name"] = "{ghost}"
    with pytest.raises(ValidationError, match="undeclared input 'ghost'"):
        load(capability_dict)


def test_a_sensitive_input_cannot_be_used_in_a_locator_or_expectation(
    capability_dict: dict[str, Any],
) -> None:
    capability_dict["inputs"]["properties"]["member_id"]["x-sensitive"] = True
    step(capability_dict, "s3")["expect"] = {"text_present": ["{member_id}"]}
    with pytest.raises(ValidationError, match="must not appear in a locator or expectation"):
        load(capability_dict)


def test_fingerprint_attribute_values_can_be_parameterized(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s2")["locator"]["strategies"][1]["attributes"] = {"name": "{member_id}"}
    load(capability_dict)


def test_regex_quantifiers_are_not_mistaken_for_placeholders(
    capability_dict: dict[str, Any],
) -> None:
    capability_dict["checkpoint"]["url_pattern"] = "^/members/[0-9]{5}$"
    load(capability_dict)


def test_placeholders_are_listed_and_filled_with_the_callers_values() -> None:
    assert placeholders_in("row {member_id} of {branch}") == ["member_id", "branch"]
    assert fill_placeholders("row {member_id}", {"member_id": "12345"}) == "row 12345"
    assert fill_placeholders("no placeholders", {}) == "no placeholders"
    assert fill_placeholders("^[0-9]{5}$", {}) == "^[0-9]{5}$"
    with pytest.raises(KeyError, match="branch"):
        fill_placeholders("row {branch}", {"member_id": "1"})


# --- expected dialogs and required secrets ------------------------------------------------------


def test_a_step_can_declare_the_native_dialog_it_provokes(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s3")["dialogs"] = [
        {"kind": "confirm", "message": "Open new sub-account for member {member_id}?"}
    ]
    cap = load(capability_dict)
    (dialog,) = cap.steps[2].dialogs
    assert (dialog.kind, dialog.action) == ("confirm", "accept")  # accepting is the default


def test_a_dialog_can_be_declared_as_dismissed(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s3")["dialogs"] = [
        {"kind": "confirm", "message": "Discard changes?", "action": "dismiss"}
    ]
    assert load(capability_dict).steps[2].dialogs[0].action == "dismiss"


def test_dialog_messages_follow_the_same_placeholder_rules(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s3")["dialogs"] = [{"kind": "alert", "message": "Hello {ghost}"}]
    with pytest.raises(ValidationError, match="undeclared input 'ghost'"):
        load(capability_dict)


def test_an_unknown_dialog_kind_is_rejected(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s3")["dialogs"] = [{"kind": "popup", "message": "x"}]
    with pytest.raises(ValidationError, match="popup"):
        load(capability_dict)


def test_the_secrets_a_capability_needs_are_listed_by_name(capability_dict: dict[str, Any]) -> None:
    assert load(capability_dict).required_secrets() == []
    step(capability_dict, "s2")["value"] = {"source": "secret", "name": "BANK_PASSWORD"}
    step(capability_dict, "s2")["sensitive"] = True
    capability_dict["steps"].insert(
        2,
        {
            **step(capability_dict, "s2"),
            "id": "s2b",
            "value": {"source": "secret", "name": "BANK_USER"},
        },
    )
    assert load(capability_dict).required_secrets() == ["BANK_PASSWORD", "BANK_USER"]


# --- inputs, outputs and extraction -----------------------------------------------------------


def test_params_referenced_by_steps_must_be_declared_inputs(
    capability_dict: dict[str, Any],
) -> None:
    step(capability_dict, "s2")["value"] = {"source": "param", "name": "ghost"}
    with pytest.raises(ValidationError, match="undeclared input 'ghost'"):
        load(capability_dict)


def test_inputs_must_be_an_object_schema(capability_dict: dict[str, Any]) -> None:
    capability_dict["inputs"] = {"type": "string"}
    with pytest.raises(ValidationError, match="inputs must be an object schema"):
        load(capability_dict)


def test_inputs_must_be_a_valid_json_schema(capability_dict: dict[str, Any]) -> None:
    capability_dict["inputs"]["properties"]["member_id"]["type"] = "not-a-type"
    with pytest.raises(ValidationError, match="not a valid JSON Schema"):
        load(capability_dict)


def test_extract_output_must_be_a_declared_output(capability_dict: dict[str, Any]) -> None:
    step(capability_dict, "s4")["output"] = "nothing"
    with pytest.raises(ValidationError, match="undeclared output 'nothing'"):
        load(capability_dict)


def test_every_required_output_must_be_produced(capability_dict: dict[str, Any]) -> None:
    capability_dict["outputs"]["properties"]["holder_name"] = {"type": "string"}
    capability_dict["outputs"]["required"].append("holder_name")
    with pytest.raises(ValidationError, match="required output 'holder_name' is never extracted"):
        load(capability_dict)


def test_an_output_cannot_be_extracted_twice(capability_dict: dict[str, Any]) -> None:
    dup = json.loads(json.dumps(step(capability_dict, "s4")))
    dup["id"] = "s5"
    capability_dict["steps"].append(dup)
    with pytest.raises(ValidationError, match="output 'savings_balance' is extracted twice"):
        load(capability_dict)


# --- checkpoint -------------------------------------------------------------------------------


def test_checkpoint_needs_at_least_one_condition(capability_dict: dict[str, Any]) -> None:
    capability_dict["checkpoint"] = {}
    with pytest.raises(ValidationError, match="checkpoint must check at least one"):
        load(capability_dict)


# --- error map --------------------------------------------------------------------------------


def test_business_outcomes_need_an_outcome_code(capability_dict: dict[str, Any]) -> None:
    del capability_dict["error_map"][0]["outcome_code"]
    with pytest.raises(ValidationError, match="business_outcome requires outcome_code"):
        load(capability_dict)


def test_recoverable_conditions_need_a_recovery(capability_dict: dict[str, Any]) -> None:
    del capability_dict["error_map"][1]["recovery"]
    with pytest.raises(ValidationError, match="recoverable requires recovery"):
        load(capability_dict)


def test_hard_failures_need_a_code(capability_dict: dict[str, Any]) -> None:
    del capability_dict["error_map"][2]["code"]
    with pytest.raises(ValidationError, match="hard_failure requires code"):
        load(capability_dict)


def test_error_rules_reject_fields_from_other_classes(capability_dict: dict[str, Any]) -> None:
    capability_dict["error_map"][2]["outcome_code"] = "oops"
    with pytest.raises(ValidationError, match="hard_failure must not set outcome_code"):
        load(capability_dict)


def test_a_hard_failure_can_ask_for_a_human_instead_of_stopping(
    capability_dict: dict[str, Any],
) -> None:
    capability_dict["error_map"][2]["escalate"] = True
    assert load(capability_dict).error_map[2].escalate is True
    assert load({**capability_dict, "error_map": []}).error_map == []
    del capability_dict["error_map"][2]["escalate"]
    assert load(capability_dict).error_map[2].escalate is False  # stopping is the default


@pytest.mark.parametrize("index", [0, 1])
def test_only_a_hard_failure_can_escalate(capability_dict: dict[str, Any], index: int) -> None:
    capability_dict["error_map"][index]["escalate"] = True
    with pytest.raises(ValidationError, match="only a hard_failure can escalate"):
        load(capability_dict)


def test_error_rule_ids_are_unique(capability_dict: dict[str, Any]) -> None:
    capability_dict["error_map"][1]["id"] = "no_such_member"
    with pytest.raises(ValidationError, match="duplicate error rule id"):
        load(capability_dict)


def test_a_detector_needs_at_least_one_signal(capability_dict: dict[str, Any]) -> None:
    capability_dict["error_map"][0]["detect"] = {}
    with pytest.raises(ValidationError, match="detect must specify at least one"):
        load(capability_dict)


def test_dismiss_recovery_needs_something_to_click(capability_dict: dict[str, Any]) -> None:
    del capability_dict["error_map"][1]["recovery"]["locator"]
    with pytest.raises(ValidationError, match="dismiss recovery requires locator"):
        load(capability_dict)


def test_wait_retry_recovery_needs_no_locator(capability_dict: dict[str, Any]) -> None:
    recovery = capability_dict["error_map"][1]["recovery"]
    recovery["kind"] = "wait_retry"
    del recovery["locator"]
    assert load(capability_dict).error_map[1].recovery is not None


def test_recovery_retries_are_bounded(capability_dict: dict[str, Any]) -> None:
    capability_dict["error_map"][1]["recovery"]["max_attempts"] = 50
    with pytest.raises(ValidationError, match="max_attempts"):
        load(capability_dict)


# --- exported JSON Schema ---------------------------------------------------------------------


def test_exported_json_schema_is_valid_and_accepts_the_fixture(
    capability_dict: dict[str, Any],
) -> None:
    schema = capability_json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(json.loads(load(capability_dict).model_dump_json()), schema)


def test_committed_schema_file_is_current() -> None:
    committed = json.loads((ROOT / "schemas" / "capability.schema.json").read_text())
    assert committed == capability_json_schema(), "run: uv run python scripts/export_schemas.py"
