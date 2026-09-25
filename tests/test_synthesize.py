"""From a recorded run to a reusable capability: parameters, expectations and a checkpoint are
derived, and nothing that belonged to the one discovery run is allowed to leak into the artifact."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from cua.agent import (
    DiscoveryTask,
    ObservationDigest,
    OutputSpec,
    ParamSpec,
    RecordedRun,
    RecordedStep,
)
from cua.agent.recording import FrameDigest, RecordedDialog
from cua.artifact import (
    AncestorAnchorLocator,
    Capability,
    ErrorRule,
    FrameSelector,
    LiteralValue,
    LocatorBundle,
    ParamRef,
    RiskClass,
    RoleNameLocator,
    SecretRef,
    Target,
    TextLocator,
)
from cua.artifact.synthesize import CapabilitySpec, SynthesisError, synthesize
from cua.evlog import ArtifactLeak
from cua.redact import Redactor

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
BASE = "http://127.0.0.1:4310"


def digest(url: str, *frames: tuple[str | None, str]) -> ObservationDigest:
    return ObservationDigest(
        url=url,
        frames=[
            FrameDigest(
                name=name, path=[name] if name else [], url="/msv/x.cgi", text=text, elements=3
            )
            for name, text in frames
        ],
    )


def anchor(label: str, tag: str = "input", frame: str = "main") -> LocatorBundle:
    return LocatorBundle(
        description=f"{tag} near {label!r}",
        frame_path=[FrameSelector(name=frame)],
        strategies=[
            AncestorAnchorLocator(
                anchor_text=label,
                container="row",
                target_tag=tag,
                nth=0,
                rationale="Anchored on the neighbouring label in the same row",
            )
        ],
    )


LOGIN = digest(
    "/msv/login.cgi", (None, "MEMBERSERV 3.1 SIGN-ON\nEXAMPLE FCU (DEMO)\nUser ID\nPassword")
)
SHELL = digest(
    "/msv/frameset.cgi",
    (None, ""),
    ("main", "Search by:\nMember No:\nName:"),
)
RESULTS = digest(
    "/msv/frameset.cgi",
    ("main", "Search results\nMember Name Status\n12345 TESTERSON, ADA ACTIVE"),
)
MEMBER = digest(
    "/msv/frameset.cgi",
    (
        "main",
        "TESTERSON, ADA\nMember: 12345\nStatus: ACTIVE\nBranch: MAIN ST\n"
        "Member since: 01/01/2001\nTIN: 000-00-0001\nOpen Sub-Account...",
    ),
    ("acct", "Type Account Balance\nSHARE SAVINGS SYN-12345-S01 $2,480.15 Close\nCHECKING $312.40"),
)


def recorded(**overrides: Any) -> RecordedRun:
    steps = [
        RecordedStep(
            id="s1",
            kind="navigate",
            description="open the start page",
            risk=RiskClass.READ,
            url=f"{BASE}/msv/login.cgi",
            after=LOGIN,
        ),
        RecordedStep(
            id="s2",
            kind="fill",
            description="user",
            risk=RiskClass.READ,
            locator=anchor("User ID"),
            value=SecretRef(name="MOCK_USER"),
            after=LOGIN,
        ),
        RecordedStep(
            id="s3",
            kind="fill",
            description="password",
            risk=RiskClass.READ,
            locator=anchor("Password"),
            value=SecretRef(name="MOCK_PASS"),
            after=LOGIN,
        ),
        RecordedStep(
            id="s4",
            kind="click",
            description="sign in",
            risk=RiskClass.READ,
            locator=anchor("", "input"),
            after=SHELL,
        ),
        RecordedStep(
            id="s5",
            kind="fill",
            description="member number",
            risk=RiskClass.READ,
            locator=anchor("Member No:"),
            value=ParamRef(name="member_id"),
            after=SHELL,
        ),
        RecordedStep(
            id="s6",
            kind="click",
            description="search",
            risk=RiskClass.READ,
            locator=anchor("Go", "img"),
            after=RESULTS,
        ),
        RecordedStep(
            id="s7",
            kind="click",
            description="open the member",
            risk=RiskClass.READ,
            locator=LocatorBundle(
                description="result row",
                frame_path=[FrameSelector(name="main")],
                strategies=[
                    TextLocator(text="{member_id}", exact=False, rationale="the searched number")
                ],
            ),
            after=MEMBER,
        ),
        RecordedStep(
            id="s8",
            kind="extract",
            description="the balance",
            risk=RiskClass.READ,
            output="savings_balance",
            locator=LocatorBundle(
                description="value near 'SHARE SAVINGS'",
                frame_path=[FrameSelector(name="main"), FrameSelector(name="acct")],
                strategies=[
                    AncestorAnchorLocator(
                        anchor_text="SHARE SAVINGS",
                        container="row",
                        target_tag="td",
                        nth=2,
                        rationale="The row labelled by this text holds the value",
                    )
                ],
            ),
        ),
    ]
    data: dict[str, Any] = {
        "run_id": "run_1",
        "goal": "Look up a member's savings balance",
        "start_url": f"{BASE}/msv/login.cgi",
        "model": "claude-sonnet-5",
        "params": {"member_id": "12345"},
        "secrets_used": ["MOCK_USER", "MOCK_PASS"],
        "steps": steps,
        "outputs": {"savings_balance": "$2,480.15"},
        "missing_outputs": [],
        "outcome": "finished",
        "reason": "the model reported that the goal was achieved",
        "summary": "read it",
        "llm_steps": 9,
        "tokens": 12000,
        "cost_usd": 0.04,
        "duration_s": 31.0,
        "final": MEMBER,
    }
    data.update(overrides)
    return RecordedRun.model_validate(data)


TASK = DiscoveryTask(
    goal="Look up a member's savings balance",
    start_url=f"{BASE}/msv/login.cgi",
    params={"member_id": ParamSpec("12345", "Member number", pattern="^[0-9]{5}$")},
    outputs={"savings_balance": OutputSpec("Current share savings balance")},
    secrets=("MOCK_USER", "MOCK_PASS"),
)

RULES = [
    ErrorRule.model_validate(
        {
            "id": "no_such_member",
            "detect": {"text_present": ["NO RECORDS FOUND FOR CRITERIA"]},
            "classification": "business_outcome",
            "outcome_code": "member_not_found",
            "message": "No member matches the number",
        }
    )
]

SPEC = CapabilitySpec(
    id="member_lookup",
    title="Look up a member and read their savings balance",
    description="Sign in, find a member by number and read the current share savings balance.",
    target=Target(vendor="Fictional Systems", product="MemberServ", version_range=">=3.1,<4"),
    error_map=RULES,
)


def build(run: RecordedRun | None = None, task: DiscoveryTask = TASK) -> Capability:
    return synthesize(run or recorded(), task, SPEC, now=NOW).capability


# --- the shape of the result -------------------------------------------------------------------


def test_a_finished_run_becomes_a_valid_versioned_draft_capability() -> None:
    cap = build()
    assert (cap.id, cap.capability_version, cap.status) == ("member_lookup", "1.0.0", "draft")
    assert cap.target.product == "MemberServ"
    assert [s.id for s in cap.steps] == [f"s{i}" for i in range(1, 9)]
    assert [s.action.value for s in cap.steps] == [
        "navigate",
        "fill",
        "fill",
        "click",
        "fill",
        "click",
        "click",
        "extract",
    ]
    assert Capability.model_validate_json(cap.model_dump_json()) == cap


def test_values_locators_and_risk_carry_over_from_the_recording() -> None:
    cap = build()
    assert cap.steps[1].value == SecretRef(name="MOCK_USER")
    assert cap.steps[4].value == ParamRef(name="member_id")
    assert cap.steps[4].locator is not None
    assert isinstance(cap.steps[4].locator.strategies[0], AncestorAnchorLocator)
    assert cap.steps[7].output == "savings_balance"
    assert all(s.risk_class is RiskClass.READ for s in cap.steps)
    assert cap.steps[1].sensitive is True  # a secret is never stored, and the step says so


def test_navigation_is_a_portable_path_not_a_host() -> None:
    cap = build()
    assert cap.steps[0].url_template == "/msv/login.cgi"


def test_the_error_map_is_the_one_the_spec_declares() -> None:
    assert [r.id for r in build().error_map] == ["no_such_member"]


def test_provenance_records_where_it_came_from_without_the_transcript() -> None:
    cap = build()
    assert cap.provenance is not None
    assert (cap.provenance.run_id, cap.provenance.model, cap.provenance.steps_observed) == (
        "run_1",
        "claude-sonnet-5",
        8,
    )
    assert cap.provenance.discovered_at == NOW


def test_synthesis_is_deterministic() -> None:
    assert build().content_digest() == build().content_digest()


# --- inputs and outputs ------------------------------------------------------------------------


def test_the_input_schema_comes_from_the_parameters_that_were_actually_used() -> None:
    cap = build()
    assert cap.inputs == {
        "type": "object",
        "properties": {
            "member_id": {"type": "string", "description": "Member number", "pattern": "^[0-9]{5}$"}
        },
        "required": ["member_id"],
        "additionalProperties": False,
    }


def test_an_unused_parameter_is_dropped_with_a_warning() -> None:
    task = DiscoveryTask(
        **{**TASK.__dict__, "params": {**TASK.params, "branch": ParamSpec("MAIN ST", "Branch")}}
    )
    result = synthesize(recorded(), task, SPEC, now=NOW)
    assert "branch" not in result.capability.inputs["properties"]
    assert any("branch" in w for w in result.warnings)


def test_a_sensitive_parameter_is_flagged_in_the_schema() -> None:
    task = DiscoveryTask(
        **{
            **TASK.__dict__,
            "params": {
                "member_id": ParamSpec("12345", "Member number"),
                "tin": ParamSpec("000-00-0001", "Tax id", sensitive=True),
            },
        }
    )
    run = recorded()
    run.steps[4] = run.steps[4].model_copy(update={"value": ParamRef(name="tin")})
    cap = build(run, task)
    assert cap.inputs["properties"]["tin"]["x-sensitive"] is True
    assert cap.sensitive_params() == {"tin"}
    assert cap.steps[4].sensitive is True  # typed from the caller's value, never stored


def test_a_sensitive_parameter_that_was_used_to_find_something_is_refused() -> None:
    task = DiscoveryTask(
        **{
            **TASK.__dict__,
            "params": {"member_id": ParamSpec("12345", "Member number", sensitive=True)},
        }
    )
    with pytest.raises(SynthesisError, match="sensitive"):
        build(task=task)  # the row locator {member_id} would be logged and stored


def test_the_output_schema_comes_from_the_requested_outputs() -> None:
    assert build().outputs == {
        "type": "object",
        "properties": {
            "savings_balance": {"type": "string", "description": "Current share savings balance"}
        },
        "required": ["savings_balance"],
        "additionalProperties": False,
    }


def test_the_secrets_it_needs_are_listed() -> None:
    assert build().required_secrets() == ["MOCK_PASS", "MOCK_USER"]


# --- what will not be synthesized --------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["failed", "dead_end", "max_steps", "timeout"])
def test_only_a_finished_run_can_become_a_capability(outcome: str) -> None:
    with pytest.raises(SynthesisError, match=outcome):
        build(recorded(outcome=outcome))


def test_a_run_that_missed_a_requested_output_is_refused() -> None:
    with pytest.raises(SynthesisError, match="savings_balance"):
        build(recorded(missing_outputs=["savings_balance"], outputs={}))


def test_an_invalid_result_is_reported_as_a_synthesis_error() -> None:
    bad = SPEC.__class__(**{**SPEC.__dict__, "id": "Not A Slug"})
    with pytest.raises(SynthesisError, match="id"):
        synthesize(recorded(), TASK, bad, now=NOW)


# --- nothing from the discovery run leaks into the artifact ------------------------------------


def test_no_example_data_from_the_discovery_run_is_in_the_artifact() -> None:
    text = build().model_dump_json()
    assert "12345" not in text  # the member number used while discovering
    assert "$2,480.15" not in text  # the value that was read
    assert "TESTERSON" not in text  # a name from one member's page
    assert "000-00-0001" not in text  # SSN-shaped data on that page
    assert "MOCK_USER" in text  # secret *names* are fine; values are never known here
    assert "demo-only" not in text
    assert "127.0.0.1" not in text  # nor the host it ran against


def test_a_constant_typed_during_discovery_stays_a_literal() -> None:
    run = recorded()
    run.steps[4] = run.steps[4].model_copy(update={"value": LiteralValue(value="college fund")})
    task = DiscoveryTask(**{**TASK.__dict__, "params": {}})
    with pytest.raises(SynthesisError):  # member_id placeholder in step 7 is now undeclared
        build(run, task)


# --- expectations ------------------------------------------------------------------------------


def opening_run() -> RecordedRun:
    base = recorded()
    submit_after = digest(
        "/msv/frameset.cgi", ("main", "PROCESSING - PLEASE WAIT\nDO NOT CLICK BACK")
    )
    done = digest(
        "/msv/frameset.cgi",
        ("main", "SUB-ACCOUNT OPENED\nReference No: CNF-000001\nNew Account: SYN-12345-S02"),
    )
    extra = [
        RecordedStep(
            id="s9",
            kind="click",
            description="Submit",
            risk=RiskClass.REVERSIBLE_WRITE,
            locator=anchor("", "input"),
            after=submit_after,
            dialogs=[
                RecordedDialog(
                    kind="confirm", message="Open new sub-account for member 12345?", accepted=True
                )
            ],
        ),
        RecordedStep(
            id="s10",
            kind="wait_for",
            description="the confirmation",
            risk=RiskClass.READ,
            expect_text="SUB-ACCOUNT OPENED",
            after=done,
        ),
    ]
    return recorded(steps=[*base.steps, *extra], final=done)


def test_a_write_step_expects_what_appeared_and_declares_the_dialog_it_provokes() -> None:
    cap = build(opening_run())
    submit = cap.steps[8]
    assert submit.risk_class is RiskClass.REVERSIBLE_WRITE
    assert submit.expect is not None
    assert submit.expect.text_present == ["PROCESSING - PLEASE WAIT", "DO NOT CLICK BACK"]
    (dialog,) = submit.dialogs
    assert (dialog.kind, dialog.action) == ("confirm", "accept")
    assert dialog.message == "Open new sub-account for member {member_id}?"  # not member 12345


def test_waiting_for_text_becomes_an_expectation_with_room_to_arrive() -> None:
    wait = build(opening_run()).steps[9]
    assert wait.action.value == "wait_for"
    assert wait.expect is not None
    assert wait.expect.text_present == ["SUB-ACCOUNT OPENED"]
    assert wait.expect.timeout_ms >= 10_000


def test_read_steps_carry_no_text_expectation_the_next_locator_is_the_check() -> None:
    cap = build()
    assert all(s.expect is None for s in cap.steps)


def test_a_write_with_nothing_new_to_expect_gets_a_weak_expectation_and_a_warning() -> None:
    run = opening_run()
    same = run.steps[7].after
    run.steps[8] = run.steps[8].model_copy(update={"after": same})
    result = synthesize(run, TASK, SPEC, now=NOW)
    submit = result.capability.steps[8]
    assert submit.expect is not None
    assert submit.expect.url_pattern == "/msv/frameset.cgi"
    assert any("s9" in w and "weak" in w for w in result.warnings)


# --- the checkpoint ----------------------------------------------------------------------------


def test_the_checkpoint_pins_the_member_the_page_labels_and_the_row_that_was_read() -> None:
    check = build().checkpoint
    assert check.url_pattern == "/msv/frameset.cgi"
    assert check.text_present == ["Member: {member_id}", "SHARE SAVINGS", "Status:"]


def test_the_checkpoint_never_contains_values_only_labels() -> None:
    joined = " ".join(build().checkpoint.text_present)
    for value in ("ACTIVE", "TESTERSON", "MAIN ST", "000-00-0001", "$"):
        assert value not in joined


def test_a_wait_for_text_is_part_of_the_checkpoint_when_it_ends_the_flow() -> None:
    check = build(opening_run()).checkpoint
    assert "SUB-ACCOUNT OPENED" in check.text_present


# --- the review view ---------------------------------------------------------------------------


def test_the_review_view_says_what_it_does_needs_returns_and_risks() -> None:
    from cua.artifact.render import render_review

    text = render_review(build(opening_run()))
    assert "member_lookup 1.0.0" in text
    assert "draft" in text
    assert "Needs" in text
    assert "member_id" in text
    assert "Member number" in text
    assert "MOCK_USER" in text
    assert "Returns" in text
    assert "savings_balance" in text
    assert "s9" in text
    assert "WRITE" in text  # the risky step is unmissable
    assert "role_name" not in text or "ancestor_anchor" in text
    assert "ancestor_anchor" in text
    assert "no_such_member" in text
    assert "business outcome" in text.lower()
    assert "demo-only" not in text


def test_the_review_view_flags_a_capability_with_no_writes_as_read_only() -> None:
    from cua.artifact.render import render_review

    assert "read-only" in render_review(build())


# --- saving and loading ------------------------------------------------------------------------


def test_a_capability_is_saved_under_its_id_and_loads_back_identically(tmp_path: Path) -> None:
    from cua.artifact.store import list_capabilities, load_capability, save_capability

    cap = build()
    path = save_capability(cap, tmp_path / "capabilities")
    assert path.name == "member_lookup.json"
    assert load_capability(path) == cap
    assert [c.id for c in list_capabilities(tmp_path / "capabilities")] == ["member_lookup"]


def test_saving_refuses_a_capability_that_would_leak_a_secret(tmp_path: Path) -> None:
    from cua.artifact.store import save_capability

    cap = build().model_copy(update={"description": "password=hunter2 works"})
    with pytest.raises(ArtifactLeak):
        save_capability(cap, tmp_path / "capabilities", Redactor())
    assert not (tmp_path / "capabilities" / "member_lookup.json").exists()


def test_a_role_name_locator_is_carried_over_unchanged() -> None:
    run = recorded()
    role = LocatorBundle(
        description="link 'Loans'",
        frame_path=[FrameSelector(name="nav")],
        strategies=[
            RoleNameLocator(role="link", name="Loans", exact=True, rationale="accessible name")
        ],
    )
    run.steps[3] = run.steps[3].model_copy(update={"locator": role})
    assert build(run).steps[3].locator == role
