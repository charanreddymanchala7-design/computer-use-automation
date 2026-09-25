"""The discovery loop's mechanics: tools, binding of parameters and secrets, recording, and the
limits that stop a model that is stuck. Fake surface, scripted model, real gateway and policy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.agent_support import (
    BASE,
    FakeRecordingSurface,
    element,
    frame,
    make_gateway,
    page,
)

from cua.agent import (
    SYSTEM_PROMPT,
    DiscoveryLoop,
    DiscoveryTask,
    Limits,
    OutputSpec,
    ParamSpec,
    RecordedRun,
)
from cua.artifact import (
    CoordinatesLocator,
    LiteralValue,
    ParamRef,
    RiskClass,
    SecretRef,
)
from cua.evlog import EventLog
from cua.llm import (
    FakeLLM,
    ImagePart,
    LLMRequest,
    LLMResponse,
    Message,
    TextPart,
    ToolResult,
    ToolUse,
    Usage,
    say,
    tool_call,
)
from cua.surface import Action

TASK = DiscoveryTask(
    goal="Find the current savings balance of a member",
    start_url=f"{BASE}/msv/login.cgi",
    params={"member_id": ParamSpec("12345", "Member number")},
    outputs={"savings_balance": OutputSpec("Current savings balance")},
    secrets=("MOCK_USER", "MOCK_PASS"),
)


def elements() -> list:  # type: ignore[type-arg]
    return [
        element("e1", "input", role="textbox", label_hint="Member No:", name="F1"),
        element("e2", "img", "Go", alt="Go", onclick="doSearch()"),
        element("e3", "a", "Close", role="link", href="javascript:closeAcct('x')"),
        element("e4", "input", role="textbox", type="password", name="p"),
        element("e5", "input", role="textbox", name="u"),
        element("e6", "select", "Share Savings", role="combobox", name="F7"),
        element("e7", "a", "Loans", role="link", href="javascript:go('LN')"),
    ]


def screens(count: int = 30) -> list:  # type: ignore[type-arg]
    """Distinct pages, so a test that acts many times is not mistaken for a stalled model."""
    return [page(frame(1, "main", f"screen {i}", elements())) for i in range(count)]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Ran:
    run: RecordedRun
    llm: FakeLLM
    surface: FakeRecordingSurface
    log: EventLog
    clock: Clock

    def result_text(self, request_index: int, part: int = 0) -> str:
        """The text of a tool result the model was sent as part of a request."""
        message = self.llm.calls[request_index].messages[-1]
        result = message.parts[part]
        assert isinstance(result, ToolResult)
        return "".join(p.text for p in result.content if isinstance(p, TextPart))


def go(
    tmp_path: Path,
    script: Sequence[LLMResponse | Callable[[LLMRequest], LLMResponse]],
    *,
    observations: Sequence[object] | None = None,
    task: DiscoveryTask = TASK,
    limits: Limits | None = None,
    configure: Callable[[FakeRecordingSurface], None] | None = None,
) -> Ran:
    surface = FakeRecordingSurface(observations or screens())  # type: ignore[arg-type]
    if configure:
        configure(surface)
    gateway, log = make_gateway(surface, tmp_path)
    llm = FakeLLM(script)
    clock = Clock()
    loop = DiscoveryLoop(
        surface, gateway, llm, log, limits=limits or Limits(), clock=clock, run_id="run_t"
    )
    return Ran(loop.run(task), llm, surface, log, clock)


FINISH = tool_call("finish", success=True, summary="done")


def act(**args: object) -> LLMResponse:
    args.setdefault("reason", "because")
    return tool_call("act", **args)


def texts(message: Message) -> str:
    return "\n".join(p.text for p in message.parts if isinstance(p, TextPart))


# --- the first turn ----------------------------------------------------------------------------


def test_the_model_is_given_the_goal_parameters_outputs_secret_names_and_the_first_page(
    tmp_path: Path,
) -> None:
    ran = go(tmp_path, [FINISH])
    first = ran.llm.calls[0]
    text = texts(first.messages[0])
    assert TASK.goal in text
    assert "member_id" in text
    assert "12345" in text
    assert "Member number" in text
    assert "savings_balance" in text
    assert "MOCK_USER" in text
    assert "MOCK_PASS" in text
    assert "e1 textbox" in text  # the first observation
    assert any(isinstance(p, ImagePart) for p in first.messages[0].parts)
    assert first.system == SYSTEM_PROMPT
    assert [t.name for t in first.tools] == ["observe", "act", "extract", "finish"]


def test_the_start_page_is_opened_through_the_gateway_and_recorded_as_the_first_step(
    tmp_path: Path,
) -> None:
    ran = go(tmp_path, [FINISH])
    assert ran.surface.acts[0] == Action.navigate(TASK.start_url)
    first = ran.run.steps[0]
    assert (first.id, first.kind, first.url) == ("s1", "navigate", TASK.start_url)
    assert first.risk is RiskClass.READ


def test_a_start_page_outside_the_allowlist_ends_the_run_before_the_model_is_asked(
    tmp_path: Path,
) -> None:
    task = DiscoveryTask(**{**TASK.__dict__, "start_url": "http://evil.example/"})
    ran = go(tmp_path, [FINISH], task=task)
    assert ran.run.outcome == "failed"
    assert "start page" in ran.run.reason
    assert ran.llm.calls == []
    assert ran.surface.acts == []


# --- recording actions -------------------------------------------------------------------------


def test_the_locator_is_harvested_before_the_action_because_the_page_changes_after(
    tmp_path: Path,
) -> None:
    ran = go(tmp_path, [act(kind="click", ref="e2"), FINISH])
    assert ran.surface.calls == ["act", "observe", "harvest", "act", "observe"]
    step = ran.run.steps[1]
    assert (step.id, step.kind, step.description) == ("s2", "click", "because")
    assert step.locator is not None
    assert step.locator.description == "element e2"
    assert ran.surface.harvest_calls == [("e2", {"member_id": "12345"}, True)]
    assert step.after is not None


def test_a_click_result_carries_the_new_page_and_a_fresh_screenshot(tmp_path: Path) -> None:
    ran = go(tmp_path, [act(kind="click", ref="e2"), FINISH])
    request = ran.llm.calls[1]
    result = request.messages[-1].parts[0]
    assert isinstance(result, ToolResult)
    assert not result.is_error
    body = "".join(p.text for p in result.content if isinstance(p, TextPart))
    assert body.startswith("OK: did click")
    assert "screen 1" in body
    assert any(isinstance(p, ImagePart) for p in result.content)


def test_a_parameter_is_typed_from_its_value_and_recorded_by_name(tmp_path: Path) -> None:
    ran = go(tmp_path, [act(kind="fill", ref="e1", param="member_id"), FINISH])
    assert ran.surface.acts[1] == Action.fill("e1", text="12345")
    step = ran.run.steps[1]
    assert step.value == ParamRef(name="member_id")
    assert ran.surface.harvest_calls[0][1] == {"member_id": "12345"}


def test_a_secret_is_typed_by_name_and_recorded_by_name(tmp_path: Path) -> None:
    ran = go(tmp_path, [act(kind="fill", ref="e5", secret="MOCK_USER"), FINISH])
    assert ran.surface.acts[1] == Action.fill("e5", secret="MOCK_USER")
    assert ran.run.steps[1].value == SecretRef(name="MOCK_USER")
    assert ran.run.secrets_used == ["MOCK_USER"]


def test_literal_text_that_equals_a_parameter_value_is_recorded_as_the_parameter(
    tmp_path: Path,
) -> None:
    ran = go(tmp_path, [act(kind="fill", ref="e1", text="12345"), FINISH])
    assert ran.run.steps[1].value == ParamRef(
        name="member_id"
    )  # otherwise replay is stuck on 12345


def test_a_sensitive_parameter_is_typed_but_never_shown_to_the_model_or_kept_in_the_run(
    tmp_path: Path,
) -> None:
    task = DiscoveryTask(
        goal="g",
        start_url=f"{BASE}/msv/login.cgi",
        params={
            "member_id": ParamSpec("12345", "Member number"),
            "tin": ParamSpec("000-00-0001", "Tax id", sensitive=True),
        },
    )
    ran = go(tmp_path, [act(kind="fill", ref="e1", param="tin"), FINISH], task=task)
    prompt = texts(ran.llm.calls[0].messages[0])
    assert "000-00-0001" not in prompt
    assert "tin = <withheld>" in prompt
    assert "member_id = 12345" in prompt  # only sensitive values are withheld
    assert ran.surface.acts[1] == Action.fill("e1", text="000-00-0001")  # it is still typed
    assert ran.run.params == {"member_id": "12345", "tin": "<withheld>"}
    assert "000-00-0001" not in ran.run.model_dump_json()
    assert ran.run.steps[1].value == ParamRef(name="tin")


def test_a_constant_stays_a_literal(tmp_path: Path) -> None:
    ran = go(tmp_path, [act(kind="fill", ref="e1", text="college fund"), FINISH])
    assert ran.run.steps[1].value == LiteralValue(value="college fund")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"kind": "fill", "ref": "e1"}, "exactly one of text, secret or param"),
        ({"kind": "fill", "ref": "e1", "text": "a", "param": "member_id"}, "exactly one"),
        ({"kind": "fill", "ref": "e1", "param": "nope"}, "unknown param 'nope'"),
        ({"kind": "fill", "ref": "e5", "secret": "NOPE"}, "unknown secret 'NOPE'"),
        ({"kind": "click"}, "click needs a ref"),
        ({"kind": "select", "ref": "e6"}, "select needs an option"),
        ({"kind": "navigate"}, "navigate needs a url"),
        ({"kind": "press"}, "press needs a key"),
        ({"kind": "wait_for"}, "wait_for needs text"),
        ({"kind": "juggle"}, "unknown kind 'juggle'"),
    ],
)
def test_a_badly_formed_action_is_an_error_the_model_can_correct(
    tmp_path: Path, args: dict[str, object], message: str
) -> None:
    ran = go(tmp_path, [act(**args), FINISH])
    assert ran.result_text(1).startswith("ERROR:")
    assert message in ran.result_text(1)
    assert len(ran.surface.acts) == 1  # only the start page was opened
    assert len(ran.run.steps) == 1


def test_an_unknown_tool_is_an_error(tmp_path: Path) -> None:
    ran = go(tmp_path, [tool_call("teleport"), FINISH])
    assert "unknown tool 'teleport'" in ran.result_text(1)


def test_select_press_and_navigate_are_recorded_with_their_values(tmp_path: Path) -> None:
    script = [
        act(kind="select", ref="e6", option="Holiday Club"),
        act(kind="press", key="Enter"),
        act(kind="navigate", url=f"{BASE}/msv/member.cgi?mid=12345"),
        FINISH,
    ]
    ran = go(tmp_path, script)
    select, press, navigate = ran.run.steps[1:]
    assert select.value == LiteralValue(value="Holiday Club")
    assert (press.kind, press.value, press.risk) == (
        "press",
        LiteralValue(value="Enter"),
        RiskClass.REVERSIBLE_WRITE,
    )
    assert navigate.url == f"{BASE}/msv/member.cgi?mid={{member_id}}"  # the parameter, not 12345


def test_a_click_by_coordinates_is_recorded_as_a_coordinate_locator(tmp_path: Path) -> None:
    ran = go(tmp_path, [act(kind="click", x=40.0, y=25.0), FINISH])
    assert ran.surface.acts[1] == Action.click_at(40.0, 25.0)
    step = ran.run.steps[1]
    assert step.locator is not None
    strategy = step.locator.strategies[0]
    assert isinstance(strategy, CoordinatesLocator)
    assert (strategy.x, strategy.y, strategy.viewport_width) == (40.0, 25.0, 1280)
    assert step.risk is RiskClass.REVERSIBLE_WRITE  # a blind click cannot be vouched for


def test_a_wait_is_a_pause_not_a_recorded_step(tmp_path: Path) -> None:
    ran = go(tmp_path, [act(kind="wait", ms=200), FINISH])
    assert len(ran.run.steps) == 1
    assert ran.surface.acts[1] == Action.wait(200)


def test_waiting_for_text_is_recorded_as_an_expectation_and_a_timeout_is_an_error(
    tmp_path: Path,
) -> None:
    ran = go(tmp_path, [act(kind="wait_for", text="SUB-ACCOUNT OPENED"), FINISH])
    step = ran.run.steps[1]
    assert (step.kind, step.expect_text) == ("wait_for", "SUB-ACCOUNT OPENED")

    def slow(surface: FakeRecordingSurface) -> None:
        surface.wait_ok = False

    timed_out = go(tmp_path, [act(kind="wait_for", text="never"), FINISH], configure=slow)
    assert "did not appear" in timed_out.result_text(1)
    assert len(timed_out.run.steps) == 1


def test_dialogs_handled_during_an_action_are_recorded_on_the_step(tmp_path: Path) -> None:
    from cua.surface import DialogEvent

    def with_dialog(surface: FakeRecordingSurface) -> None:
        surface.dialogs = [DialogEvent("confirm", "Open new sub-account for member 12345?", True)]

    ran = go(tmp_path, [act(kind="click", ref="e2"), FINISH], configure=with_dialog)
    (dialog,) = ran.run.steps[1].dialogs
    assert (dialog.kind, dialog.accepted) == ("confirm", True)


# --- what does not get recorded ----------------------------------------------------------------


def test_an_irreversible_action_is_refused_told_why_and_never_recorded(tmp_path: Path) -> None:
    ran = go(tmp_path, [act(kind="click", ref="e3"), FINISH])
    text = ran.result_text(1)
    assert text.startswith("BLOCKED:")
    assert "needs a human's confirmation" in text
    assert "finish" in text
    assert len(ran.surface.acts) == 1
    assert len(ran.run.steps) == 1


def test_a_navigation_off_the_allowlist_is_refused(tmp_path: Path) -> None:
    ran = go(tmp_path, [act(kind="navigate", url="http://evil.example/x"), FINISH])
    assert ran.result_text(1).startswith("BLOCKED:")
    assert "host_not_allowed" in ran.result_text(1)
    assert len(ran.run.steps) == 1


def test_a_click_that_sends_the_page_somewhere_forbidden_is_refused_and_not_recorded(
    tmp_path: Path,
) -> None:
    def admin_link(surface: FakeRecordingSurface) -> None:
        surface.requests_on_click = [f"{BASE}/msv/admin.cgi"]

    ran = go(tmp_path, [act(kind="click", ref="e7"), FINISH], configure=admin_link)
    text = ran.result_text(1)
    assert text.startswith("BLOCKED:")
    assert "/msv/admin.cgi" in text
    assert "not recorded" in text
    assert len(ran.run.steps) == 1  # replaying that click would hit the same wall


def test_a_failed_action_is_reported_and_not_recorded(tmp_path: Path) -> None:
    def failing(surface: FakeRecordingSurface) -> None:
        surface.ok = False
        surface.act_error = "Timeout 8000ms exceeded"

    ran = go(tmp_path, [act(kind="click", ref="e2"), FINISH], configure=failing)
    # the start-page navigation failed too, which ends the run before the model is asked
    assert ran.run.outcome == "failed"


def test_a_click_that_fails_after_a_good_start_is_an_error_the_model_sees(tmp_path: Path) -> None:
    class FlakySurface(FakeRecordingSurface):
        def act(self, action: Action):  # type: ignore[no-untyped-def]
            result = super().act(action)
            if action.kind == "click":
                result.ok, result.error = False, "Timeout 8000ms exceeded"
            return result

    surface = FlakySurface(screens())
    gateway, log = make_gateway(surface, tmp_path)
    llm = FakeLLM([act(kind="click", ref="e2"), FINISH])
    run = DiscoveryLoop(surface, gateway, llm, log, clock=Clock(), run_id="r").run(TASK)
    result = llm.calls[1].messages[-1].parts[0]
    assert isinstance(result, ToolResult)
    assert result.is_error
    assert "Timeout 8000ms exceeded" in "".join(
        p.text for p in result.content if isinstance(p, TextPart)
    )
    assert len(run.steps) == 1


def test_an_element_with_no_reliable_locator_is_refused_before_anything_happens(
    tmp_path: Path,
) -> None:
    def cannot_locate(surface: FakeRecordingSurface) -> None:
        surface.harvest_errors["e2"] = "no reliable locator for img 'Go': try a different element"

    ran = go(tmp_path, [act(kind="click", ref="e2"), FINISH], configure=cannot_locate)
    assert "no reliable locator" in ran.result_text(1)
    assert len(ran.surface.acts) == 1  # never clicked something replay could not find again
    assert len(ran.run.steps) == 1


# --- extracting outputs ------------------------------------------------------------------------


def known(surface: FakeRecordingSurface) -> None:
    surface.known_values = {"$2,480.15"}


def extract(**args: object) -> LLMResponse:
    args.setdefault("reason", "read it")
    return tool_call("extract", **args)


def test_an_extracted_value_is_located_verified_read_back_and_recorded(tmp_path: Path) -> None:
    call = extract(name="savings_balance", value="$2,480.15", anchor_text="SHARE SAVINGS")
    ran = go(tmp_path, [call, FINISH], configure=known)
    step = ran.run.steps[1]
    assert (step.kind, step.output) == ("extract", "savings_balance")
    assert step.locator is not None
    assert ran.run.outputs == {"savings_balance": "$2,480.15"}
    assert ran.result_text(1).startswith("OK:")
    assert "resolve" in ran.surface.calls  # the locator was proven by reading the value back


def test_an_extraction_whose_read_back_differs_is_refused(tmp_path: Path) -> None:
    def wrong(surface: FakeRecordingSurface) -> None:
        known(surface)
        surface.read_back = "$1.00"

    call = extract(name="savings_balance", value="$2,480.15", anchor_text="SHARE SAVINGS")
    ran = go(tmp_path, [call, FINISH], configure=wrong)
    assert "read back" in ran.result_text(1)
    assert ran.run.outputs == {}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"name": "nope", "value": "$2,480.15"}, "unknown output 'nope'"),
        ({"name": "savings_balance"}, "extract needs a value"),
        ({"value": "$2,480.15"}, "extract needs a name"),
        ({"name": "savings_balance", "value": "$9.99"}, "not found"),
    ],
)
def test_a_bad_extraction_explains_what_to_change(
    tmp_path: Path, args: dict[str, object], message: str
) -> None:
    ran = go(tmp_path, [extract(**args), FINISH], configure=known)
    assert ran.result_text(1).startswith("ERROR:")
    assert message in ran.result_text(1)
    assert ran.run.outputs == {}


# --- finishing ---------------------------------------------------------------------------------


def test_finish_with_success_ends_the_run_with_the_models_summary(tmp_path: Path) -> None:
    ran = go(tmp_path, [tool_call("finish", success=True, summary="Balance read: $2,480.15")])
    assert (ran.run.outcome, ran.run.summary) == ("finished", "Balance read: $2,480.15")


def test_finish_without_success_is_a_failed_run_not_a_finished_one(tmp_path: Path) -> None:
    ran = go(tmp_path, [tool_call("finish", success=False, summary="the member was frozen")])
    assert ran.run.outcome == "failed"
    assert ran.run.reason == "the model reported that the goal was not achieved"
    assert ran.run.summary == "the member was frozen"


def test_the_observe_tool_returns_a_fresh_page_and_screenshot(tmp_path: Path) -> None:
    ran = go(tmp_path, [tool_call("observe"), FINISH])
    result = ran.llm.calls[1].messages[-1].parts[0]
    assert isinstance(result, ToolResult)
    assert "screen 1" in ran.result_text(1)
    assert any(isinstance(p, ImagePart) for p in result.content)
    assert len(ran.run.steps) == 1  # looking is not doing


# --- limits and dead ends ----------------------------------------------------------------------


def test_the_run_stops_at_the_step_limit(tmp_path: Path) -> None:
    ran = go(tmp_path, [tool_call("observe")] * 5, limits=Limits(max_steps=3))
    assert ran.run.outcome == "max_steps"
    assert len(ran.llm.calls) == 3


def test_the_run_stops_at_the_time_limit(tmp_path: Path) -> None:
    def slow(request: LLMRequest) -> LLMResponse:
        ran_clock.now += 400
        return tool_call("observe")

    ran_clock = Clock()
    surface = FakeRecordingSurface(screens())
    gateway, log = make_gateway(surface, tmp_path)
    llm = FakeLLM([slow, slow, slow])
    loop = DiscoveryLoop(
        surface, gateway, llm, log, limits=Limits(timeout_s=300), clock=ran_clock, run_id="r"
    )
    run = loop.run(TASK)
    assert run.outcome == "timeout"
    assert len(llm.calls) == 1


def test_repeating_the_same_action_is_a_dead_end(tmp_path: Path) -> None:
    ran = go(tmp_path, [act(kind="click", ref="e2")] * 6, limits=Limits(max_repeats=3))
    assert ran.run.outcome == "dead_end"
    assert "same action" in ran.run.reason
    assert len(ran.llm.calls) == 3


def test_a_run_of_errors_is_a_dead_end(tmp_path: Path) -> None:
    ran = go(
        tmp_path, [act(kind="click", ref="zzz")] * 8, limits=Limits(max_errors=4, max_repeats=99)
    )
    assert ran.run.outcome == "dead_end"
    assert "errors in a row" in ran.run.reason


def test_a_page_that_never_changes_is_no_progress(tmp_path: Path) -> None:
    refs = ["e1", "e2", "e5", "e6", "e7", "e4", "e1", "e2"]
    script = [act(kind="click", ref=ref) for ref in refs]
    ran = go(
        tmp_path,
        script,
        observations=[page(frame(1, "main", "same", elements()))],
        limits=Limits(max_stalled=4, max_repeats=99),
    )
    assert ran.run.outcome == "dead_end"
    assert "no progress" in ran.run.reason


def test_a_text_only_reply_is_nudged_and_the_run_can_recover(tmp_path: Path) -> None:
    ran = go(tmp_path, [say("hmm, let me think"), FINISH])
    assert ran.run.outcome == "finished"
    nudge = texts(ran.llm.calls[1].messages[-1])
    assert "Call a tool" in nudge


def test_a_model_that_keeps_talking_without_acting_is_a_dead_end(tmp_path: Path) -> None:
    ran = go(tmp_path, [say("a"), say("b"), say("c"), say("d")], limits=Limits(max_nudges=2))
    assert ran.run.outcome == "dead_end"
    assert "stopped calling tools" in ran.run.reason


# --- the conversation the model sees -----------------------------------------------------------


def test_only_the_latest_screenshot_is_kept_in_the_history(tmp_path: Path) -> None:
    script = [
        act(kind="click", ref="e2"),
        act(kind="click", ref="e7"),
        act(kind="click", ref="e1"),
        FINISH,
    ]
    ran = go(tmp_path, script)
    last = ran.llm.calls[-1].messages
    images = [
        part
        for message in last
        for part in message.parts
        if isinstance(part, ImagePart)
        or (isinstance(part, ToolResult) and any(isinstance(p, ImagePart) for p in part.content))
    ]
    assert len(images) == 1
    earlier = "".join(
        p.text
        for m in last[:-1]
        for part in m.parts
        if isinstance(part, ToolResult)
        for p in part.content
        if isinstance(p, TextPart)
    )
    assert "earlier screenshot omitted" in earlier


def test_every_tool_call_in_one_reply_is_answered_in_order(tmp_path: Path) -> None:
    both = LLMResponse(
        text="",
        tool_calls=(
            ToolUse("tu_a", "observe", {}),
            ToolUse("tu_b", "act", {"kind": "click", "ref": "e2", "reason": "go"}),
        ),
        stop_reason="tool_use",
    )
    ran = go(tmp_path, [both, FINISH])
    parts = ran.llm.calls[1].messages[-1].parts
    assert [p.tool_use_id for p in parts if isinstance(p, ToolResult)] == ["tu_a", "tu_b"]


def test_usage_and_the_step_count_are_carried_into_the_run(tmp_path: Path) -> None:
    metered = LLMResponse(
        text="",
        tool_calls=(ToolUse("t", "observe", {}),),
        usage=Usage(1000, 100),
        stop_reason="tool_use",
    )
    ran = go(tmp_path, [metered, FINISH])
    assert ran.run.llm_steps == 2
    assert ran.run.tokens == 1100


def test_the_recorded_run_serializes_and_never_holds_a_secret_value(tmp_path: Path) -> None:
    script = [
        act(kind="fill", ref="e5", secret="MOCK_USER"),
        act(kind="fill", ref="e4", secret="MOCK_PASS"),
        act(kind="fill", ref="e1", param="member_id"),
        FINISH,
    ]
    ran = go(tmp_path, script)
    text = ran.run.model_dump_json()
    assert RecordedRun.model_validate_json(text) == ran.run
    assert "demo-only" not in text
    assert ran.run.secrets_used == ["MOCK_USER", "MOCK_PASS"]
    assert ran.run.params == {"member_id": "12345"}


# --- keeping what the model saw ------------------------------------------------------------------


def test_every_page_shown_to_the_model_can_be_kept_as_a_numbered_screenshot(tmp_path: Path) -> None:
    kept: list[tuple[str, bytes]] = []
    surface = FakeRecordingSurface(screens())
    gateway, log = make_gateway(surface, tmp_path)
    loop = DiscoveryLoop(
        surface,
        gateway,
        FakeLLM([act(kind="click", ref="e2"), FINISH]),
        log,
        on_screenshot=lambda label, png: kept.append((label, png)),
    )
    loop.run(TASK)
    assert [label for label, _ in kept] == ["00-start", "01-act"]
    assert all(png.startswith(b"\x89PNG") for _, png in kept)


def test_screenshots_are_optional(tmp_path: Path) -> None:
    assert go(tmp_path, [FINISH]).run.outcome == "finished"
