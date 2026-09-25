"""Regulated data must never reach a log or an artifact, however it gets there."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from cua.artifact import Capability
from cua.evlog import ArtifactLeak, EventLog, read_events, write_artifact, write_redacted_json
from cua.redact import Redactor

R = Redactor()

SENSITIVE_SHAPES = [
    ("ssn 123-45-6789 on file", "123-45-6789"),
    ("acct 1234567890123456 charged", "1234567890123456"),
    ("key sk-fake-abcdefghijklmnop0123456789", "abcdefghijklmnop0123456789"),
    ("token ghp_abcdefghijklmnopqrstuvwx", "abcdefghijklmnopqrstuvwx"),
    ("id AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    ("Authorization: Bearer abcdefghijklmnop0123456789", "abcdefghijklmnop0123456789"),
    (
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdEFGHijkl",
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0",
    ),
    ("password=hunter2", "hunter2"),
    ('{"api_key": "abc123xyz"}', "abc123xyz"),
    ("pwd: s3cret!", "s3cret!"),
]


# --- text -------------------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "secret"), SENSITIVE_SHAPES)
def test_sensitive_shapes_are_masked(text: str, secret: str) -> None:
    out = R.redact_text(text)
    assert secret not in out
    assert "<redacted" in out


@pytest.mark.parametrize(
    "text",
    [
        "Balance $1,250.00 for member 12345",
        "Search results",
        "Opened 2026-09-25T00:00:00Z",
        "Call 555-0123 or (555) 010-0123",
        "Step s3 clicked Search",
    ],
)
def test_ordinary_content_is_left_alone(text: str) -> None:
    assert R.redact_text(text) == text


def test_registered_secrets_are_removed_wherever_they_appear() -> None:
    r = Redactor(secrets=["hunter2", "Tr0ub4dor&3"])
    out = r.redact_text("login with hunter2 then Tr0ub4dor&3 failed")
    assert "hunter2" not in out
    assert "Tr0ub4dor" not in out


def test_very_short_registered_secrets_are_ignored_so_text_is_not_shredded() -> None:
    assert Redactor(secrets=["ab", ""]).redact_text("about a lab") == "about a lab"


def test_mask_id_keeps_only_the_last_two_characters() -> None:
    assert R.mask_id("12345") == "***45"
    assert R.mask_id("1234") == "***"
    assert R.mask_id("") == "***"


def test_placeholder_reveals_only_the_length() -> None:
    assert R.placeholder("hunter2") == "<redacted len=7>"


# --- structures -------------------------------------------------------------------------------


def test_sensitive_keys_are_masked_whatever_the_value_type() -> None:
    data = {"Password": "x", "nested": {"api_key": 12345}, "list": [{"token": ["a", "b"]}]}
    assert R.redact(data) == {
        "Password": "<redacted>",
        "nested": {"api_key": "<redacted>"},
        "list": [{"token": "<redacted>"}],
    }


def test_extra_sensitive_keys_and_masked_id_keys_are_configurable() -> None:
    r = Redactor(sensitive_keys={"member_number"}, masked_id_keys={"account_id"})
    assert r.redact({"member_number": "12345", "account_id": "9876543210", "step": "s1"}) == {
        "member_number": "<redacted>",
        "account_id": "***10",
        "step": "s1",
    }


def test_redact_walks_lists_tuples_and_leaves_scalars_alone() -> None:
    data = {"items": ("ssn 123-45-6789", 7, 2.5, True, None)}
    assert R.redact(data) == {"items": ["ssn <redacted:ssn>", 7, 2.5, True, None]}


def test_key_masking_can_be_switched_off_for_structural_documents() -> None:
    # A JSON Schema legitimately has properties *named* password or member_id.
    schema = {"properties": {"password": {"type": "string", "x-sensitive": True}, "member_id": {}}}
    assert R.redact(schema, by_key=False) == schema
    assert R.redact({"note": "ssn 123-45-6789"}, by_key=False) == {"note": "ssn <redacted:ssn>"}


def test_redact_never_mutates_its_input() -> None:
    data = {"a": ["password=hunter2"], "password": "x"}
    before = copy.deepcopy(data)
    R.redact(data)
    assert data == before


# --- the event log ----------------------------------------------------------------------------

FIXED = datetime(2026, 9, 25, 12, 0, 0, tzinfo=UTC)


def make_log(tmp_path: Path, redactor: Redactor | None = None) -> EventLog:
    return EventLog(
        tmp_path / "logs" / "run.jsonl",
        run_id="run_001",
        redactor=redactor or Redactor(),
        clock=lambda: FIXED,
    )


def test_each_event_is_one_json_line_with_the_debugging_fields(tmp_path: Path) -> None:
    log = make_log(tmp_path)
    log.emit(
        "act", step="s2", action="fill", target="Member # field", outcome="ok", reason="typed id"
    )
    (line,) = (tmp_path / "logs" / "run.jsonl").read_text().splitlines()
    event = json.loads(line)
    assert event == {
        "ts": "2026-09-25T12:00:00+00:00",
        "run_id": "run_001",
        "event": "act",
        "step": "s2",
        "action": "fill",
        "target": "Member # field",
        "outcome": "ok",
        "reason": "typed id",
    }


def test_the_default_clock_stamps_events_in_timezone_aware_utc(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run.jsonl", run_id="run_001", redactor=Redactor())
    log.emit("start")
    stamp = datetime.fromisoformat(read_events(log.path)[0]["ts"])
    assert stamp.utcoffset() is not None
    assert stamp.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


def test_the_log_exposes_its_redactor_so_callers_redact_the_same_way(tmp_path: Path) -> None:
    log = make_log(tmp_path, Redactor(secrets=["hunter2"]))
    assert "hunter2" not in log.redactor.redact_text("the word is hunter2")


def test_omitted_fields_are_left_out_not_null(tmp_path: Path) -> None:
    log = make_log(tmp_path)
    log.emit("start")
    assert read_events(log.path) == [
        {"ts": "2026-09-25T12:00:00+00:00", "run_id": "run_001", "event": "start"}
    ]


def test_events_append_across_log_instances(tmp_path: Path) -> None:
    make_log(tmp_path).emit("one")
    make_log(tmp_path).emit("two")
    assert [e["event"] for e in read_events(tmp_path / "logs" / "run.jsonl")] == ["one", "two"]


def test_secrets_never_reach_the_log_file(tmp_path: Path) -> None:
    log = make_log(tmp_path, Redactor(secrets=["hunter2"]))
    log.emit(
        "act",
        target="password field hunter2",
        reason="ssn 123-45-6789 shown; key sk-fake-abcdefghijklmnop0123456789",
        observed="acct 1234567890123456",
        extra={"password": "hunter2", "note": "hunter2 again"},
    )
    text = log.path.read_text()
    for leaked in ("hunter2", "123-45-6789", "abcdefghijklmnop0123456789", "1234567890123456"):
        assert leaked not in text


def test_id_fields_are_masked_in_the_log(tmp_path: Path) -> None:
    log = make_log(tmp_path)
    log.emit("act", member_id="12345")
    (event,) = read_events(log.path)
    assert event["member_id"] == "***45"


def test_typed_values_for_sensitive_fields_are_logged_as_length_only(tmp_path: Path) -> None:
    log = make_log(tmp_path)
    log.emit("act", action="fill", typed="hunter2", sensitive=True)
    log.emit("act", action="fill", typed="Search", sensitive=False)
    first, second = read_events(log.path)
    assert first["typed"] == "<redacted len=7>"
    assert second["typed"] == "Search"


# --- artifacts --------------------------------------------------------------------------------


def test_a_clean_capability_is_written_unchanged(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    cap = Capability.model_validate(capability_dict)
    path = tmp_path / "capabilities" / "member_lookup.json"
    write_artifact(path, cap, Redactor())
    assert Capability.model_validate_json(path.read_text()) == cap


def test_a_capability_with_a_sensitive_input_can_still_be_written(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    capability_dict["inputs"]["properties"]["password"] = {"type": "string", "x-sensitive": True}
    cap = Capability.model_validate(capability_dict)
    path = tmp_path / "with_password.json"
    write_artifact(path, cap, Redactor())
    assert Capability.model_validate_json(path.read_text()) == cap


def test_evidence_files_are_redacted_not_refused(tmp_path: Path) -> None:
    path = tmp_path / "evidence" / "result.json"
    write_redacted_json(
        path, {"outputs": {"password": "hunter2", "note": "ssn 123-45-6789"}}, Redactor()
    )
    text = path.read_text()
    assert "hunter2" not in text
    assert "123-45-6789" not in text
    assert json.loads(text)["outputs"]["password"] == "<redacted>"


def test_a_planted_secret_stops_the_write_and_is_never_echoed(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    capability_dict["steps"][1]["description"] = "Type password=hunter2 into the box"
    path = tmp_path / "leaky.json"
    with pytest.raises(ArtifactLeak, match=r"\$\.steps\[1\]\.description") as info:
        write_artifact(path, capability_dict, Redactor())
    assert "hunter2" not in str(info.value)
    assert not path.exists()


def test_a_registered_secret_stops_the_write_too(tmp_path: Path) -> None:
    path = tmp_path / "leaky.json"
    with pytest.raises(ArtifactLeak):
        write_artifact(
            path, {"notes": ["fine", "the word is Tr0ub4dor&3"]}, Redactor(secrets=["Tr0ub4dor&3"])
        )
    assert not path.exists()


def test_a_planted_secret_appears_in_no_file_at_all(
    tmp_path: Path, capability_dict: dict[str, Any]
) -> None:
    planted = ["hunter2", "123-45-6789", "sk-fake-abcdefghijklmnop0123456789"]
    redactor = Redactor(secrets=["hunter2"])
    log = make_log(tmp_path, redactor)
    for value in planted:
        log.emit("observe", observed=f"page showed {value}", extra={"secret": value})
        capability_dict["description"] = f"contains {value}"
        with pytest.raises(ArtifactLeak):
            write_artifact(tmp_path / "artifact.json", capability_dict, redactor)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            content = path.read_text()
            assert not any(value in content for value in planted), path


def test_a_secret_is_removed_however_the_application_cases_it() -> None:
    # legacy screens shout: a header that shows the signed-in user upper-cased is still the secret
    redactor = Redactor(secrets=["teller01"])
    assert redactor.redact_text("User: TELLER01 Log Off") == "User: <redacted:secret> Log Off"
    assert redactor.redact_text("Teller01 teller01") == "<redacted:secret> <redacted:secret>"
