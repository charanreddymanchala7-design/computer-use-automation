"""The command line: what a person or a script sees. Argument handling and rendering are checked
without a browser; the end-to-end commands run in real Chromium (see test_cli_e2e.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from cua.artifact import Capability
from cua.artifact.store import save_capability
from cua.cli import app, coerce_inputs, find_capability

runner = CliRunner()


@pytest.fixture
def saved(tmp_path: Path, capability_dict: dict[str, Any]) -> Path:
    return save_capability(Capability.model_validate(capability_dict), tmp_path / "capabilities")


def test_help_names_every_command() -> None:
    out = runner.invoke(app, ["--help"])
    assert out.exit_code == 0
    for command in ("run", "show", "replay"):
        assert command in out.output


def test_show_prints_the_review_of_a_saved_capability(saved: Path) -> None:
    out = runner.invoke(app, ["show", str(saved)])
    assert out.exit_code == 0
    assert "Look up a member and read their savings balance" in out.output
    assert "s3" in out.output


def test_a_capability_can_be_named_instead_of_pathed(
    saved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(saved.parent.parent)
    out = runner.invoke(app, ["show", "member_lookup"])
    assert out.exit_code == 0
    assert "member_lookup" in out.output


def test_an_unknown_capability_is_a_clear_usage_error_not_a_traceback(tmp_path: Path) -> None:
    out = runner.invoke(app, ["show", str(tmp_path / "nope.json")])
    assert out.exit_code == 2
    assert "no capability" in out.output
    assert "Traceback" not in out.output


def test_a_file_that_is_not_a_capability_is_refused_plainly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"nope": 1}')
    out = runner.invoke(app, ["show", str(bad)])
    assert out.exit_code == 2
    assert "not a valid capability" in out.output


def test_a_param_without_an_equals_sign_is_refused(saved: Path) -> None:
    out = runner.invoke(app, ["replay", str(saved), "--param", "member_id"])
    assert out.exit_code == 2
    assert "name=value" in out.output


def test_an_unknown_operator_is_refused(saved: Path) -> None:
    out = runner.invoke(app, ["replay", str(saved), "--operator", "pigeon"])
    assert out.exit_code == 2
    assert "pigeon" in out.output


def test_a_missing_policy_file_is_a_usage_error(saved: Path, tmp_path: Path) -> None:
    out = runner.invoke(app, ["replay", str(saved), "--policy", str(tmp_path / "none.json")])
    assert out.exit_code == 2
    assert "policy" in out.output


# --- typed inputs -------------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {
        "member_id": {"type": "string"},
        "count": {"type": "integer"},
        "amount": {"type": "number"},
        "urgent": {"type": "boolean"},
    },
}


def test_inputs_are_typed_from_the_capabilitys_own_schema() -> None:
    typed = coerce_inputs(
        SCHEMA, ["member_id=12345", "count=3", "amount=25.50", "urgent=true", "extra=x"]
    )
    assert typed == {
        "member_id": "12345",  # a string stays a string, leading zeros and all
        "count": 3,
        "amount": 25.5,
        "urgent": True,
        "extra": "x",  # unknown names pass through so replay can refuse them by name
    }


@pytest.mark.parametrize("pair", ["count=three", "amount=lots", "urgent=maybe"])
def test_a_value_that_does_not_fit_its_type_is_refused_by_name(pair: str) -> None:
    with pytest.raises(typer.BadParameter, match=pair.split("=")[0]):
        coerce_inputs(SCHEMA, [pair])


def test_an_equals_sign_inside_a_value_is_kept() -> None:
    assert coerce_inputs(SCHEMA, ["member_id=a=b"]) == {"member_id": "a=b"}


def test_find_capability_prefers_a_real_path_then_the_capabilities_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: Path
) -> None:
    monkeypatch.chdir(saved.parent.parent)
    assert find_capability(str(saved)).id == "member_lookup"
    assert find_capability("member_lookup").id == "member_lookup"
    with pytest.raises(typer.BadParameter, match="no capability"):
        find_capability("missing")


def test_json_flag_is_documented() -> None:
    out = runner.invoke(app, ["replay", "--help"])
    assert "--json" in out.output
    assert "NO_COLOR" in out.output or "--no-color" in out.output
    assert json.loads('{"a": 1}') == {"a": 1}


# --- the agent-facing catalog -------------------------------------------------------------------


def test_capabilities_list_shows_what_each_one_does_to_the_application(saved: Path) -> None:
    out = runner.invoke(app, ["capabilities", "list", "--dir", str(saved.parent)])
    assert out.exit_code == 0
    assert "member_lookup" in out.output
    assert "1.0.0" in out.output
    assert "read-only" in out.output
    assert "member_id" in out.output


def test_capabilities_list_says_when_there_are_none(tmp_path: Path) -> None:
    out = runner.invoke(app, ["capabilities", "list", "--dir", str(tmp_path)])
    assert out.exit_code == 0
    assert "no capabilities" in out.output


def test_capabilities_schema_prints_the_tool_definition(saved: Path) -> None:
    out = runner.invoke(
        app, ["capabilities", "schema", "member_lookup", "--dir", str(saved.parent)]
    )
    assert out.exit_code == 0
    tool = json.loads(out.output)
    assert tool["name"] == "member_lookup"
    assert tool["inputSchema"]["required"] == ["member_id"]


def test_capabilities_schema_of_an_unknown_tool_lists_the_real_ones(saved: Path) -> None:
    out = runner.invoke(app, ["capabilities", "schema", "nope", "--dir", str(saved.parent)])
    assert out.exit_code == 2
    assert "member_lookup" in out.output


def test_export_writes_the_catalog_and_check_notices_when_it_goes_stale(
    saved: Path, tmp_path: Path
) -> None:
    target = tmp_path / "catalog.json"
    args = ["capabilities", "export", "--dir", str(saved.parent), "--out", str(target)]
    assert runner.invoke(app, args).exit_code == 0
    assert json.loads(target.read_text())["tools"][0]["name"] == "member_lookup"
    assert runner.invoke(app, [*args, "--check"]).exit_code == 0
    target.write_text('{"version": 1, "tools": []}\n')
    stale = runner.invoke(app, [*args, "--check"])
    assert stale.exit_code == 1
    assert "out of date" in stale.output


def test_check_fails_when_the_catalog_file_is_missing(saved: Path, tmp_path: Path) -> None:
    out = runner.invoke(
        app,
        [
            "capabilities",
            "export",
            "--dir",
            str(saved.parent),
            "--out",
            str(tmp_path / "missing.json"),
            "--check",
        ],
    )
    assert out.exit_code == 1


def test_call_refuses_arguments_that_are_not_json(saved: Path) -> None:
    out = runner.invoke(
        app,
        ["capabilities", "call", "member_lookup", "--args", "{oops", "--dir", str(saved.parent)],
    )
    assert out.exit_code == 2
    assert "JSON" in out.output


def test_call_refuses_arguments_that_do_not_fit_the_schema_before_any_browser_starts(
    saved: Path,
) -> None:
    out = runner.invoke(
        app,
        [
            "capabilities",
            "call",
            "member_lookup",
            "--args",
            '{"member_id": "abc"}',
            "--dir",
            str(saved.parent),
        ],
    )
    assert out.exit_code == 2
    assert "invalid arguments" in out.output
    assert "abc" not in out.output
