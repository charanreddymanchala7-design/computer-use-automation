"""The .env reader and the colour rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from cua.config import load_dotenv, use_color


def test_pairs_are_added_and_only_their_names_are_reported(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\nANTHROPIC_API_KEY=sk-test-123\n\nMOCK_USER='teller01'\n")
    env: dict[str, str] = {}
    assert load_dotenv(env_file, env) == ["ANTHROPIC_API_KEY", "MOCK_USER"]
    assert env == {"ANTHROPIC_API_KEY": "sk-test-123", "MOCK_USER": "teller01"}


def test_what_is_already_set_wins_over_the_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-file\n")
    env = {"ANTHROPIC_API_KEY": "from-shell"}
    assert load_dotenv(env_file, env) == []
    assert env["ANTHROPIC_API_KEY"] == "from-shell"


def test_empty_values_and_junk_lines_are_ignored(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('ANTHROPIC_API_KEY=\nnot a pair\nexport OTHER="x y"\n=nokey\n')
    env: dict[str, str] = {}
    load_dotenv(env_file, env)
    assert env == {"OTHER": "x y"}  # an empty key is never treated as configured


def test_a_missing_file_is_fine(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nope.env", {}) == []


def test_the_process_environment_is_used_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUA_TEST_ONLY_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("CUA_TEST_ONLY_KEY=abc\n")
    load_dotenv(env_file)
    import os

    assert os.environ["CUA_TEST_ONLY_KEY"] == "abc"
    monkeypatch.delenv("CUA_TEST_ONLY_KEY")


@pytest.mark.parametrize(
    ("isatty", "env", "expected"),
    [
        (True, {}, True),
        (False, {}, False),  # piped output stays plain
        (True, {"NO_COLOR": "1"}, False),
        (True, {"NO_COLOR": ""}, False),  # the convention: present at all means off
        (True, {"TERM": "dumb"}, False),
    ],
)
def test_colour_only_on_a_terminal_and_never_against_no_color(
    isatty: bool, env: dict[str, str], expected: bool
) -> None:
    assert use_color(isatty=isatty, env=env) is expected
