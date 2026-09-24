"""Scaffold guarantees: importable package, and nothing secret or raw can slip into the repo."""

from __future__ import annotations

import re
from pathlib import Path

import cua

ROOT = Path(__file__).resolve().parent.parent


def test_package_exposes_a_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", cua.__version__)


def test_gitignore_keeps_secrets_and_raw_artifacts_out() -> None:
    ignored = set((ROOT / ".gitignore").read_text().splitlines())
    required = {
        ".env",
        ".env.*",
        "!.env.example",
        ".venv/",
        ".browser-profile/",
        "*.har",
        "*.webm",
        "trace.zip",
        "evidence/**/raw/",
    }
    assert required <= ignored


def test_env_example_holds_placeholders_only() -> None:
    text = (ROOT / ".env.example").read_text()
    assert not re.search(r"sk-[A-Za-z0-9_-]{8,}", text)
    for line in text.splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            assert line.split("=", 1)[1].strip() == ""
