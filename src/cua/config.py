"""Local configuration: a small ``.env`` reader and the rule for when colour is allowed.

Keys are read from the environment or a gitignored ``.env`` file, never from a flag (a flag would
land in shell history and process listings) and never echoed back.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from pathlib import Path


def load_dotenv(path: Path, env: MutableMapping[str, str] | None = None) -> list[str]:
    """Add the file's ``KEY=value`` pairs to ``env`` without overriding anything already set.

    Returns the names that were added (never the values)."""
    target = os.environ if env is None else env
    added: list[str] = []
    if not path.is_file():
        return added
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value and key not in target:
            target[key] = value
            added.append(key)
    return added


def use_color(*, isatty: bool, env: Mapping[str, str]) -> bool:
    """Colour only on a terminal, and never when NO_COLOR is set (https://no-color.org)."""
    return isatty and "NO_COLOR" not in env and env.get("TERM") != "dumb"
