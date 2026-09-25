"""Structured evidence: a JSON-lines event log and a guarded artifact writer.

Both writers redact on the way out. The log redacts and keeps going (losing a log line would
hide what happened); the artifact writer refuses to write at all if redaction would change
anything, because a capability that needed redacting means sensitive data got into a recording.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cua.redact import Redactor


class ArtifactLeak(ValueError):
    """An artifact contained sensitive content; nothing was written."""


def _now() -> datetime:
    return datetime.now(UTC)


class EventLog:
    """One JSON object per line: run id, step, action, target, outcome and the reason why."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        redactor: Redactor,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.path = path
        self._run_id = run_id
        self._redactor = redactor
        self._clock = clock

    @property
    def redactor(self) -> Redactor:
        """So callers that put text in a result redact it exactly as the log does."""
        return self._redactor

    def emit(
        self,
        event: str,
        *,
        step: str | None = None,
        action: str | None = None,
        target: str | None = None,
        outcome: str | None = None,
        reason: str | None = None,
        observed: str | None = None,
        typed: str | None = None,
        sensitive: bool = False,
        **fields: Any,
    ) -> None:
        record: dict[str, Any] = {
            "ts": self._clock().isoformat(),
            "run_id": self._run_id,
            "event": event,
        }
        optional = {
            "step": step,
            "action": action,
            "target": target,
            "outcome": outcome,
            "reason": reason,
            "observed": observed,
            # what a person or model typed: only its length is kept when the field is sensitive
            "typed": None if typed is None else _typed(typed, sensitive, self._redactor),
        }
        record.update({key: value for key, value in optional.items() if value is not None})
        record.update(fields)
        line = json.dumps(self._redactor.redact(record), ensure_ascii=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _typed(value: str, sensitive: bool, redactor: Redactor) -> str:
    return redactor.placeholder(value) if sensitive else value


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        events.append(json.loads(line))
    return events


def _as_json(obj: BaseModel | dict[str, Any]) -> Any:
    raw = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    return json.loads(json.dumps(raw))  # normalise tuples etc. so comparisons are exact


def write_artifact(path: Path, artifact: BaseModel | dict[str, Any], redactor: Redactor) -> None:
    """Write a capability only if it is already clean; otherwise raise and write nothing.

    Checked by content, not by key: a capability's schema has properties named ``password``.
    """
    data = _as_json(artifact)
    where = _first_difference(data, redactor.redact(data, by_key=False), "$")
    if where is not None:
        # the location only: echoing the value would leak it into the error message
        raise ArtifactLeak(f"refusing to write {path.name}: sensitive content at {where}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_redacted_json(
    path: Path, document: BaseModel | dict[str, Any], redactor: Redactor
) -> None:
    """Write an evidence file (result, observation) after redacting it by key and by content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(redactor.redact(_as_json(document)), indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def _first_difference(before: Any, after: Any, path: str) -> str | None:
    if isinstance(before, dict) and isinstance(after, dict):
        for key, value in before.items():
            found = _first_difference(value, after.get(key), f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(before, list) and isinstance(after, list):
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            found = _first_difference(old, new, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    return None if before == after else path
