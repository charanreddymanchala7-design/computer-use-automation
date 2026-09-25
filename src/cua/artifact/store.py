"""Saving and loading capabilities: one JSON file per capability, named by its id."""

from __future__ import annotations

from pathlib import Path

from cua.artifact.schema import Capability
from cua.evlog import write_artifact
from cua.redact import Redactor


def save_capability(
    capability: Capability, directory: Path, redactor: Redactor | None = None
) -> Path:
    """Write the capability, refusing (and writing nothing) if it would leak a secret."""
    path = directory / f"{capability.id}.json"
    write_artifact(path, capability, redactor or Redactor())
    return path


def load_capability(path: Path) -> Capability:
    return Capability.model_validate_json(path.read_text(encoding="utf-8"))


def list_capabilities(directory: Path) -> list[Capability]:
    if not directory.is_dir():
        return []
    return sorted(
        (load_capability(p) for p in directory.glob("*.json") if p.name != "catalog.json"),
        key=lambda c: c.id,
    )
