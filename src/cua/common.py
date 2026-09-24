"""Small pieces shared by the typed contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# Machine-readable identifiers (outcome codes, rule ids): lowercase, digits and underscores.
CODE_PATTERN = r"^[a-z][a-z0-9_]{1,63}$"


class StrictModel(BaseModel):
    """Unknown fields are errors, so a reviewer sees everything that is in a document."""

    model_config = ConfigDict(extra="forbid")
