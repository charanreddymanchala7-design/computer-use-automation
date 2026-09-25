"""Redaction of regulated and secret data.

Applied inside the log writer and the artifact writer, so it holds no matter which code path
produced the data. Three layers, cheapest to strongest:

* **shape**: credit-card / account-number-like digit runs, SSN-shaped values, API keys, tokens,
  JWTs and ``password=...`` style pairs are masked wherever they appear in text;
* **key**: values under keys such as ``password`` or ``token`` are replaced whole, and id keys
  such as ``member_id`` keep only their last two characters;
* **registered secrets**: exact values that must never appear (the API key loaded from the
  environment, a password a human typed) are removed from any text, even when they have no
  recognisable shape.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_MIN_SECRET_LENGTH = 4  # shorter values would shred ordinary text

# Order matters: specific token shapes first, generic digit runs last.
_SHAPES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "<redacted:jwt>"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer <redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "<redacted:api-key>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "<redacted:token>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<redacted:aws-key>"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<redacted:ssn>"),
    (re.compile(r"\b\d{8,19}\b"), "<redacted:number>"),
)

_KEY_VALUE = re.compile(
    r"""(?ix)
    \b(password|passwd|pwd|secret|token|api[_-]?key|authorization)
    (["']?\s*[=:]\s*)
    ("[^"]*"|'[^']*'|\S+)
    """
)

_DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "pwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "set-cookie",
        "ssn",
        "credential",
        "credentials",
        "access_token",
        "refresh_token",
        "private_key",
    }
)
_DEFAULT_ID_KEYS = frozenset({"member_id", "account_id", "member_number", "account_number"})


class Redactor:
    def __init__(
        self,
        *,
        secrets: Iterable[str] = (),
        sensitive_keys: Iterable[str] = (),
        masked_id_keys: Iterable[str] = (),
    ) -> None:
        # longest first so a secret that contains another is removed whole
        # case-insensitive: legacy screens often upper-case what was typed
        self._secrets = tuple(
            re.compile(re.escape(s), re.IGNORECASE)
            for s in sorted(
                {s for s in secrets if len(s) >= _MIN_SECRET_LENGTH}, key=len, reverse=True
            )
        )
        self._sensitive_keys = _DEFAULT_SENSITIVE_KEYS | {k.lower() for k in sensitive_keys}
        self._id_keys = _DEFAULT_ID_KEYS | {k.lower() for k in masked_id_keys}

    def redact_text(self, text: str) -> str:
        for secret in self._secrets:
            text = secret.sub("<redacted:secret>", text)
        for pattern, replacement in _SHAPES:
            text = pattern.sub(replacement, text)
        return _KEY_VALUE.sub(r"\1\2<redacted>", text)

    def redact(self, obj: Any, *, by_key: bool = True) -> Any:
        """A redacted copy of any JSON-like structure; the input is never modified.

        ``by_key`` masks values stored under keys such as ``password`` or ``member_id``. That is
        right for free-form log and evidence records. It is wrong for structural documents such
        as a capability, whose JSON Schema legitimately has properties *named* ``password``;
        those are checked by content (shapes and registered secrets) only.
        """
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, dict):
            return {key: self._redact_entry(str(key), value, by_key) for key, value in obj.items()}
        if isinstance(obj, list | tuple):
            return [self.redact(item, by_key=by_key) for item in obj]
        return obj

    def _redact_entry(self, key: str, value: Any, by_key: bool) -> Any:
        lowered = key.lower()
        if by_key and lowered in self._sensitive_keys:
            return "<redacted>"
        if by_key and lowered in self._id_keys and value is not None:
            return self.mask_id(str(value))
        return self.redact(value, by_key=by_key)

    @staticmethod
    def mask_id(value: str) -> str:
        """Keep the last two characters so a person can still match records by eye."""
        return f"***{value[-2:]}" if len(value) >= 5 else "***"

    @staticmethod
    def placeholder(value: str) -> str:
        """Stand-in for a typed value that must not be stored: only its length survives."""
        return f"<redacted len={len(value)}>"
