"""The safety policy: where the agent may go, which verbs it may use, and what counts as risky.

Everything here is explicit and configurable (a Policy loads from JSON), and it is conservative
by construction: an unrecognised URL is denied, a deny rule always beats an allow rule, and the
words on a control can raise its risk but nothing a caller declares can lower it.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import ConfigDict, Field, field_validator

from cua.artifact import RiskClass
from cua.common import StrictModel
from cua.surface.base import Action, ElementInfo

DEFAULT_VERBS = frozenset({"navigate", "click", "fill", "select", "press", "wait"})

# Prefix-matched at word starts, so `closeAcct(...)` and "Closed" match but "Disclosed" does not.
# The bias is deliberate: a false positive costs a confirmation, a false negative costs an account.
DEFAULT_IRREVERSIBLE_TERMS = (
    "close",
    "delete",
    "remove",
    "terminate",
    "purge",
    "wipe",
    "void",
    "reverse",
    "write off",
    "charge off",
    "wire transfer",
    "freeze",
)
DEFAULT_WRITE_TERMS = (
    "submit",
    "save",
    "apply",
    "post",
    "update",
    "approve",
    "authorize",
    "confirm",
    "send",
    "create",
    "add",
)

_DEFAULT_PORTS = {"http": 80, "https": 443}
_RANK = {RiskClass.READ: 0, RiskClass.REVERSIBLE_WRITE: 1, RiskClass.IRREVERSIBLE_WRITE: 2}


def max_risk(*risks: RiskClass | None) -> RiskClass:
    return max((r for r in risks if r is not None), key=_RANK.__getitem__, default=RiskClass.READ)


class _Frozen(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class UrlDecision:
    allowed: bool
    code: str  # ok | invalid_url | scheme_not_allowed | userinfo_not_allowed | host_not_allowed
    #            | path_not_allowed | denied_by_rule


class UrlRule(_Frozen):
    host: str
    port: int | None = Field(default=None, ge=1, le=65535)  # None matches any port
    path_prefix: str = "/"
    schemes: tuple[Literal["http", "https"], ...] = ("http", "https")

    @field_validator("host")
    @classmethod
    def _plain_host(cls, value: str) -> str:
        host = value.strip().lower().rstrip(".")
        if not host or any(ch in host for ch in "/@ \t"):
            raise ValueError("host must be a bare hostname or IP address")
        return host

    @field_validator("path_prefix")
    @classmethod
    def _rooted(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("path_prefix must start with '/'")
        return value

    def host_matches(self, scheme: str, host: str, port: int | None) -> bool:
        effective = port if port is not None else _DEFAULT_PORTS[scheme]
        return host == self.host and (self.port is None or self.port == effective)

    def path_matches(self, path: str) -> bool:
        prefix = self.path_prefix.rstrip("/")
        return prefix == "" or path == prefix or path.startswith(prefix + "/")


def _normalize_path(raw: str) -> str:
    """Resolve encodings, backslashes, dot segments and doubled slashes before any comparison."""
    path = unquote(raw).replace("\\", "/")
    normalized = posixpath.normpath(path) if path else "/"
    return "/" + normalized.lstrip("/")


def _corpus(info: ElementInfo) -> str:
    parts = [info.text, info.label_hint or ""]
    parts += [info.attrs.get(key, "") for key in ("title", "alt", "value", "href", "onclick")]
    return " ".join(parts).lower()


def _mentions(text: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(r"\b" + re.escape(term.lower()), text) for term in terms)


class Policy(_Frozen):
    allow: tuple[UrlRule, ...]
    deny: tuple[UrlRule, ...] = ()
    verbs: frozenset[str] = DEFAULT_VERBS
    # "confirm": an irreversible action waits for a human's yes; "block": it never runs.
    irreversible: Literal["block", "confirm"] = "confirm"
    irreversible_terms: tuple[str, ...] = DEFAULT_IRREVERSIBLE_TERMS
    write_terms: tuple[str, ...] = DEFAULT_WRITE_TERMS

    def url_decision(self, url: str) -> UrlDecision:
        try:
            parts = urlsplit(url.strip())
            port = parts.port
            host = parts.hostname
        except ValueError:
            return UrlDecision(False, "invalid_url")
        scheme = parts.scheme.lower()
        if scheme not in _DEFAULT_PORTS:
            return UrlDecision(False, "scheme_not_allowed")
        if host is None:
            return UrlDecision(False, "invalid_url")
        # `http://127.0.0.1@evil.example/` is a request to evil.example
        if parts.username is not None or parts.password is not None:
            return UrlDecision(False, "userinfo_not_allowed")
        host = host.lower().rstrip(".")
        path = _normalize_path(parts.path)

        for rule in self.deny:
            if rule.host_matches(scheme, host, port) and rule.path_matches(path):
                return UrlDecision(False, "denied_by_rule")
        refusal = "host_not_allowed"
        for rule in self.allow:
            if not rule.host_matches(scheme, host, port):
                continue
            if scheme not in rule.schemes:
                refusal = "scheme_not_allowed" if refusal == "host_not_allowed" else refusal
            elif rule.path_matches(path):
                return UrlDecision(True, "ok")
            else:
                refusal = "path_not_allowed"
        return UrlDecision(False, refusal)

    def classify(self, action: Action, info: ElementInfo | None) -> RiskClass:
        """How risky this action is, judged from what the page itself says about the control."""
        if action.kind == "press":
            return (
                RiskClass.REVERSIBLE_WRITE
                if (action.key or "").lower() in ("enter", "return")
                else RiskClass.READ
            )
        if action.kind != "click":
            return RiskClass.READ
        if info is None:
            return RiskClass.REVERSIBLE_WRITE  # a blind click cannot be vouched for
        corpus = _corpus(info)
        if _mentions(corpus, self.irreversible_terms):
            return RiskClass.IRREVERSIBLE_WRITE
        if _mentions(corpus, self.write_terms):
            return RiskClass.REVERSIBLE_WRITE
        return RiskClass.READ
