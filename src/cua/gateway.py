"""The action gateway: the single choke point every action passes through.

Discovery, replay, assisted recovery and a human's hand-back all go through ``ActionGateway.act``,
so the allowlist, the verb list, the risk gate and the audit trail cannot be skipped by taking a
different path. The same policy also guards the network (``Surface.set_request_guard``), because a
page can navigate on its own without any action being issued.

Every decision, allowed or not, is written through the redacting event log.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from cua.artifact import RiskClass
from cua.evlog import EventLog
from cua.policy import Policy, max_risk
from cua.result import CuaError
from cua.surface import Action, ActionResult, ElementInfo, RequestInfo, Surface


class GatewayError(Exception):
    """Misuse of the gateway itself (for example confirming something never asked for)."""


class Reason(StrEnum):
    OK = "ok"
    CONFIRMED = "confirmed"
    VERB_NOT_ALLOWED = "verb_not_allowed"
    URL_NOT_ALLOWED = "url_not_allowed"
    IRREVERSIBLE_BLOCKED = "irreversible_blocked"
    CONFIRMATION_REQUIRED = "confirmation_required"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: Reason
    risk: RiskClass
    verb: str
    target: str
    detail: str | None = None
    confirmation_id: str | None = None


class PolicyViolation(CuaError):
    def __init__(self, decision: Decision) -> None:
        super().__init__(decision.reason.value)
        self.decision = decision


@dataclass
class GatedResult:
    decision: Decision
    result: ActionResult | None = None

    @property
    def executed(self) -> bool:
        return self.result is not None

    def require(self) -> ActionResult:
        """The action's result, or the policy violation that stopped it."""
        if self.result is None:
            raise PolicyViolation(self.decision)
        return self.result


def describe_target(action: Action, info: ElementInfo | None) -> str:
    if info is not None:
        return f"{info.ref} <{info.tag}> {(info.text or info.label_hint or '')[:60]!r}"
    if action.kind == "navigate":
        return action.url or ""
    if action.kind == "press":
        return action.key or ""
    if action.kind == "wait":
        return f"{action.ms} ms"
    if action.x is not None and action.y is not None:
        return f"({action.x:.0f}, {action.y:.0f})"
    return action.kind


def _confirmation_key(action: Action, info: ElementInfo | None) -> str:
    """What a confirmation is bound to: the same verb on the same control, nothing broader."""
    if info is None:
        return f"{action.kind}|{action.url}|{action.x}|{action.y}|{action.key}"
    what = info.attrs.get("href") or info.attrs.get("onclick") or ""
    return f"{action.kind}|{info.frame}|{info.tag}|{info.text}|{what}"


class ActionGateway:
    def __init__(
        self,
        surface: Surface,
        policy: Policy,
        log: EventLog,
        *,
        guard_requests: bool = True,
    ) -> None:
        self._surface = surface
        self._policy = policy
        self._log = log
        self._pending: dict[str, str] = {}  # confirmation id -> action key, awaiting a human
        self._granted: dict[str, str] = {}  # confirmation id -> action key, confirmed, unused
        # Navigations the network guard refused. An action that provoked one did not do what it
        # looked like, so callers use this to avoid recording it.
        self.blocked_navigations: list[str] = []
        if guard_requests:
            surface.set_request_guard(self._request_allowed)

    # --- confirmations --------------------------------------------------------------------------

    def confirm(self, confirmation_id: str, *, approver: str) -> None:
        """A person says yes to one held action. Single use, bound to that exact control."""
        key = self._pending.pop(confirmation_id, None)
        if key is None:
            raise GatewayError(f"no pending confirmation {confirmation_id!r}")
        self._granted[confirmation_id] = key
        self._log.emit(
            "confirmation_granted",
            outcome="granted",
            confirmation_id=confirmation_id,
            approver=approver,
        )

    def _take_grant(self, key: str, confirmation_id: str | None) -> str | None:
        if confirmation_id is not None:
            if self._granted.get(confirmation_id) == key:
                del self._granted[confirmation_id]
                return confirmation_id
            return None
        for granted_id, granted_key in list(self._granted.items()):
            if granted_key == key:
                del self._granted[granted_id]
                return granted_id
        return None

    def _pending_id(self, key: str) -> str:
        for pending_id, pending_key in self._pending.items():
            if pending_key == key:
                return pending_id  # asking again about the same control reuses the same id
        pending_id = f"cf_{secrets.token_hex(4)}"
        self._pending[pending_id] = key
        return pending_id

    # --- the choke point ------------------------------------------------------------------------

    def act(
        self,
        action: Action,
        *,
        declared_risk: RiskClass | None = None,
        confirmation_id: str | None = None,
        step: str | None = None,
    ) -> GatedResult:
        info = self._surface.element_info(action.ref) if action.ref is not None else None
        target = describe_target(action, info)
        # what the page says about a control wins over what a caller declares: risk only goes up
        risk = max_risk(self._policy.classify(action, info), declared_risk)
        verb = action.kind

        def blocked(
            reason: Reason, detail: str | None = None, cid: str | None = None
        ) -> GatedResult:
            decision = Decision(False, reason, risk, verb, target, detail, cid)
            extra: dict[str, Any] = {}
            if detail:
                extra["detail"] = detail
            if cid:
                extra["confirmation_id"] = cid
            self._log.emit(
                "action_blocked",
                step=step,
                action=verb,
                target=target,
                outcome="blocked",
                reason=reason.value,
                risk=risk.value,
                **extra,
            )
            return GatedResult(decision)

        if verb not in self._policy.verbs:
            return blocked(Reason.VERB_NOT_ALLOWED)
        if verb == "navigate":
            url = self._policy.url_decision(action.url or "")
            if not url.allowed:
                return blocked(Reason.URL_NOT_ALLOWED, url.code)

        reason = Reason.OK
        released_by: str | None = None
        if risk is RiskClass.IRREVERSIBLE_WRITE:
            if self._policy.irreversible == "block":
                return blocked(Reason.IRREVERSIBLE_BLOCKED)
            key = _confirmation_key(action, info)
            released_by = self._take_grant(key, confirmation_id)
            if released_by is None:
                return blocked(Reason.CONFIRMATION_REQUIRED, cid=self._pending_id(key))
            reason = Reason.CONFIRMED

        result = self._surface.act(action)
        done: dict[str, Any] = {}
        if action.secret:
            done["secret_name"] = action.secret  # the name only: the value is never seen here
        if released_by:
            done["confirmation_id"] = released_by
        if result.error:
            done["error"] = result.error
        self._log.emit(
            "action",
            step=step,
            action=verb,
            target=target,
            outcome="ok" if result.ok else "failed",
            reason=reason.value,
            risk=risk.value,
            typed=action.text if action.secret is None else None,
            **done,
        )
        return GatedResult(Decision(True, reason, risk, verb, target, None, released_by), result)

    # --- the network guard ----------------------------------------------------------------------

    def _request_allowed(self, request: RequestInfo) -> bool:
        decision = self._policy.url_decision(request.url)
        if not decision.allowed:
            if request.is_navigation:
                self.blocked_navigations.append(request.url)
            self._log.emit(
                "url_blocked",
                outcome="blocked",
                reason=Reason.URL_NOT_ALLOWED.value,
                detail=decision.code,
                url=request.url,
                method=request.method,
                resource_type=request.resource_type,
                navigation=request.is_navigation,
            )
        return decision.allowed
