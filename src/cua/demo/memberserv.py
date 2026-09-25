"""What a person decides about the MemberServ capabilities that a recording cannot know.

A recording shows what one run did. It cannot know that "NO RECORDS FOUND FOR CRITERIA" is a
legitimate answer rather than a failure, that a notice can be safely acknowledged, or that the
sign-in form appearing mid-flow means the session expired. Those judgements are declared here,
once, as the capability's error map, and travel with the artifact.
"""

from __future__ import annotations

from cua.agent import DiscoveryTask, OutputSpec, ParamSpec
from cua.artifact import ErrorRule, Target
from cua.artifact.synthesize import CapabilitySpec

TARGET = Target(vendor="Fictional Systems Ltd", product="MemberServ", version_range=">=3.1,<4")
SECRETS = ("MOCK_USER", "MOCK_PASS")


def _rule(**fields: object) -> ErrorRule:
    return ErrorRule.model_validate(fields)


NO_SUCH_MEMBER = _rule(
    id="no_such_member",
    detect={"text_present": ["NO RECORDS FOUND FOR CRITERIA"]},
    classification="business_outcome",
    outcome_code="member_not_found",
    message="No member matches the supplied number",
)

DEPOSIT_REJECTED = _rule(
    id="deposit_rejected",
    detect={"text_present": ["ERR 1042"]},
    classification="business_outcome",
    outcome_code="validation_rejected",
    message="The application rejected the opening deposit",
)

# Sign-in credentials are never stored, so an expired session cannot be repaired unattended:
# a person has to sign in again. (The login form appearing where it should not is the signal.)
SESSION_EXPIRED = _rule(
    id="session_expired",
    detect={"text_present": ["User ID"]},
    classification="hard_failure",
    code="session_expired",
    escalate=True,
    message="The session expired and signing in again needs credentials that are never stored",
)

EOD_NOTICE = _rule(
    id="eod_notice",
    detect={"text_present": ["SYSTEM NOTICE: END-OF-DAY BATCH"]},
    classification="recoverable",
    recovery={
        "kind": "dismiss",
        "max_attempts": 2,
        "locator": {
            "description": "Acknowledge button",
            "frame_path": [{"name": "main"}],
            "strategies": [
                {
                    "kind": "text",
                    "text": "Acknowledge",
                    "exact": True,
                    "rationale": "The notice's only control, labelled by its visible text",
                }
            ],
        },
    },
)

APP_ERROR = _rule(
    id="app_error",
    detect={"text_present": ["APPLICATION ERROR 0x8004"]},
    classification="hard_failure",
    code="app_error",
    message="The application reported an error. If this capability changes state, check whether "
    "the change was applied before trying again",
)


def lookup_task(base_url: str) -> DiscoveryTask:
    return DiscoveryTask(
        goal=(
            "Sign in, find the member with the given member number, and read their current "
            "share savings balance."
        ),
        start_url=f"{base_url}/msv/login.cgi",
        params={
            "member_id": ParamSpec("12345", "Member number (five digits)", pattern="^[0-9]{5}$")
        },
        outputs={
            "savings_balance": OutputSpec(
                "Current share savings balance, exactly as displayed, for example $2,480.15"
            )
        },
        secrets=SECRETS,
    )


def lookup_spec() -> CapabilitySpec:
    return CapabilitySpec(
        id="member_lookup",
        title="Look up a member and read their savings balance",
        description=(
            "Sign in, find a member by number and read the current share savings balance. "
            "Read-only: nothing is changed."
        ),
        target=TARGET,
        error_map=[NO_SUCH_MEMBER, SESSION_EXPIRED, EOD_NOTICE, APP_ERROR],
    )


def open_subaccount_task(base_url: str) -> DiscoveryTask:
    return DiscoveryTask(
        goal=(
            "Sign in, find the member with the given member number, open a new Share Savings "
            "sub-account for them with the given opening deposit, wait for the confirmation "
            "screen, and read the confirmation reference."
        ),
        start_url=f"{base_url}/msv/login.cgi",
        params={
            "member_id": ParamSpec("12345", "Member number (five digits)", pattern="^[0-9]{5}$"),
            "deposit": ParamSpec(
                "25.00", "Opening deposit in dollars", pattern=r"^[0-9]+\.[0-9]{2}$"
            ),
        },
        outputs={
            "confirmation_ref": OutputSpec(
                "Reference number on the confirmation screen, for example CNF-000001"
            )
        },
        secrets=SECRETS,
    )


def open_subaccount_spec() -> CapabilitySpec:
    return CapabilitySpec(
        id="open_subaccount",
        title="Open a share savings sub-account for a member",
        description=(
            "Sign in, find a member by number, open a new share savings sub-account with the "
            "given opening deposit and return the confirmation reference. Changes state: the "
            "submit step creates the account."
        ),
        target=TARGET,
        error_map=[NO_SUCH_MEMBER, DEPOSIT_REJECTED, SESSION_EXPIRED, EOD_NOTICE, APP_ERROR],
    )
