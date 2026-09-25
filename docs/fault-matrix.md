# Fault matrix

MemberServ can inject the runtime conditions a legacy back-office app produces. The point is
not layout drift: the UI is stable. The interesting failures are the ones that happen at
runtime and that a replay must *detect* and *classify* instead of blindly proceeding.

Faults are armed out-of-band, never in a URL the agent can see (a fault name in a URL would be
recorded into a "reusable" capability):

```
POST   /_admin/faults   {"mode": "app_error", "step": "newsub_submit", "times": 1, "params": {}}
GET    /_admin/faults   armed faults and remaining counts
DELETE /_admin/faults   disarm everything
GET    /_admin/log      ground truth: every request (with the fault applied) and every side effect
POST   /_admin/reset    reset data, sessions, confirmation counter, log and armed faults
```

Or at boot: `MOCK_FAULTS="session_timeout@newsub_form:1,slow_load@search:always"`
(`mode[@step][:times]`, `times` is a positive integer or `always`).

`step` is `login | search | results | member | newsub_form | newsub_submit | any`. `any` matches
every app page except images and the header/navigation frames, so a fault is not consumed by
chrome that loads alongside the page.

## The six implemented faults

| Mode | Default step | What the app does | How replay detects it | Classification | Result `status` / `outcome_code` | Exit |
|---|---|---|---|---|---|---|
| `member_not_found` | `results` | Search returns zero rows: red `NO RECORDS FOUND FOR CRITERIA`, HTTP 200 | No result rows plus that text in the `main` frame | Expected business outcome | `business_outcome` / `member_not_found` | 10 |
| `validation_error` | `newsub_submit` | Form re-rendered with `ERR 1042: OPENING DEPOSIT BELOW MINIMUM ($5.00)`, fields cleared, nothing created | Error banner text `ERR \d{4}:` with the form still present | Expected business outcome (caller input rejected, never auto-retried) | `business_outcome` / `validation_rejected` | 10 |
| `session_timeout` | `newsub_form` | The login form is rendered inside the `main` frame, HTTP 200, session invalidated | The `session_expired` rule: the sign-in form (`User ID`) showing where it should not | Escalate: credentials are never stored, so a human must sign in | `escalated` / `session_expired` | 20 |
| `slow_load` | `search` | The server delays its response (`params.delay_ms`, default 3000) | The wait itself: a step (an action while it settles, or an element or expectation being waited for) took at least a second | Recoverable: wait within the budget, and report it | `success` + `recoveries[slow_response]`; beyond the step budget, `hard_failure` / `locator_not_found` or `expectation_failed` at that step | 0 / 30 |
| `interstitial_known` | `member` | A `SYSTEM NOTICE: END-OF-DAY BATCH AT 17:00` page with an `Acknowledge` cell replaces the page | The `eod_notice` rule in the artifact's `error_map`, consulted when what the step wanted is missing | Recoverable: dismiss (at most 2 tries per step), then look again | `success` + `recoveries[eod_notice]`; if it will not clear, `hard_failure` / `recovery_exhausted` | 0 / 30 |
| `app_error` | `newsub_submit` | `APPLICATION ERROR 0x8004 - CONTACT YOUR SYSTEM ADMINISTRATOR` (HTTP 200, or `params.status`). With `params.commit: true` the sub-account **was created** before the error is shown | Error signature; checkpoint missing | Hard failure. A submit is never blindly retried, because with `commit` the outcome is uncertain | `hard_failure` / `app_error` | 30 |

`GET /_admin/log` returns `requests` (method, path, status, step, `fault_applied`) and
`effects` (`subaccount_created`, `account_closed`). It is the ground truth for tests such as
"the replay never double-submitted": count the `subaccount_created` effects.

## How the matrix is proved

`tests/test_fault_matrix.py` discovers two capabilities once (a read-only member lookup and a
state-changing sub-account opening), then replays every case in real Chromium through the real
engine, and checks two things for each: what the caller is told (status, outcome code, exit
code, recoveries, failed step, evidence files) and what the **server** says happened, through
`/_admin/log`. For the write capability that ground truth is decisive:

| Case | The server must show |
|---|---|
| `app_error` with `commit: true` | exactly one `subaccount_created` effect, and exactly one POST of the submit, even though the replay failed (the outcome is uncertain, so it is never blindly retried) |
| `app_error` without `commit` | no effect, one POST |
| `validation_error` (natural or injected) | no effect, one POST, and the caller gets `business_outcome` / `validation_rejected` |
| every case | no request to the admin page or the irreversible Close link |

Session expiry cannot be repaired unattended because credentials are never stored, so it is a
hard failure that asks for a human (`escalated` / `session_expired`).

## Not implemented (and why)

The full taxonomy in the research also lists `permission_denied`, `interstitial_unknown`,
`dialog_benign` and `dialog_unknown`. They exercise the same three classes already covered
above and were cut to keep the thin slice thin. `interstitial_unknown` and `dialog_unknown`
would be the natural additional escalation triggers.

The query-string channel (`?__fault=`) was deliberately not built: it contradicts the rule that
a fault must never appear in a URL the agent sees.
