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
| `session_timeout` | `newsub_form` | The login form is rendered inside the `main` frame, HTTP 200, session invalidated | Login-form signature in frame `main`; the header frame loses the user | Escalate: credentials are never stored, so a human must sign in | `escalated` / `session_expired` | 20 |
| `slow_load` | `search` | The server delays its response (`params.delay_ms`, default 3000) | Step timeout budget; checkpoint not yet present | Recoverable: wait within the budget. Beyond the budget it becomes a hard failure | `success` + `recoveries[wait_retry]`, or `hard_failure` / `recovery_exhausted` | 0 / 30 |
| `interstitial_known` | `member` | A `SYSTEM NOTICE: END-OF-DAY BATCH AT 17:00` page with an `Acknowledge` cell replaces the page | Known interstitial signature declared in the artifact's `error_map` | Recoverable: dismiss, then re-verify | `success` + `recoveries[dismiss]` | 0 |
| `app_error` | `newsub_submit` | `APPLICATION ERROR 0x8004 - CONTACT YOUR SYSTEM ADMINISTRATOR` (HTTP 200, or `params.status`). With `params.commit: true` the sub-account **was created** before the error is shown | Error signature; checkpoint missing | Hard failure. A submit is never blindly retried, because with `commit` the outcome is uncertain | `hard_failure` / `app_error` | 30 |

`GET /_admin/log` returns `requests` (method, path, status, step, `fault_applied`) and
`effects` (`subaccount_created`, `account_closed`). It is the ground truth for tests such as
"the replay never double-submitted": count the `subaccount_created` effects.

## Not implemented (and why)

The full taxonomy in the research also lists `permission_denied`, `interstitial_unknown`,
`dialog_benign` and `dialog_unknown`. They exercise the same three classes already covered
above and were cut to keep the thin slice thin. `interstitial_unknown` and `dialog_unknown`
would be the natural additional escalation triggers.

The query-string channel (`?__fault=`) was deliberately not built: it contradicts the rule that
a fault must never appear in a URL the agent sees.
