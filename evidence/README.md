# Evidence

Everything here was produced by the real `cua` commands against the synthetic MemberServ mock. Nothing is hand-edited. Each replay folder keeps `command.txt` (what was typed) and `output.txt` (what was printed) beside the run's own redacted log and result, so you can read what happened without running anything.

| Folder | What it shows | Exit |
|---|---|---|
| [`01-discovery`](01-discovery) | **A genuine model-driven run.** `gemini-3.5-flash-lite` explored the app once and produced the capability in [`capabilities/member_lookup.json`](../capabilities/member_lookup.json) | 0 |
| [`02-replay-success`](02-replay-success) | The capability replayed for a *different* member (12346) with **no model**: returns `$100.00` | 0 |
| [`03-business-outcome`](03-business-outcome) | Member 99999 does not exist. That is an answer, not a crash: `business_outcome` / `member_not_found` | 10 |
| [`04-hard-failure`](04-hard-failure) | **The error case.** The application is made to crash when searching (injected fault). Hard failure at step `s7` with expected, observed, a screenshot and a page snapshot | 30 |
| [`05-handoff`](05-handoff) | **Human takeover.** The session expires mid-run; the run stops, a person takes the *same live browser*, signs in and searches again, hands back, and the run finishes | 0 |
| [`06-agent-call`](06-agent-call) | The capability called the way an AI agent would: JSON arguments in, an MCP-shaped reply out | 0 |

## 01 Discovery

| | |
|---|---|
| Goal | Sign in, find the member with the given member number, and read their current share savings balance |
| Model | `gemini-3.5-flash-lite` through the Gemini API (free tier) |
| Cost in the loop | 8 recorded steps from 9 model calls, 32,667 tokens, 25 s. No charge: free tier |
| Result | `summary.json`: `finished`, "the model reported that the goal was achieved" |

`run.jsonl` is the redacted structured log (every action, with the reason). `00-start.png` and `01-act.png` to `06-act.png` are the pages the model was shown, in order. `review.md` is the capability as a reviewer reads it, and `capability.json` is the artifact itself.

How it got here, honestly, since the first attempts did not go well:

- **A local `llama3.1:8b` failed twice.** It typed the member number into the password field and never used the `secret` argument. Too weak for this loop, so I did not tune around it.
- **Gemini returned 404, then 503 and 429 for a while.** The 404 was a model that is listed for the key but retired for new keys; the 503/429 were overload and the free-tier limit. The adapter now retries dropped connections and says what kind of failure it was.
- **One full Gemini run (gemini-3.8-flash) succeeded and exposed a defect in my harness.** The model quoted an entire results row as the anchor for the balance, so the saved locator fell back to a positional CSS selector and kept one member's account number and balance in a description. I fixed it (`d3ee622`: the anchor is cut back to its label) and re-ran discovery, and that run is the one committed here.
- **Reading its log then showed a second leak:** the audit trail copied the text of the results row I had clicked. Element text is now logged only for controls (`audit_label`), and discovery was run again. The evidence here is from that last run.

## 05 Handoff

The person is `scripts/simulate_operator.py`: a **second client attached to the run's own browser** over its loopback debug port, using the operator web API to take control and hand back. It types the credentials it is given, as a person would, because the system never stores them.

Read `output.txt` for the story: the run is stuck at `s7` (`session_expired`), a person takes control, **10 actions are recorded**, control is handed back, the page is re-checked, and `s7` runs again. The list of actions is in the RESULT block and in `run.jsonl` as `human_action` events. Typing is recorded by length only (`typed 8 characters into text field 'name=u'`); a password field is recorded as a password field. `intervention-1.png` is what the run saw when it stopped; `handback-ir_05-handoff_1.png` is what the person handed back.

## What is and is not redacted

- Logs, results, the capability and page snapshots go through the redactor (registered secrets, matched case-insensitively; value shapes such as SSNs and long numbers; sensitive keys). `scripts/check_evidence_clean.sh` fails on credentials, personal paths, email addresses and raw browser state; it passes on this folder.
- **Screenshots are not pixel-redacted.** They show what was on screen. The data is synthetic (every page says so), and the sign-in is the mock's public demo login (it is in `.env.example`), so its username is visible in a screenshot of the login form. The password is masked by the browser.
- No traces, HAR files, videos or browser profiles are kept.

## Regenerate

```bash
make mock                                              # terminal 1
uv run cua run --task member_lookup --provider gemini --evidence evidence/01-discovery   # a live model
make evidence                                          # 02 to 06, then the cleanliness check
```
