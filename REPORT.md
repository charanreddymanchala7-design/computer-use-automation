The model discovers, the artifact becomes a capability, deterministic replay is how an agent invokes it. Everything below serves that: keep the model out of the production path, and still be able to hand a stuck run to a person. The target is a synthetic legacy web app I built (`targets/mockbank`); evidence of real runs is in `evidence/README.md`.

## Architecture

One synchronous process, five layers, each behind a narrow seam:

- **Surface** (`surface/`). `observe()` returns per-frame accessibility snapshots, element facts and a screenshot; `act()` takes click, fill, select, press, navigate or wait, aimed at a harness-issued ref or at coordinates. Playwright is one implementation; the loop and replay never import it.
- **Gateway** (`gateway.py`). The only path to `act()` for discovery, replay, recovery and a person's hand-back. It applies the allowlist, verb list and risk gate and writes the audit record, so no caller can skip a check by taking another route.
- **Discovery** (`agent/`). Observe, decide, act on standard tool calling, with hard stops (steps, time, repeated action, unchanged page, errors in a row). Only successful actions are recorded. A locator is harvested and verified *before* the action, because a click can destroy the page it was aimed at.
- **Artifact and replay** (`artifact/`, `replay/`). Deterministic synthesis turns a finished recording into a capability; the engine runs it with no model.
- **Control** (`control/`). The lease, stuck detection, the handoff, operator channels.

Decisions and their cost:

1. **Custom observe/act tools, not the provider's computer-use tool.** The common case here has no clean DOM (framesets, nested tables, no test IDs), so I need per-frame perception, refs the harness owns, and locators verified against the live page. Cost: more code, and a structure-first bias with coordinates as the fallback.
2. **The model never writes the artifact.** It acts, the harness records, synthesis is code. The model can shape *what* is recorded but cannot put a selector or value into the artifact that the harness did not verify.
3. **Synchronous, one process.** Playwright's sync API allows one instance per thread, so loop, replay and browser share a thread and operator channels touch only the thread-safe lease. Cost: no parallel runs; a queue with workers is the scaling step and nothing blocks it.
4. **A thin model interface** (`LLM.step()`: messages and tools in, tool calls and token accounting out), with only metadata logged. It has three real implementations (Claude, Gemini, a local Ollama model) and a scripted `FakeLLM`, so CI needs no key and the take-home does not depend on one paid account. Which model produced a capability is recorded in its provenance.
5. **A hostile mock with fault injection and a server-side log of what really happened.** Most bugs I found (a password leaking into the accessibility snapshot, a forbidden navigation being recorded as a step) surfaced because the mock could tell me the truth.

## Artifact schema

Two halves for two readers. The **contract** is what an agent or reviewer reads: `id`, semver `capability_version`, `schema_version`, `status` (draft, active, deprecated), `target` (vendor, product, version range, tenant profile, surface), `inputs` and `outputs` as plain JSON Schema, the `error_map`, and per-step risk. The **recipe** is what replay runs: ordered `steps` (action, ranked locator bundle, value, `expect`, declared dialogs) and a `checkpoint`. `cua show` renders both; `cua capabilities schema` renders the contract as a tool definition.

- **JSON Schema for inputs and outputs**, so they map onto function-calling and MCP tools and replay validates them with a standard validator. Sensitive inputs are marked `x-sensitive` and cannot carry a default, example or enum.
- **A locator is a ranked bundle and each strategy states why it should survive a UI change.** Harvest tries role and name, label, ancestor anchor ("the cell in the row labelled X"), attribute fingerprint, then visible text, with a positional CSS selector only if none verifies. Coordinates are allowed only last and are tied to a viewport. Only strategies checked to resolve to exactly the recorded element are kept.
- **Parameterised, portable, free of example data.** Caller values become `{member_id}` in locators, expectations and dialog text; URLs are stored as paths (the host belongs to the tenant); a description like `tr containing {member_id}` replaces the row text that was on screen. I found that leak by reading my first evidence run; it is now a test.
- **Write steps must declare success.** A click is verified, not assumed.
- **Sensitive data cannot enter it.** A sensitive step cannot carry a literal, a sensitive input cannot appear in a locator, URL or expectation, and saving refuses a capability that would leak a registered secret.
- **Provenance, not the transcript**: run id, model, time, step count. A digest over canonical JSON tells an edited capability from the approved one. `schemas/*.json` are generated and a test fails on drift.

## Determinism & error handling

**Deterministic replay.** No model. Inputs and secrets are validated before the application is touched. Elements are *waited for by polling* within a per-step budget, never slept for. Locators are tried in ranked order and a fallback that matches is reported as `degraded`, the earliest sign of UI drift. Each step's `expect` and the final checkpoint are verified. The same inputs give the same result document (tested).

**Result contract.** `success` (exit 0, outputs), `business_outcome` (10), `escalated` (20), `hard_failure` (30), each with a machine-readable `outcome_code`. A failure names the step, what was expected and what was observed, with a screenshot and a redacted page snapshot. Recoverable conditions are not a status: the run continues and `recoveries[]` records them.

**Taxonomy, declared in the capability's error map:**

- *Business outcome*: `NO RECORDS FOUND` is `member_not_found`; a below-minimum deposit is `validation_rejected` and is never auto-retried.
- *Recoverable*, bounded per rule and step: an end-of-day notice is dismissed (at most twice); a slow page is waited for with backoff and reported as `slow_response`. A spent budget is a hard failure.
- *Hard failure*: an application error. A state-changing submit is never blindly retried, because its outcome is uncertain (the fault can commit the write and then show the error).
- *Escalate*: the sign-in form appearing mid-flow. Credentials are never stored, so only a person can repair it.

The fault matrix (`docs/fault-matrix.md`, 15 cases in real Chromium) is checked against the *server's* log, not the client's report: a write happens exactly once, nothing outside the allowlist is requested, the admin page and the irreversible link are never touched.

## Heterogeneity & multi-tenant

Only the web surface is built. What the seams allow:

**Surface abstraction.** The artifact says what to do and how to find it; the surface says how to perceive and act. Replay talks to a `LocatingSurface` (`observe`, `act`, `resolve`, `harvest`, `read_text`), and `Target.surface` already separates `web` from `desktop`. A legacy web app needs nothing new: `frame_path` models framesets and nesting. A desktop surface implements the same protocol over the OS accessibility tree (UI Automation, AX) and adds locator kinds to the existing tagged union (`automation_id`, an accessibility path, an image template for canvas-drawn controls). Coordinates stay the universal last resort. Steps, checkpoint, error map, gateway and lease do not change.

**Multi-tenant reuse.** A *base* capability is keyed by `(vendor, product, version_range)`. A tenant's variation is an *overlay*, not a fork: a patch keyed by step id (a locator or label override, a different base URL, extra checkpoint text or error rules) applied at load, with the digest covering both. Locators prefer role, label and structure over branding and ids, so most tenants need no overlay. **Drift** comes from data replay already emits: `degraded` locators by step, tenant and version; a version outside `version_range` refuses to run; a scheduled replay of a read-only capability is a canary. Repair is bounded: re-discover the one broken step, store it as an overlay, promote it to the base when several tenants need it. Built today: the `target` block, portable paths and the degraded signal. The overlay resolver is not built.

## Escalation & handoff

**Detecting stuck.** `stuck_reason()` maps outcomes to reasons a person can act on: expired session, unrecognised state (an element or expectation that never appears), unexpected dialog, a condition that would not clear, a risky step, a discovery dead end. Answers, known crashes and bad inputs do not escalate; a person could not help. A run asks at most twice, then fails.

**The request** carries capability and goal, step, why it stopped, expected against seen, a path-only URL (query strings hold tokens) and a screenshot, all redacted. It reaches a person through a terminal prompt or a loopback web page.

**Who is in control.** A `ControlLease` moves through `IDLE, RUNNING, WAITING_FOR_HUMAN, HUMAN_IN_CONTROL, HANDED_BACK` (or `ABORTED`) with an *epoch* that advances on every change. A browser lets any client drive a page, so single-writer is enforced at the gateway: agent actions carry their epoch and a stale one is refused before it reaches the page. The agent stops at a step boundary and holds nothing; the browser is neither closed nor replaced, so the person works in **the same window and session**. While waiting the agent keeps the browser's event loop turning (`pause`, not a blocking wait), or page events and dialogs would stall.

**Handing back is not resuming.** The page is observed and logged first; the agent resumes under a fresh epoch and the engine retries the step it stopped at. If that step's action had already run, only its *check* is retried, never the action, so a Submit is not pressed twice. A person can also abort; a timeout ends the run as `escalated`.

**What the person did is recorded.** The surface reports trusted clicks, Enter, selections and navigations, and typing by *length only* (a password field not even that; a data cell is never named, because its text is someone's record). Actions are logged as they happen and summarised on the result. A dialog raised while a person holds control is cancelled, not accepted for them.

Tested with a second process attached to the run's own browser over its debug port, acting as the person through the operator API; the test then checks the server's request log.

## Safety

**Allowlist.** Explicit, JSON-configurable, deny beats allow. URL parsing is deliberate (userinfo tricks, `..`, `%2e`, backslashes, ports). It is enforced on the `navigate` verb and again on the network with a request guard, because a page can navigate itself; a blocked request gets a 403 and is logged, and a click that provoked one is not recorded. Verbs are allowlisted too.

**Risk.** Actions are read, reversible write or irreversible write, judged from what the page says about the control (close, delete, void, wire transfer...). A declared risk can only raise it; a blind coordinate click counts as a write. An irreversible action needs a single-use confirmation bound to that exact control (`block` is a policy setting). The bias is deliberate: a false positive costs a confirmation, a false negative costs an account.

**Data.** Artifacts hold secret *names*; the harness types the value and the model never sees it. Observations mask password values and registered secrets. Redaction is layered (registered secrets, matched case-insensitively because legacy screens upper-case what you type; value shapes such as SSNs, long numbers, keys; sensitive keys) and applied to every log line, result and evidence file. Model logging is metadata only, identifiers are masked in the CLI, and the audit trail names a control by its label but never copies the text of a row or cell (found by reading my own live run's log), and `scripts/check_evidence_clean.sh` fails on credentials, personal paths, emails and raw browser state. The operator page binds to loopback, checks the Host header, needs a per-server token to change anything, and sets content as text under a nonce-based policy.

**Limits.** Screenshots and failure snapshots show what was on screen, and the operator page shows the same; they are not pixel-redacted (the demo data is synthetic), and a real deployment would treat them as sensitive. During discovery the model *does* see page text, which goes to the provider: fine for a sandbox, a data-processing decision for real member data. Risk classification is keyword-based and can miss unusual vocabulary. The operator page has no user accounts. There is no approval workflow.

## Cuts

Left out on purpose, seams kept:

- The **desktop surface** and **multi-tenant overlays**: designed above, not built.
- **Stretch goals** other than the catalog: assisted LLM fallback, approval states, multi-run stability, code generation.
- **Discovery-time handoff.** The stuck reason exists, but a dead-end discovery run ends rather than asking a person.
- **Approving a held irreversible action from the operator page.** Single-use confirmations exist and are tested; the operator UI does not use them.
- **A person answering a native dialog by hand.** Dialogs are cancelled during a handoff; whether a headed window lets a person accept one is unverified.
- **Live co-browsing.** The person uses the real headed window; the operator page only moves the lease.
- **Unevaluated schema fields**: the `dialog_message` detector and `aria_contains` checkpoint.
- **More fault modes** (permission denied, unknown interstitial, unexpected dialog): the same three classes as the six built.
- **A task file format.** Discovery tasks are Python objects, not free text from a file.

Next, in order: the overlay resolver and a canary schedule (multi-tenant drift is my biggest unproven claim); approval from the operator page; discovery-time handoff, so a person can finish what the model could not; a desktop surface behind the same seam; assisted single-step recovery with a policy check.
