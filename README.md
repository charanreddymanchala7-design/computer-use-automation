# Computer-use automation

A model works out how to do a task in a legacy back-office web app, once. What it learned is saved as a typed, versioned **capability**. After that, the capability runs again and again with no model involved. If a run gets stuck, a person can take over the same live browser session and hand it back.

```
goal ─► discover (LLM, observe/decide/act) ─► capability artifact (JSON) ─► replay (no LLM) ─► result
                                                        │                       │
                                                        └── agent-facing catalog └── stuck? ─► a person takes the live session ─► hand back ─► resume
```

The design write-up is in [`REPORT.md`](REPORT.md). Evidence of real runs is in [`evidence/`](evidence/README.md).

**Built with Claude Code.** Most of the code, tests and prose in this repository were written with Claude Code (Anthropic's coding assistant), with me setting the design, reviewing it, and running the live model discovery myself. The decisions in `REPORT.md` are mine to defend.

## The target

The application is a stand-in I built, not a real bank system: `targets/mockbank`, a synthetic "MemberServ 3.1" credit-union back office on `127.0.0.1:4310`. It is deliberately hostile: framesets, nested table layout, no test IDs, element ids regenerated on every render, JavaScript-only links, an irreversible "Close account" link and an admin page. All data is made up. It can inject runtime faults (session timeout, slow load, unexpected notice, validation error, application error) and keeps a server-side log of what actually happened, so tests can check the truth rather than the client's claim.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run playwright install chromium
cp .env.example .env        # then put ONE model key in .env (below)
```

`.env` is gitignored. Only **discovery** calls a model; replay, the catalog and the whole test suite need no key. The mock's sign-in (`teller01` / `demo-only`) is synthetic and already in `.env.example`.

Discovery works with any of three models, chosen with `cua run --provider`:

| `--provider` | Needs | Notes |
|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | Claude; prompt caching; screenshots sent alongside the page description |
| `gemini` | `GEMINI_API_KEY`, free from [aistudio.google.com/apikey](https://aistudio.google.com/apikey), no card | Gemini 2.5 Flash by default; the free tier is rate limited, and the adapter waits and retries |
| `ollama` | a local `ollama serve` | No key and nothing leaves the machine. Text only. I tried `llama3.1:8b` and it was too weak for this loop (it typed a member number into the password field), so use a larger model |

## Run it without any live service

No key and no network are needed for the tests: they use a scripted stand-in for the model (`FakeLLM`) that reads the rendered page the way a real model does, and the mock application runs in-process.

```bash
make check     # ruff, format check, mypy --strict, 800 tests in real Chromium
make cov       # the same with coverage: 70% overall, 90% on the load-bearing modules
```

## Demo path

Terminal 1 starts the application:

```bash
make mock
```

Terminal 2:

```bash
# 1. Discover (the only step that uses the model). Saves capabilities/member_lookup.json
uv run cua run --task member_lookup --provider gemini --evidence evidence/01-discovery
#    (--provider anthropic is the default; the model used is recorded in evidence/01-discovery/summary.json)

# 2. Read what it learned, the way a reviewer would
uv run cua show member_lookup

# 3. Replay it for a different member. No model. Exit code 0
uv run cua replay member_lookup -p member_id=12346 --evidence runs/success

# 4. A business outcome, not a crash: no such member. Exit code 10
uv run cua replay member_lookup -p member_id=99999 --evidence runs/not-found

# 5. An injected failure: the application crashes when searching. Exit code 30, with evidence
curl -s -X POST 127.0.0.1:4310/_admin/faults -d '{"mode":"app_error","step":"results","times":1}'
uv run cua replay member_lookup -p member_id=12345 --evidence runs/crash

# 6. A person takes over. The session expires mid-run; the run stops and asks.
curl -s -X POST 127.0.0.1:4310/_admin/faults -d '{"mode":"session_timeout","step":"results","times":1}'
uv run cua replay member_lookup -p member_id=12345 --headed --operator both --evidence runs/handoff
#    Follow the prompt (type "take"), sign in and search again in the browser window, type "done".
#    The run re-checks the page and finishes. (--operator both also opens a web page for this.)

# 7. Call it the way an AI agent would: JSON in, MCP-shaped reply out
uv run cua capabilities call member_lookup --args '{"member_id": "12346"}'
```

To regenerate evidence 02 to 06 through the real CLI, with a scripted second client playing the person in step 6:

```bash
make evidence
```

Exit codes: `0` success, `10` business outcome, `20` escalated to a person, `30` hard failure, `2` bad usage. `--json` prints only the result document.

## Layout

| Path | What lives there |
|---|---|
| `src/cua/surface/` | The seam between "how we perceive and act" and "the recorded flow". `PlaywrightSurface` observes frame by frame (accessibility snapshot plus screenshot), harvests and resolves locators, records a person's actions |
| `src/cua/agent/` | The discovery loop: observe, decide, act, with limits (steps, time, repeats, stalled page) |
| `src/cua/artifact/` | The capability schema, synthesis from a recorded run, review rendering, storage |
| `src/cua/replay/` | The deterministic engine and its result contract |
| `src/cua/control/` | The control lease, stuck detection, the handoff, and the terminal and web operator |
| `src/cua/gateway.py`, `policy.py`, `redact.py` | Every action goes through the gateway: allowlist, verbs, risk, audit. Redaction is applied to everything written |
| `src/cua/cli.py`, `catalog.py`, `runner.py` | The `cua` command, the agent-facing catalog, and the composition root |
| `targets/mockbank/` | The synthetic application and its fault injection |
| `schemas/` | JSON Schema of the capability and the replay result (tests fail if they go stale) |
| `capabilities/` | Saved capabilities and `catalog.json` |
| `scripts/` | Coverage gates, evidence generation and checks, the simulated operator |

## What is and is not real

Real: the discovery loop against a live model, the artifact, deterministic replay in real Chromium, the error taxonomy, the gateway and network guard, the control lease and a handoff on the live session (a second client attached to the same browser), redaction, the agent-facing catalog.

Stubbed or designed only, and why: see the last section of `REPORT.md`. The short version is that the desktop surface, multi-tenant overlays and an operator console with live co-browsing are designs, not code.
