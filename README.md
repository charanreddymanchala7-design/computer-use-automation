# Computer-Use Automation System

An LLM discovers how to complete a task in a legacy back-office UI, the successful run is
saved as a typed, versioned capability, and that capability is replayed deterministically
without the model. A human can take over the live session when the system is stuck.

Status: work in progress. Setup, demo commands and the design write-up (`REPORT.md`) land
with the final tasks.

Built with Claude Code assistance; the design decisions are the author's.

## Development

```bash
uv sync
make check   # lint, format check, type check, tests
```
