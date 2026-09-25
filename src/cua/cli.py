"""The ``cua`` command line.

``cua run``      discover a capability with a model (needs an API key)
``cua show``     read a saved capability the way a reviewer would
``cua replay``   run a saved capability deterministically, with no model

Exit codes follow the replay result: 0 success, 10 business outcome, 20 escalated to a person,
30 hard failure; 2 is a usage error. With ``--json`` stdout carries only the result document.
Secrets come from the environment or a gitignored ``.env``, never from a flag.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from cua import __version__
from cua.agent import Limits
from cua.artifact import Capability, RiskClass
from cua.artifact.render import render_review
from cua.artifact.store import list_capabilities, load_capability
from cua.catalog import ToolError, build_catalog, call_tool, tool_definition
from cua.config import load_dotenv, use_color
from cua.demo.memberserv import TASKS
from cua.evlog import EventLog
from cua.llm import LLM, LLMConfigError
from cua.llm.anthropic_llm import AnthropicLLM
from cua.llm.gemini_llm import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
from cua.llm.gemini_llm import GeminiLLM
from cua.llm.ollama_llm import DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL
from cua.llm.ollama_llm import OllamaLLM
from cua.policy import Policy
from cua.report import mark_up, mask_inputs, progress_lines, render_result
from cua.runner import OPERATORS, discover_task, load_policy, replay_capability

DEFAULT_TARGET = "http://127.0.0.1:4310"
DEFAULT_POLICY = Path("policies/memberserv.json")
PROVIDERS = {
    "anthropic": "claude-sonnet-5",
    "gemini": GEMINI_DEFAULT_MODEL,
    "ollama": OLLAMA_DEFAULT_MODEL,
}

app = typer.Typer(
    name="cua",
    no_args_is_help=True,
    add_completion=False,
    help="Discover UI capabilities with a model, replay them without one, hand off to people.",
)


def make_llm(provider: str, model: str, log: EventLog) -> LLM:
    """The one place a live model is created; tests substitute a scripted one here."""
    if provider == "ollama":
        return OllamaLLM(model, log=log)
    if provider == "gemini":
        return GeminiLLM(model, log=log)
    return AnthropicLLM(model, log=log)


# --- helpers -----------------------------------------------------------------------------------


def find_capability(ref: str, directory: Path = Path("capabilities")) -> Capability:
    """A capability by path, or by id from the capabilities folder."""
    path = Path(ref)
    if not path.is_file():
        path = directory / f"{ref}.json"
    if not path.is_file():
        raise typer.BadParameter(f"no capability {ref!r} (also looked for {path})")
    try:
        return load_capability(path)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "document"
        raise typer.BadParameter(
            f"not a valid capability: {path} ({where}: {first['msg']})"
        ) from exc
    except ValueError as exc:
        raise typer.BadParameter(f"not a valid capability: {path} (not JSON)") from exc


def coerce_inputs(schema: dict[str, Any], pairs: list[str]) -> dict[str, Any]:
    """``name=value`` pairs, typed by the capability's own input schema."""
    properties: dict[str, Any] = schema.get("properties", {})
    typed: dict[str, Any] = {}
    for pair in pairs:
        name, sep, raw = pair.partition("=")
        if not sep or not name:
            raise typer.BadParameter(f"{pair!r}: expected name=value", param_hint="--param")
        kind = properties.get(name, {}).get("type", "string")
        try:
            typed[name] = _typed(kind, raw)
        except ValueError:
            raise typer.BadParameter(
                f"{name}: {raw!r} is not a valid {kind}", param_hint="--param"
            ) from None
    return typed


def _typed(kind: str, raw: str) -> Any:
    if kind == "integer":
        return int(raw)
    if kind == "number":
        return float(raw)
    if kind == "boolean":
        if raw.lower() in ("true", "1", "yes"):
            return True
        if raw.lower() in ("false", "0", "no"):
            return False
        raise ValueError(raw)
    return raw


def _policy(path: Path) -> Policy:
    try:
        return load_policy(path)
    except OSError:
        raise typer.BadParameter(
            f"policy file {path} cannot be read", param_hint="--policy"
        ) from None
    except ValidationError as exc:
        raise typer.BadParameter(
            f"policy file {path} is invalid: {exc}", param_hint="--policy"
        ) from None


def _secrets(names: Sequence[str]) -> dict[str, str]:
    return {name: os.environ[name] for name in names if os.environ.get(name)}


def _color(no_color: bool) -> bool:
    return not no_color and use_color(isatty=sys.stdout.isatty(), env=os.environ)


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# --- commands ----------------------------------------------------------------------------------


@app.callback()
def _root(
    version: Annotated[bool, typer.Option("--version", help="Show the version and exit.")] = False,
) -> None:
    if version:
        typer.echo(f"cua {__version__}")
        raise typer.Exit()


@app.command()
def show(
    capability: Annotated[str, typer.Argument(help="A capability file, or an id in capabilities/")],
) -> None:
    """Print a capability the way a reviewer reads it: steps, risk, error map, secrets needed."""
    typer.echo(render_review(find_capability(capability)))


@app.command()
def replay(
    capability: Annotated[str, typer.Argument(help="A capability file, or an id in capabilities/")],
    param: Annotated[
        list[str] | None, typer.Option("--param", "-p", help="An input, as name=value (repeatable)")
    ] = None,
    target: Annotated[str, typer.Option(help="Base URL of the application")] = DEFAULT_TARGET,
    policy: Annotated[Path, typer.Option(help="Allowlist and guardrails (JSON)")] = DEFAULT_POLICY,
    evidence: Annotated[
        Path | None, typer.Option(help="Where to keep the log, result and screenshots")
    ] = None,
    headed: Annotated[bool, typer.Option(help="Show the browser window")] = False,
    debug_port: Annotated[
        int | None, typer.Option(help="Loopback port for a second client to attach to the browser")
    ] = None,
    operator: Annotated[
        str, typer.Option(help=f"Who is asked when the run is stuck: {', '.join(OPERATORS)}")
    ] = "none",
    operator_port: Annotated[
        int, typer.Option(help="Port for the operator web page (0 picks a free one)")
    ] = 0,
    wait_for_human: Annotated[
        float, typer.Option(help="Seconds to wait for a person before giving up")
    ] = 900.0,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print only the result document, as JSON, on stdout")
    ] = False,
    no_color: Annotated[
        bool, typer.Option("--no-color", help="Never colour output (NO_COLOR is honoured too)")
    ] = False,
) -> None:
    """Run a capability with no model in the loop. Exit code: 0 success, 10 business outcome,
    20 escalated, 30 hard failure."""
    load_dotenv(Path(".env"))
    if operator not in OPERATORS:
        raise typer.BadParameter(
            f"{operator!r}: choose one of {', '.join(OPERATORS)}", param_hint="--operator"
        )
    cap = find_capability(capability)
    inputs = coerce_inputs(cap.inputs, param or [])
    guard = _policy(policy)
    folder = evidence or Path("runs") / f"{cap.id}-{_stamp()}"
    color = _color(no_color)
    if not as_json:
        shown = mask_inputs(inputs, cap.sensitive_params())
        typer.echo(
            mark_up("info", f"{cap.id}@{cap.capability_version} inputs: {shown}", color=color)
        )

    def announce(text: str) -> None:
        typer.echo(mark_up("info", text, color=color), err=True)

    ran = replay_capability(
        cap,
        inputs,
        base_url=target,
        policy=guard,
        evidence_dir=folder,
        secrets=_secrets(cap.required_secrets()),
        headed=headed,
        debug_port=debug_port,
        operator=operator,
        operator_port=operator_port,
        claim_timeout_s=wait_for_human,
        announce=announce,
    )
    if as_json:
        typer.echo(ran.result.model_dump_json(indent=2))
    else:
        for line in progress_lines(ran.events, cap, color=color):
            typer.echo(line)
        typer.echo("")
        typer.echo(render_result(ran.result, color=color))
        typer.echo(mark_up("info", f"evidence: {folder}", color=color))
    raise typer.Exit(ran.result.exit_code)


@app.command()
def run(
    task: Annotated[str, typer.Option(help=f"What to discover: {', '.join(TASKS)}")],
    target: Annotated[str, typer.Option(help="Base URL of the application")] = DEFAULT_TARGET,
    policy: Annotated[Path, typer.Option(help="Allowlist and guardrails (JSON)")] = DEFAULT_POLICY,
    evidence: Annotated[
        Path | None, typer.Option(help="Where to keep the log and screenshots")
    ] = None,
    capabilities: Annotated[Path, typer.Option(help="Where the capability is saved")] = Path(
        "capabilities"
    ),
    provider: Annotated[
        str, typer.Option(help=f"Where the model runs: {', '.join(PROVIDERS)}")
    ] = "anthropic",
    model: Annotated[
        str, typer.Option(help="The model that explores it (default depends on the provider)")
    ] = "",
    headed: Annotated[bool, typer.Option(help="Show the browser window")] = False,
    max_steps: Annotated[int, typer.Option(help="Model calls before giving up")] = 40,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
) -> None:
    """Discover a capability: the model explores the application once and the result is saved
    as a reviewable artifact. Needs ANTHROPIC_API_KEY in the environment or .env."""
    load_dotenv(Path(".env"))
    if task not in TASKS:
        raise typer.BadParameter(f"{task!r}: choose one of {', '.join(TASKS)}", param_hint="--task")
    if provider not in PROVIDERS:
        raise typer.BadParameter(
            f"{provider!r}: choose one of {', '.join(PROVIDERS)}", param_hint="--provider"
        )
    guard = _policy(policy)
    chosen = model or os.environ.get("CUA_MODEL") or PROVIDERS[provider]
    folder = evidence or Path("runs") / f"discover-{task}-{_stamp()}"
    color = _color(no_color)
    names = TASKS[task][0](target).secrets
    typer.echo(
        mark_up("info", f"discovering '{task}' with {provider}:{chosen} on {target}", color=color)
    )
    try:
        found = discover_task(
            task,
            lambda log: make_llm(provider, chosen, log),
            base_url=target,
            policy=guard,
            evidence_dir=folder,
            capabilities_dir=capabilities,
            secrets=_secrets(names),
            headed=headed,
            limits=Limits(max_steps=max_steps),
        )
    except LLMConfigError as exc:
        typer.echo(mark_up("fail", str(exc), color=color), err=True)
        raise typer.Exit(2) from None
    rec = found.recorded
    cost = f", ${rec.cost_usd:.2f}" if rec.cost_usd is not None else ""
    if found.capability is None:
        typer.echo(
            mark_up("fail", f"discovery {rec.outcome}: {rec.reason or found.error}", color=color)
        )
        typer.echo(mark_up("info", f"evidence: {folder}", color=color))
        raise typer.Exit(30)
    typer.echo(
        mark_up(
            "ok",
            f"finished: {len(rec.steps)} steps recorded from {rec.llm_steps} model calls, "
            f"{rec.tokens:,} tokens{cost}",
            color=color,
        )
    )
    for warning in found.warnings:
        typer.echo(mark_up("warn", warning, color=color))
    typer.echo(mark_up("info", f"capability saved: {found.capability_path}", color=color))
    typer.echo(mark_up("info", f"evidence: {folder}", color=color))


# --- the agent-facing catalog -------------------------------------------------------------------

catalog_app = typer.Typer(
    no_args_is_help=True,
    help="Capabilities as tools an AI agent can list, inspect and call (MCP-style definitions).",
)
app.add_typer(catalog_app, name="capabilities")

CapabilityDir = Annotated[Path, typer.Option("--dir", help="Where the capabilities are saved")]


def _kind(capability: Capability) -> str:
    risks = {step.risk_class for step in capability.steps}
    if RiskClass.IRREVERSIBLE_WRITE in risks:
        return "irreversible"
    return "writes" if RiskClass.REVERSIBLE_WRITE in risks else "read-only"


@catalog_app.command("list")
def catalog_list(directory: CapabilityDir = Path("capabilities")) -> None:
    """List the tools an agent can call, and whether each only reads or changes state."""
    found = list_capabilities(directory)
    if not found:
        typer.echo(f"no capabilities in {directory}")
        return
    for c in found:
        names = ", ".join(c.inputs.get("properties", {})) or "no inputs"
        typer.echo(f"{c.id:<20} {c.capability_version:<7} {_kind(c):<13} inputs: {names}")


@catalog_app.command("schema")
def catalog_schema(
    name: Annotated[str, typer.Argument(help="The capability id")],
    directory: CapabilityDir = Path("capabilities"),
) -> None:
    """Print one tool's definition: name, description, input and output schema, annotations."""
    found = {c.id: c for c in list_capabilities(directory)}
    if name not in found:
        raise typer.BadParameter(
            f"unknown tool {name!r}; available: {', '.join(sorted(found)) or 'none'}"
        )
    typer.echo(json.dumps(tool_definition(found[name]), indent=2))


@catalog_app.command("export")
def catalog_export(
    directory: CapabilityDir = Path("capabilities"),
    out: Annotated[Path, typer.Option(help="The catalog file to write")] = Path(
        "capabilities/catalog.json"
    ),
    check: Annotated[
        bool, typer.Option("--check", help="Write nothing; exit 1 if the file is out of date")
    ] = False,
) -> None:
    """Write capabilities/catalog.json from the saved capabilities."""
    text = json.dumps(build_catalog(list_capabilities(directory)), indent=2) + "\n"
    if check:
        if not out.is_file() or out.read_text(encoding="utf-8") != text:
            typer.echo(f"{out} is out of date: run `cua capabilities export`", err=True)
            raise typer.Exit(1)
        typer.echo(f"{out} is current")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {out}")


@catalog_app.command("call")
def catalog_call(
    name: Annotated[str, typer.Argument(help="The capability id")],
    args: Annotated[str, typer.Option(help="Arguments as a JSON object")] = "{}",
    directory: CapabilityDir = Path("capabilities"),
    target: Annotated[str, typer.Option(help="Base URL of the application")] = DEFAULT_TARGET,
    policy: Annotated[Path, typer.Option(help="Allowlist and guardrails (JSON)")] = DEFAULT_POLICY,
    evidence: Annotated[Path | None, typer.Option(help="Where to keep the evidence")] = None,
    headed: Annotated[bool, typer.Option(help="Show the browser window")] = False,
    debug_port: Annotated[int | None, typer.Option(help="Loopback debug port")] = None,
    operator: Annotated[str, typer.Option(help=f"One of {', '.join(OPERATORS)}")] = "none",
    operator_port: Annotated[int, typer.Option(help="Port for the operator web page")] = 0,
    wait_for_human: Annotated[float, typer.Option(help="Seconds to wait for a person")] = 900.0,
) -> None:
    """Call a tool the way an agent would: JSON in, an MCP-shaped reply on stdout.

    Exit code: 0 completed, 10 business outcome, 20 escalated, 30 hard failure, 2 bad call."""
    load_dotenv(Path(".env"))
    if operator not in OPERATORS:
        raise typer.BadParameter(
            f"{operator!r}: choose one of {', '.join(OPERATORS)}", param_hint="--operator"
        )
    try:
        arguments = json.loads(args)
    except ValueError:
        raise typer.BadParameter("--args must be a JSON object", param_hint="--args") from None
    if not isinstance(arguments, dict):
        raise typer.BadParameter("--args must be a JSON object", param_hint="--args")
    guard = _policy(policy)
    exit_code = 0

    def run(capability: Capability, given: dict[str, Any]) -> Any:
        nonlocal exit_code
        folder = evidence or Path("runs") / f"call-{capability.id}-{_stamp()}"
        ran = replay_capability(
            capability,
            given,
            base_url=target,
            policy=guard,
            evidence_dir=folder,
            secrets=_secrets(capability.required_secrets()),
            headed=headed,
            debug_port=debug_port,
            operator=operator,
            operator_port=operator_port,
            claim_timeout_s=wait_for_human,
            announce=lambda text: typer.echo(text, err=True),
        )
        exit_code = ran.result.exit_code
        return ran.result

    try:
        reply = call_tool(list_capabilities(directory), name, arguments, run)
    except ToolError as exc:
        raise typer.BadParameter(str(exc)) from None
    typer.echo(json.dumps(reply, indent=2))
    raise typer.Exit(exit_code)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
