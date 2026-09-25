"""Regenerate the replay evidence (evidence/02..06) by running the real CLI against the mock.

Discovery (evidence/01-discovery) needs a live model and is run by hand with `cua run`; everything
here uses that saved capability and needs no key. Each run keeps what the CLI keeps: the redacted
log, the result, the capability used, screenshots on trouble, plus `command.txt` and `output.txt`
so a reader can see exactly what was typed and printed.

    uv run python scripts/make_evidence.py
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
CAPABILITY = ROOT / "capabilities" / "member_lookup.json"
USER, PASSWORD = "teller01", "demo-only"  # the mock's synthetic, public sign-in


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Mock:
    def __init__(self) -> None:
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.process = subprocess.Popen(
            [sys.executable, "-m", "targets.mockbank", "--port", str(self.port)],
            cwd=ROOT,
            env={**os.environ, "MOCK_USER": USER, "MOCK_PASS": PASSWORD},
            stdout=subprocess.DEVNULL,
        )
        for _ in range(100):
            try:
                urllib.request.urlopen(self.base + "/_admin/log", timeout=1).read()
                return
            except OSError:
                time.sleep(0.1)
        raise SystemExit("the mock did not start")

    def post(self, path: str, body: dict[str, object]) -> None:
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(request, timeout=5).read()

    def reset(self) -> None:
        self.post("/_admin/reset", {})

    def arm(self, mode: str, step: str) -> None:
        self.post("/_admin/faults", {"mode": mode, "step": step, "times": 1})

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=10)


def cli_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # replay needs no key
    env.update({"MOCK_USER": USER, "MOCK_PASS": PASSWORD, "NO_COLOR": "1"})
    return env


def record(folder: Path, args: list[str], output: str, exit_code: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "command.txt").write_text("uv run cua " + " ".join(args) + "\n", encoding="utf-8")
    (folder / "output.txt").write_text(output + f"\n(exit code {exit_code})\n", encoding="utf-8")


def run_cli(name: str, args: list[str], expect: int) -> subprocess.CompletedProcess[str]:
    folder = EVIDENCE / name
    full = [*args, "--evidence", f"evidence/{name}"]
    done = subprocess.run(
        [sys.executable, "-m", "cua", *full],
        cwd=ROOT,
        env=cli_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    record(folder, full, done.stdout + done.stderr, done.returncode)
    print(f"{name}: exit {done.returncode} (expected {expect})")
    if done.returncode != expect:
        raise SystemExit(
            f"{name} exited {done.returncode}, expected {expect}\n{done.stdout}{done.stderr}"
        )
    return done


def main() -> None:
    if not CAPABILITY.is_file():
        raise SystemExit(
            f"{CAPABILITY} is missing: discover it first with\n"
            "  uv run cua run --task member_lookup --evidence evidence/01-discovery"
        )
    for stale in EVIDENCE.glob("0[2-6]-*"):
        shutil.rmtree(stale)
    mock = Mock()
    common = ["--target", mock.base]
    try:
        run_cli(
            "02-replay-success",
            ["replay", "member_lookup", "-p", "member_id=12346", *common],
            expect=0,
        )

        mock.reset()
        run_cli(
            "03-business-outcome",
            ["replay", "member_lookup", "-p", "member_id=99999", *common],
            expect=10,
        )

        mock.reset()
        mock.arm("app_error", "results")  # injected: the application crashes when searching
        run_cli(
            "04-hard-failure",
            ["replay", "member_lookup", "-p", "member_id=12345", *common],
            expect=30,
        )

        mock.reset()
        handoff(mock)

        mock.reset()
        agent_call(mock)
    finally:
        mock.stop()


def handoff(mock: Mock) -> None:
    """The session expires mid-run; a person (a second client on the same browser) fixes it."""
    name = "05-handoff"
    mock.arm("session_timeout", "results")
    debug_port, operator_port = free_port(), free_port()
    args = [
        "replay",
        "member_lookup",
        "-p",
        "member_id=12345",
        "--target",
        mock.base,
        "--operator",
        "web",
        "--operator-port",
        str(operator_port),
        "--debug-port",
        str(debug_port),
        "--wait-for-human",
        "120",
        "--evidence",
        f"evidence/{name}",
    ]
    replay = subprocess.Popen(
        [sys.executable, "-m", "cua", *args],
        cwd=ROOT,
        env=cli_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    person = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "simulate_operator.py"),
            "--operator",
            f"http://127.0.0.1:{operator_port}",
            "--cdp",
            f"http://127.0.0.1:{debug_port}",
            "--user",
            USER,
            "--password",
            PASSWORD,
            "--member",
            "12345",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    out, err = replay.communicate(timeout=180)
    record(EVIDENCE / name, args, out + err, replay.returncode)
    (EVIDENCE / name / "person.txt").write_text(
        "The person is scripts/simulate_operator.py: a second client attached to the run's own\n"
        "browser over its loopback debug port, using the operator API to take control and\n"
        "hand back.\n" + person.stdout + person.stderr,
        encoding="utf-8",
    )
    print(f"{name}: exit {replay.returncode} (expected 0)")
    if replay.returncode != 0 or person.returncode != 0:
        raise SystemExit(f"{name} failed\n{out}{err}\n{person.stdout}{person.stderr}")


def agent_call(mock: Mock) -> None:
    """An agent-style call: JSON arguments in, an MCP-shaped reply out."""
    name = "06-agent-call"
    args = [
        "capabilities",
        "call",
        "member_lookup",
        "--args",
        '{"member_id": "12346"}',
        "--target",
        mock.base,
        "--evidence",
        f"evidence/{name}",
    ]
    done = subprocess.run(
        [sys.executable, "-m", "cua", *args],
        cwd=ROOT,
        env=cli_env(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    record(EVIDENCE / name, args, done.stdout + done.stderr, done.returncode)
    (EVIDENCE / name / "reply.json").write_text(done.stdout, encoding="utf-8")
    print(f"{name}: exit {done.returncode} (expected 0)")
    if done.returncode != 0:
        raise SystemExit(f"{name} failed\n{done.stdout}{done.stderr}")


if __name__ == "__main__":
    main()
