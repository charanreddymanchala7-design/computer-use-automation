"""Per-module coverage gate: at least 90% on the load-bearing modules (decision Q6).

Reads coverage.json (pytest --cov --cov-report=json). The 70% overall floor is enforced by
coverage.py itself via [tool.coverage.report] fail_under.

Modules that do not exist yet are skipped so the gate can run from the first commit; pass
--strict-missing (used by the pre-publish gate) to fail when a load-bearing module is absent.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

THRESHOLD = 90.0

# label -> glob relative to the repo root
LOAD_BEARING: dict[str, str] = {
    "artifact schema": "src/cua/artifact/schema.py",
    "result contract": "src/cua/result.py",
    "replay engine": "src/cua/replay/*.py",
    "action gateway": "src/cua/gateway.py",
    "policy": "src/cua/policy.py",
    "redaction": "src/cua/redact.py",
    "control lease": "src/cua/control/lease.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="coverage.json", type=Path)
    parser.add_argument("--strict-missing", action="store_true")
    args = parser.parse_args()

    if not args.report.exists():
        print(f"missing {args.report}; run pytest with --cov --cov-report=json first")
        return 2
    files: dict[str, dict[str, dict[str, float]]] = json.loads(args.report.read_text())["files"]

    failed = False
    for label, pattern in LOAD_BEARING.items():
        matches = sorted(p for p in files if fnmatch.fnmatch(p, pattern))
        if not matches:
            status = "MISSING" if args.strict_missing else "skip (not built yet)"
            print(f"{status:22} {label}: {pattern}")
            failed = failed or args.strict_missing
            continue
        for path in matches:
            pct = files[path]["summary"]["percent_covered"]
            ok = pct >= THRESHOLD
            failed = failed or not ok
            print(f"{'ok' if ok else 'BELOW ' + str(THRESHOLD):22} {label}: {path} {pct:.1f}%")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
