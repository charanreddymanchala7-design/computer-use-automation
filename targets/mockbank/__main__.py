"""Run the mock: `uv run python -m targets.mockbank` (loopback only, synthetic data)."""

from __future__ import annotations

import argparse
import os

from targets.mockbank.server import make_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4310)
    args = parser.parse_args()
    server = make_server(
        args.host,
        args.port,
        user=os.environ.get("MOCK_USER", "teller01"),
        password=os.environ.get("MOCK_PASS", "demo-only"),
        faults=os.environ.get("MOCK_FAULTS", ""),
    )
    print(f"MemberServ 3.1 (SYNTHETIC) at http://{args.host}:{args.port}/msv/login.cgi")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
