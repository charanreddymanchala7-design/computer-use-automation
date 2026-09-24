"""Write the JSON Schemas of the public contracts to schemas/. Tests fail if they go stale."""

from __future__ import annotations

import json
from pathlib import Path

from cua.artifact import capability_json_schema

OUT = Path(__file__).resolve().parent.parent / "schemas"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    targets = {"capability.schema.json": capability_json_schema()}
    for name, schema in targets.items():
        (OUT / name).write_text(json.dumps(schema, indent=2) + "\n")
        print(f"wrote schemas/{name}")


if __name__ == "__main__":
    main()
