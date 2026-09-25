.PHONY: sync lint format format-check typecheck test cov check build clean mock evidence

sync:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy --strict src tests targets

test:
	uv run pytest -q

# Same coverage gates CI enforces: 70% overall, 90% on the load-bearing modules.
cov:
	uv run pytest -q --cov --cov-report=term-missing --cov-report=json
	uv run python scripts/check_coverage.py

check: lint format-check typecheck test

build:
	uv build

clean:
	rm -rf dist .pytest_cache .mypy_cache .ruff_cache .coverage coverage.json htmlcov

# The synthetic legacy application on 127.0.0.1:4310 (sign-in teller01 / demo-only).
mock:
	uv run python -m targets.mockbank

# Regenerate evidence/02..06 with the real CLI (needs capabilities/member_lookup.json from a live
# `cua run`), then verify nothing sensitive was kept.
evidence:
	uv run python scripts/make_evidence.py
	scripts/check_evidence_clean.sh
