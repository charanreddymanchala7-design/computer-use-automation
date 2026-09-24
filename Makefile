.PHONY: sync lint format format-check typecheck test cov check build clean

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
