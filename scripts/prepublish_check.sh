#!/usr/bin/env bash
# Everything that must be true before the repository goes public. Read-only: it changes nothing.
#
#   scripts/prepublish_check.sh            all checks, including a fresh clone and the full suite
#   scripts/prepublish_check.sh --quick    skip the fresh clone and the test run
set -uo pipefail
cd "$(dirname "$0")/.."

quick=0
[ "${1:-}" = "--quick" ] && quick=1
failures=0
pass() { echo "[ok] $*"; }
warn() { echo "[!!] $*"; }
fail() { echo "[xx] $*" >&2; failures=$((failures + 1)); }

# --- 1. what is committed ------------------------------------------------------------------------
if [ -z "$(git status --porcelain)" ]; then pass "working tree is clean"; else fail "uncommitted changes:"; git status --short >&2; fi

if git log --all --diff-filter=A --name-only --format= | grep -Eq '(^|/)\.env($|\.)' \
   && git log --all --diff-filter=A --name-only --format= | grep -E '(^|/)\.env($|\.)' | grep -qv '\.env\.example'; then
  fail "a .env file was committed at some point in history"
else
  pass "no .env file ever committed"
fi

# --- 2. secrets and personal data anywhere in history --------------------------------------------
history_patterns=(
  'sk-ant-[A-Za-z0-9_-]{8,}'
  'ANTHROPIC_API_KEY=[A-Za-z0-9_-]{8,}'
  'gh[pousr]_[A-Za-z0-9]{20,}'
  'AKIA[0-9A-Z]{16}'
  '-----BEGIN [A-Z ]*PRIVATE KEY'
)
leaks=0
for pattern in "${history_patterns[@]}"; do
  if git log --all -p --format= | grep -Eq -e "$pattern"; then
    fail "history matches /$pattern/"
    leaks=1
  fi
done
[ "$leaks" -eq 0 ] && pass "no keys or tokens anywhere in $(git rev-list --all --count) commits"

authors=$(git log --all --format='%ae' | sort -u)
if echo "$authors" | grep -Evq 'users\.noreply\.github\.com$'; then
  warn "these author emails will be public in the history (a GitHub squash-merge uses the account's"
  warn "primary email). Fine if you are happy for it to be seen; otherwise decide before going public:"
  echo "$authors" | grep -Ev 'users\.noreply\.github\.com$' | sed 's/^/     /'
else
  pass "every commit is authored with a GitHub noreply address"
fi

if git ls-files | grep -Eq '\.(har|webm|mp4|zip)$|(^|/)\.browser-profile/|(^|/)trace'; then
  fail "raw browser artifacts are tracked"; git ls-files | grep -E '\.(har|webm|mp4|zip)$|trace' >&2
else
  pass "no traces, HAR files, videos or browser profiles tracked"
fi

# --- 3. the evidence ---------------------------------------------------------------------------------
for dir in 01-discovery 02-replay-success 03-business-outcome 04-hard-failure 05-handoff 06-agent-call; do
  [ -d "evidence/$dir" ] && [ -n "$(ls -A "evidence/$dir")" ] || fail "evidence/$dir is missing or empty"
done
[ -f evidence/README.md ] || fail "evidence/README.md is missing"
scripts/check_evidence_clean.sh >/dev/null 2>&1 && pass "evidence is clean" \
  || { fail "evidence check failed:"; scripts/check_evidence_clean.sh >&2; }

# --- 4. the deliverables --------------------------------------------------------------------------------
for file in README.md REPORT.md capabilities/member_lookup.json capabilities/catalog.json; do
  [ -f "$file" ] || fail "$file is missing"
done
expected="Architecture|Artifact schema|Determinism & error handling|Heterogeneity & multi-tenant|Escalation & handoff|Safety|Cuts"
actual=$(grep '^#' REPORT.md | sed -E 's/^#+ +//' | paste -sd'|' -)
if [ "$actual" = "$expected" ]; then pass "REPORT.md has exactly the seven headings, in order"; else fail "REPORT.md headings are: $actual"; fi
grep -qi 'claude code' README.md && pass "README discloses the AI assistance" || fail "README does not disclose AI assistance"
uv run cua capabilities export --check >/dev/null 2>&1 && pass "capabilities/catalog.json is current" || fail "catalog.json is out of date"

# --- 5. the code ----------------------------------------------------------------------------------------------
if [ "$quick" -eq 1 ]; then
  echo "[--] --quick: skipping the fresh clone and the test suite"
else
  work=$(mktemp -d)
  trap 'rm -rf "$work"' EXIT
  if git clone -q . "$work/clone" && (
       cd "$work/clone" \
       && uv sync -q \
       && uv run ruff check . >/dev/null \
       && uv run ruff format --check . >/dev/null \
       && uv run mypy src tests targets >/dev/null \
       && uv run pytest -q -p no:cacheprovider --cov --cov-report=json >/dev/null \
       && uv run python scripts/check_coverage.py --strict-missing >/dev/null
     ); then
    pass "a fresh clone passes lint, types, tests and the coverage gates"
  else
    fail "the fresh clone did not pass (run its steps by hand in $work/clone)"; trap - EXIT
  fi
fi

echo
if [ "$failures" -eq 0 ]; then
  echo "READY: nothing found that should stop this going public."
  echo "Still yours to confirm: open the screenshots in evidence/ once, and read docs/submission.md."
else
  echo "NOT READY: $failures problem(s) above." >&2
  exit 1
fi
