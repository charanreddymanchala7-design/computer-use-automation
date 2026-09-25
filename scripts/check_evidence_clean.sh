#!/usr/bin/env bash
# Fail if anything under evidence/ could leak a credential, a personal path or raw browser state.
# Text files are grepped; screenshots cannot be, so evidence/README.md states what they show.
set -euo pipefail
cd "$(dirname "$0")/.."

status=0
fail() { echo "FAIL: $*" >&2; status=1; }

[ -d evidence ] || { echo "no evidence/ folder"; exit 1; }

# only redacted, reviewable file types
while IFS= read -r file; do
  case "$file" in
    *.json|*.jsonl|*.md|*.png|*.txt) ;;
    *) fail "unexpected file type: $file" ;;
  esac
done < <(find evidence -type f)

# nothing raw: traces, HAR, video, browser profiles
if find evidence \( -name '*.har' -o -name 'trace*.zip' -o -name '*.webm' -o -name '*.mp4' \
     -o -name 'Cookies' -o -name 'Local State' -o -name '.env*' \) | grep -q .; then
  fail "raw browser or environment artifacts present"
fi

# no file is large enough to hide a dump
while IFS= read -r big; do fail "file over 2 MB: $big"; done < <(find evidence -type f -size +2M)

patterns=(
  'sk-ant-[A-Za-z0-9_-]{8,}'          # an Anthropic key
  'ANTHROPIC_API_KEY=[^[:space:]]+'   # a key assignment
  'demo-only'                          # the mock password, which must always be redacted
  'teller01'                           # the mock user, likewise
  'Authorization:|Bearer [A-Za-z0-9._-]{12,}'
  '-----BEGIN [A-Z ]*PRIVATE KEY'
  '/Users/[A-Za-z0-9._-]+'             # a local home directory
  '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'  # an email address
)
for pattern in "${patterns[@]}"; do
  if grep -rEIl --exclude='*.png' -e "$pattern" evidence >/dev/null 2>&1; then
    fail "matches /$pattern/:"
    grep -rEIn --exclude='*.png' -e "$pattern" evidence | cut -c1-160 | head -5 >&2
  fi
done

[ "$status" -eq 0 ] && echo "evidence is clean ($(find evidence -type f | wc -l | tr -d ' ') files)"
exit "$status"
