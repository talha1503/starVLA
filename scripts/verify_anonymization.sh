#!/usr/bin/env bash
set -euo pipefail

if ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

FAIL=0
IN_GIT_REPO=0
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  IN_GIT_REPO=1
fi
RG_COMMON=(--hidden --no-ignore --glob '!.git/**' --glob '!*.zip' --glob '!*.tar' --glob '!*.tar.gz' --glob '!scripts/verify_anonymization.sh')

check_no_matches() {
  local label="$1"
  shift

  echo
  echo "== ${label} =="

  set +e
  "$@"
  local status=$?
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "FAIL: matches found above."
    FAIL=1
  elif [[ "$status" -eq 1 ]]; then
    echo "OK: no matches."
  else
    echo "ERROR: command failed with status ${status}."
    FAIL=1
  fi
}

IDENTITY_PATTERN='talha|chafekar|talha1503|talha15032|northwestern|zhejiang|dongqianyu|zihanwang|lixinyuan|mindorigin150|jinhui|junqiu|fangjing|zhijie|gaoning|yejinhui|jye624|hkust|fudan|sust|pku\.edu|gmail\.com|outlook\.com|163\.com'

PATH_PATTERN='(/Users/|/home/[A-Za-z0-9_-]+|/mnt/petrelfs/|/gpfs/|/mnt/data/[A-Za-z0-9_-]+)'

PUBLIC_LINK_PATTERN='talha1503/|talha15032/|latency-sensitive-bench|anonymous-latency-bench|anonymous-iclr|starvla\.github\.io|github\.com/starVLA/starVLA|repo=starVLA/starVLA|huggingface\.co/(collections/StarVLA|datasets/StarVLA|StarVLA)|huggingface\.co/[^[:space:]\)\"<>]*StarVLA|Simplicissimus-S|arxiv\.org/abs/2604'

EMAIL_PATTERN='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'

check_no_matches \
  "Known identity strings" \
  rg -n -i "${RG_COMMON[@]}" -e "$IDENTITY_PATTERN" .

check_no_matches \
  "Personal/local absolute paths" \
  rg -n "${RG_COMMON[@]}" -e "$PATH_PATTERN" .

check_no_matches \
  "Public repo/model/citation identifiers" \
  rg -n "${RG_COMMON[@]}" -e "$PUBLIC_LINK_PATTERN" .

echo
echo "== Email-like strings =="
set +e
EMAIL_HITS="$(rg -n "${RG_COMMON[@]}" -e "$EMAIL_PATTERN" . | grep -v 'git@github.com')"
email_status=$?
set -e
if [[ "$email_status" -eq 0 ]]; then
  echo "$EMAIL_HITS"
  echo "FAIL: email-like matches found above."
  FAIL=1
elif [[ "$email_status" -eq 1 ]]; then
  echo "OK: no email-like matches other than generic git@github.com SSH rewrites."
else
  echo "ERROR: email check failed with status ${email_status}."
  FAIL=1
fi

echo
echo "== Git whitespace check =="
if [[ "$IN_GIT_REPO" -eq 0 ]]; then
  echo "OK: skipped outside a git repository."
elif git diff --check; then
  echo "OK: git diff --check passed."
else
  echo "FAIL: git diff --check found issues."
  FAIL=1
fi

echo
echo "== Summary =="
if [[ "$FAIL" -eq 0 ]]; then
  echo "Anonymization verification passed."
else
  echo "Anonymization verification failed. Review the matches above."
fi

exit "$FAIL"
