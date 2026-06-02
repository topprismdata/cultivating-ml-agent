#!/bin/bash
# UserPromptSubmit Hook — High-Stakes Cross-Review Trigger
# Detects prompts about submitting/publishing/committing major work.
# Injects reminder to invoke /review first.
# Exit 0 = allow, additionalContext injected for Claude

set -e

INPUT=$(cat)

PROMPT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('prompt', d.get('user_prompt', '')))
except:
    print('')
" 2>/dev/null)

CWD=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('cwd', ''))
except:
    print('')
" 2>/dev/null)
[ -z "$CWD" ] && CWD=$(pwd)

cd "$CWD" || exit 0

# Only act on meaningful prompts
[ -z "$PROMPT" ] && exit 0
[ ${#PROMPT} -lt 10 ] && exit 0

# Detect high-stakes operations
PROMPT_TMP=$(mktemp 2>/dev/null || echo "/tmp/.cross-review-prompt-$$")
printf '%s' "$PROMPT" > "$PROMPT_TMP" 2>/dev/null || {
  PROMPT_TMP="/Users/mac/ml-agent-code-template/.cross-review-prompt-tmp"
  printf '%s' "$PROMPT" > "$PROMPT_TMP"
}
HIGH_STAKES=$(PROMPT_FILE="$PROMPT_TMP" python3 <<'PYEOF'
import os, re, sys

try:
    with open(os.environ['PROMPT_FILE']) as f:
        prompt = f.read().strip()
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(0)

patterns = [
    # Submission-related
    (r'\b(submit|submitting|submission)\b', 'submission'),
    (r'\bmlebench\s+grade\b', 'mlebench grade'),
    (r'\bgrade\s+(this|it|the)\b', 'grading'),
    # Git/release
    (r'\bgit\s+push\b', 'git push'),
    (r'\b(merge|merging)\s+(to|into)\s+(main|master|prod)\b', 'merge to main'),
    (r'\b(release|publish|deploy)\b', 'release'),
    # Final
    (r'\bfinal\s+(version|submission|answer|decision)\b', 'final decision'),
    (r"\b(this\s+is\s+it|that's\s+it|we're\s+done)\b", 'finalization'),
    # Strategic
    (r'\b(architecture|architectural)\s+(decision|choice|change)\b', 'architectural decision'),
    (r'\b(start|begin|launch)\s+(production|live|prod)\b', 'going to production'),
    # NEW: Pre-execution simulation (P0 enhancement)
    (r"\b(let['\u2019]?s|let\s+me|i['\u2019]?m\s+going\s+to|going\s+to)\s+(train|execute|run|build|fit|deploy)\b", 'pre-execution'),
    (r'\b(training|fitting|executing|deploying)\s+(on|with|a|an|the)\s+\w+', 'pre-execution'),
    (r'\b(run|execute|launch)\s+(\w+\s+){0,3}(training|pipeline|experiment|simulation|script)\b', 'pre-execution'),
]

# Ignore patterns (questions, hypotheticals)
ignore_patterns = [
    r'^\s*how\s+(do|can|should|would)\b',
    r'^\s*what\s+(is|are|if)\b',
    r'^\s*can\s+you\b',
    r'^\s*would\b',
    r'^\s*should\s+we\b',
    r"^\s*let'?s\s+(see|look|check|think|consider)\b",
    r'^\s*explain\b',
    r'^\s*why\b',
    r'\?$',  # Ends with question mark
]

# Check ignore first
for pat in ignore_patterns:
    if re.search(pat, prompt, re.IGNORECASE):
        sys.exit(0)

# Check high-stakes
for pat, label in patterns:
    if re.search(pat, prompt, re.IGNORECASE):
        print(label)
        sys.exit(0)
PYEOF
)
rm -f "$PROMPT_TMP" 2>/dev/null

if [ -z "$HIGH_STAKES" ]; then
  exit 0
fi

# Get available backend
BACKEND=""
for cmd in /Users/mac/.local/bin/agy agy gemini codex ollama; do
  if [ -x "$cmd" ] || command -v "$cmd" >/dev/null 2>&1; then
    BACKEND=$(basename "$cmd")
    break
  fi
done

if [ -z "$BACKEND" ]; then
  CONTEXT="⚠️  HIGH-STAKES OPERATION DETECTED: $HIGH_STAKES. No external LLM CLI available — please do adversarial self-check (5 questions in codex-review skill) before proceeding."
else
  CONTEXT="⚠️  HIGH-STAKES OPERATION DETECTED: $HIGH_STAKES. Strongly recommend invoking /review first to get an independent second opinion from $BACKEND. Two models agreeing is more reliable than one."
fi

ESCAPED=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read().rstrip()))" <<< "$CONTEXT")

cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": ${ESCAPED}
  }
}
EOF

exit 0