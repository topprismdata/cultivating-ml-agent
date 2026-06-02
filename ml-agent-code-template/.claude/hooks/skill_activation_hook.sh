#!/bin/bash
# UserPromptSubmit Hook — Skill Auto-Activation
# Reads .claude/skill-rules.json and suggests relevant skills based on prompt patterns.
# Also reminds to read memory/MEMORY.md on first prompt of session.
# Exit 0 = allow, additionalContext injected for Claude

set -e

INPUT=$(cat)

# Extract prompt text
PROMPT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('prompt', d.get('user_prompt', '')))
except:
    print('')
" 2>/dev/null)

# Get CWD
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

# Find skill-rules.json
RULES_FILE=""
for f in ".claude/skill-rules.json" "${HOME}/.claude/skill-rules.json"; do
  if [ -f "$f" ]; then
    RULES_FILE="$f"
    break
  fi
done

# Find session marker (24h window for "first prompt of session")
SESSION_MARKER="${HOME}/.claude/.session-marker"
FIRST_PROMPT=false
if [ ! -f "$SESSION_MARKER" ]; then
  FIRST_PROMPT=true
  mkdir -p "$(dirname "$SESSION_MARKER")" 2>/dev/null
  touch "$SESSION_MARKER"
elif [ "$(find "$SESSION_MARKER" -mmin +1440 2>/dev/null)" ]; then
  # Marker older than 24h — treat as new session
  FIRST_PROMPT=true
  touch "$SESSION_MARKER"
fi

# === Build context ===
CONTEXT=""

# First-prompt check
if [ "$FIRST_PROMPT" = "true" ] && [ -f "memory/MEMORY.md" ]; then
  CONTEXT="${CONTEXT}📋 First prompt of session. Read memory/MEMORY.md for context. "
fi

# Dead-end check (always suggest for non-trivial prompts)
if [ -n "$PROMPT" ] && [ ${#PROMPT} -gt 20 ]; then
  CONTEXT="${CONTEXT}⛔ Before trying any new direction: check memory/feedback_no_recheck_confirmed_dead.md. "
fi

# Skill-rules matching
if [ -n "$RULES_FILE" ] && [ -n "$PROMPT" ] && [ ${#PROMPT} -gt 5 ]; then
  # Write prompt to a temp file to avoid heredoc/script-body confusion
  PROMPT_TMP=$(mktemp)
  printf '%s' "$PROMPT" > "$PROMPT_TMP"
  MATCHES=$(PROMPT_FILE="$PROMPT_TMP" RULES_FILE="$RULES_FILE" python3 <<'PYEOF'
import json, os, re

rules_file = os.environ['RULES_FILE']
with open(rules_file) as f:
    config = json.load(f)

with open(os.environ['PROMPT_FILE']) as f:
    prompt = f.read().strip()

# Check ignore patterns
for pat in config.get('ignore_patterns', []):
    try:
        if re.match(pat, prompt, re.IGNORECASE):
            import sys; sys.exit(0)
    except re.error:
        continue

# Match rules
matched = []
for rule in config.get('rules', []):
    for pat in rule.get('patterns', []):
        try:
            if re.search(pat, prompt, re.IGNORECASE):
                matched.append(rule)
                break
        except re.error:
            continue

# Sort by priority (highest first)
matched.sort(key=lambda r: r.get('priority', 0), reverse=True)

# De-dupe by skill name
seen = set()
unique = []
for r in matched:
    s = r.get('skill', '')
    if s and s not in seen:
        seen.add(s)
        unique.append(r)

# Output top 3
for r in unique[:3]:
    skill = r.get('skill', '')
    reason = r.get('reason', '')
    print(f"💡 Activate skill: {skill} — {reason}")
PYEOF
)
  rm -f "$PROMPT_TMP"
  if [ -n "$MATCHES" ]; then
    CONTEXT="${CONTEXT}${MATCHES} "
  fi
fi

# Output
if [ -n "$CONTEXT" ]; then
  # Truncate to avoid bloat
  CONTEXT_TRIMMED=$(echo "$CONTEXT" | head -c 1500)
  ESCAPED=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read().rstrip()))" <<< "$CONTEXT_TRIMMED")
  cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": ${ESCAPED}
  }
}
EOF
fi

exit 0