#!/bin/bash
# SessionStart Hook — Load active dev docs into context
# Reads dev/active/*-plan.md, dev/active/*-context.md, dev/active/*-tasks.md
# and injects them as additionalContext for Claude.
# Exit 0 = allow, additionalContext injected for Claude

set -e

INPUT=$(cat)

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

# Find dev/active directory
DEV_DIR=""
for d in "dev/active" ".claude/dev/active"; do
  if [ -d "$d" ]; then
    DEV_DIR="$d"
    break
  fi
done

[ -z "$DEV_DIR" ] && exit 0

# Check for active triples (must have plan + context + tasks)
CONTEXT_TEXT="📂 Active dev docs found:\n"

# Group files by slug
SLUGS=$(ls "$DEV_DIR"/*-plan.md 2>/dev/null | sed -E "s|.*/(.*)-plan\.md|\1|" | sort -u | grep -v '^TEMPLATE$')

if [ -z "$SLUGS" ]; then
  exit 0
fi

COUNT=0
for SLUG in $SLUGS; do
  PLAN="$DEV_DIR/$SLUG-plan.md"
  CTX="$DEV_DIR/$SLUG-context.md"
  TASKS="$DEV_DIR/$SLUG-tasks.md"

  # Only include if all three exist
  if [ ! -f "$PLAN" ] || [ ! -f "$CTX" ] || [ ! -f "$TASKS" ]; then
    continue
  fi

  # Check mtime — only show recent (modified in last 7 days)
  if [ "$(find "$PLAN" -mtime +7 2>/dev/null)" ]; then
    continue
  fi

  COUNT=$((COUNT + 1))
  CONTEXT_TEXT="${CONTEXT_TEXT}\n## Active: $SLUG\n"

  # Get tasks summary
  PENDING=$(grep -c "^- \[ \]" "$TASKS" 2>/dev/null || echo "0")
  DONE=$(grep -c "^- \[x\]" "$TASKS" 2>/dev/null || echo "0")
  BLOCKED=$(grep -c "^- \[!\]" "$TASKS" 2>/dev/null || echo "0")
  CONTEXT_TEXT="${CONTEXT_TEXT}- Tasks: ${DONE} done, ${PENDING} pending, ${BLOCKED} blocked\n"
  CONTEXT_TEXT="${CONTEXT_TEXT}- Plan: ${PLAN#./}\n"
  CONTEXT_TEXT="${CONTEXT_TEXT}- Context: ${CTX#./}\n"
done

if [ "$COUNT" -eq 0 ]; then
  exit 0
fi

# Format as additionalContext
ESCAPED=$(python3 -c "
import json, sys
text = sys.stdin.read()
# Truncate to 800 chars to avoid context bloat
print(json.dumps(text[:800]))
" <<< "$CONTEXT_TEXT")

cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": ${ESCAPED}
  }
}
EOF

exit 0