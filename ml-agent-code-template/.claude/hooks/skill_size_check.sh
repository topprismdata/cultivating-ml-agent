#!/bin/bash
# PostToolUse Hook — Skill Size Validator (500-line rule)
# Triggered when any SKILL.md file is edited/created.
# If file exceeds 500 lines, warn and suggest splitting into resources/.
# Exit 0 = allow, output shown to user (warning, not blocker)

set -e

INPUT=$(cat)

# Get file path
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tool = d.get('tool', d.get('tool_name', ''))
    inp = d.get('tool_input', d.get('input', {}))
    if 'Write' in tool or 'Edit' in tool or 'MultiEdit' in tool:
        print(inp.get('file_path', inp.get('path', '')))
    else:
        print('')
except:
    print('')
" 2>/dev/null)

# Only run on SKILL.md files
if [ -z "$FILE_PATH" ] || ! echo "$FILE_PATH" | grep -qE 'SKILL\.md$'; then
  exit 0
fi

# Check file exists and is readable
[ ! -f "$FILE_PATH" ] && exit 0

LINE_COUNT=$(wc -l < "$FILE_PATH" | tr -d ' ')

# Threshold: 500 lines
if [ "$LINE_COUNT" -gt 500 ]; then
  echo "" >&2
  echo "═══ 500-line rule violation: ${FILE_PATH} (${LINE_COUNT} lines) ═══" >&2
  echo "" >&2
  echo "Skills should be <500 lines for progressive disclosure." >&2
  echo "Split this skill into:" >&2
  echo "  ${FILE_PATH%.md}/" >&2
  echo "    SKILL.md           (main, <500 lines)" >&2
  echo "    resources/" >&2
  echo "      topic-1.md      (<500 lines each)" >&2
  echo "      topic-2.md" >&2
  echo "      topic-3.md" >&2
  echo "" >&2
  echo "Claude loads SKILL.md first, resources only when needed." >&2
  echo "" >&2
  exit 2
fi

# Soft warning at 80% of limit
if [ "$LINE_COUNT" -gt 400 ]; then
  echo "⚠️  ${FILE_PATH} is ${LINE_COUNT}/500 lines. Consider splitting soon." >&2
fi

exit 0