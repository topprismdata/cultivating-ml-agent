#!/bin/bash
# Git Risk Detector Hook
# Blocks destructive git and shell operations before they execute.
# Exit 0 = allow, Exit 2 = block + show reason

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('command','')[:1000])" 2>/dev/null)

if [ -z "$COMMAND" ]; then
    echo "OK" >&2
    exit 0
fi

# Destructive patterns — block immediately
DESTRUCTIVE_PATTERNS="rm -rf|rm -fr|rm -- -f|/dev/sda|/dev/nvme0|/dev/nvme1|mkfs|dd if=/dev/zero"

# Git force/dangerous patterns
GIT_FORCE_PATTERNS="git push --force|git push -f|git push +f|git reset --hard|git reset --mixed|git clean -fdx|git checkout --orphan"

# Combined pattern
if echo "$COMMAND" | grep -Ei "$DESTRUCTIVE_PATTERNS|$GIT_FORCE_PATTERNS" > /dev/null 2>&1; then
    echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Destructive command blocked by git-risk hook. If this is intentional, run the command directly outside Claude Code."}}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['hookSpecificOutput']['permissionDecisionReason'])" >&2
    exit 2
fi

# Warning patterns — allow but log
WARN_PATTERNS="git rebase|git reflog|git stash drop"
if echo "$COMMAND" | grep -Ei "$WARN_PATTERNS" > /dev/null 2>&1; then
    echo "[git-risk] Caution: $COMMAND" >&2
fi

echo "OK" >&2
exit 0