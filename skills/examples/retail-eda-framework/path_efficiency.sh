#!/bin/bash
# Path Efficiency Analyzer
# Reads a session log (or stdin) and computes the 4 efficiency metrics.
# Per memory/skills/path-efficiency-analysis.md
#
# Usage:
#   bash path_efficiency.sh <log-file>
#   cat session.log | bash path_efficiency.sh
#
# The script analyzes the log heuristically — looking for:
#   - Repeated commands (3+ occurrences = repetitiveness)
#   - Dead-end patterns (commands that fail or produce no output)
#   - Tool call distribution
#   - Time on failed vs successful paths

set -e

LOG_FILE="${1:-}"
if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
  # Try stdin
  LOG_CONTENT=$(cat)
  LOG_FILE="<stdin>"
else
  LOG_CONTENT=$(cat "$LOG_FILE")
fi

if [ -z "$LOG_CONTENT" ]; then
  echo "ERROR: No log content provided"
  echo "Usage: $0 <log-file> or cat log | $0"
  exit 1
fi

echo "=== Path Efficiency Report ==="
echo "Source: $LOG_FILE"
echo "Size: $(echo "$LOG_CONTENT" | wc -l) lines, $(echo "$LOG_CONTENT" | wc -c) chars"
echo ""

# Basic stats
TOTAL_LINES=$(echo "$LOG_CONTENT" | wc -l | tr -d ' ')
TOTAL_CHARS=$(echo "$LOG_CONTENT" | wc -c | tr -d ' ')
echo "Total actions (line count): $TOTAL_LINES"
echo ""

# 1. Repetitiveness Rate
echo "1. Repetitiveness Rate"
echo "   (looking for repeated commands/operations)"
echo ""

# Extract "command-like" lines (lines starting with $, >, or containing "Tool:" etc.)
COMMANDS=$(echo "$LOG_CONTENT" | grep -E "^\s*(\$ |>|Tool:|Running|Executing)" | sed -E 's/^[\s$>]+\s*//' | head -200)
UNIQUE_COMMANDS=$(echo "$COMMANDS" | sort -u | wc -l | tr -d ' ')
TOTAL_COMMANDS=$(echo "$COMMANDS" | wc -l | tr -d ' ')

if [ "$TOTAL_COMMANDS" -gt 0 ]; then
  REPETITIVENESS_PCT=$(( 100 * (TOTAL_COMMANDS - UNIQUE_COMMANDS) / TOTAL_COMMANDS ))
  echo "   Total command-like lines: $TOTAL_COMMANDS"
  echo "   Unique: $UNIQUE_COMMANDS"
  echo "   Repetitiveness Rate: ${REPETITIVENESS_PCT}% (target: < 10%)"

  # Show top repeated
  echo ""
  echo "   Top repeated commands (potential stuck patterns):"
  echo "$COMMANDS" | sort | uniq -c | sort -rn | head -5 | while read count cmd; do
    if [ "$count" -gt 2 ]; then
      echo "     ${count}× $(echo "$cmd" | head -c 60)"
    fi
  done
else
  echo "   No command-like patterns found"
fi
echo ""

# 2. Error/Recovery detection
echo "2. Error Recovery Detection"
echo "   (looking for 'Error', 'Failed', 'panic', 'Traceback')"
echo ""

ERROR_LINES=$(echo "$LOG_CONTENT" | grep -ciE "error|failed|panic|traceback|exception" || echo "0")
echo "   Total error mentions: $ERROR_LINES"
if [ "$ERROR_LINES" -gt 0 ]; then
  echo "   (Recovery Rate = how many errors were followed by correction; requires manual review)"
fi
echo ""

# 3. Tool Productivity
echo "3. Tool Productivity (heuristic)"
echo ""

# Count tool calls
TOOL_CALLS=$(echo "$LOG_CONTENT" | grep -cE "^\s*(\$ |>|Tool:|Running)" || echo "0")
SUCCESS_LINES=$(echo "$LOG_CONTENT" | grep -ciE "success|saved|generated|completed|created" || echo "0")
ERROR_COUNT=$(echo "$LOG_CONTENT" | grep -ciE "error|failed" || echo "0")
echo "   Tool calls (heuristic): $TOOL_CALLS"
echo "   Success indicators:    $SUCCESS_LINES"
echo "   Error indicators:     $ERROR_COUNT"
if [ "$TOOL_CALLS" -gt 0 ]; then
  PRODUCTIVITY_PCT=$(( 100 * SUCCESS_LINES / (SUCCESS_LINES + ERROR_COUNT + 1) ))
  echo "   Productivity: ~${PRODUCTIVITY_PCT}% (target: > 80%)"
fi
echo ""

# 4. Costliest Wrong Path (heuristic: longest run of error mentions)
echo "4. Costliest Wrong Path (heuristic)"
echo "   (longest sequence of errors before pivot)"
echo ""

# Find longest error sequence
LONGEST_ERROR_RUN=0
CURRENT_RUN=0
while IFS= read -r line; do
  if echo "$line" | grep -qiE "error|failed|panic|traceback"; then
    CURRENT_RUN=$(( CURRENT_RUN + 1 ))
    if [ "$CURRENT_RUN" -gt "$LONGEST_ERROR_RUN" ]; then
      LONGEST_ERROR_RUN=$CURRENT_RUN
    fi
  else
    CURRENT_RUN=0
  fi
done <<< "$LOG_CONTENT"

echo "   Longest error run: $LONGEST_ERROR_RUN lines"
echo ""

# Recommendations
echo "=== Recommendations ==="
if [ "$REPETITIVENESS_PCT" -gt 20 ] 2>/dev/null; then
  echo "  ⚠️  Repetitiveness high — add dead-end check at start of new direction"
fi
if [ "$ERROR_LINES" -gt 10 ]; then
  echo "  ⚠️  Many errors — review /claudeception for learning opportunities"
fi
if [ "$TOOL_CALLS" -lt 5 ]; then
  echo "  ⚠️  Few tool calls — session may be too short for meaningful analysis"
fi

echo "  → To improve: run self-critique-checkpoint before each new direction"
echo "  → Reference: memory/skills/path-efficiency-analysis.md"