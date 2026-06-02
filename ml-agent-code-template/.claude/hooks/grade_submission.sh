#!/bin/bash
# MLE-Bench Submission Grader
# Validates submission format, runs mlebench grade, reports results.
#
# Usage:
#   bash .claude/hooks/grade_submission.sh <competition>
#
# Returns:
#   0 = grade PASS
#   1 = grade FAIL
#   2 = file/format issue
#   3 = mlebench not installed

set -e

COMPETITION="${1:-}"
if [ -z "$COMPETITION" ]; then
  echo "Usage: $0 <competition-slug>"
  echo "Example: $0 tps-may-2022"
  exit 2
fi

# Resolve submission path
SUBMISSION="submissions/${COMPETITION}/submission.csv"
[ ! -f "$SUBMISSION" ] && SUBMISSION="submissions/${COMPETITION}/submission_chaii_v32_lgb.csv"

if [ ! -f "$SUBMISSION" ]; then
  echo "ERROR: Submission not found at $SUBMISSION"
  echo "Run the builder first to produce submission.csv"
  exit 2
fi

# Validate format
echo "=== Submission validation ==="
echo "File: $SUBMISSION"
echo "Size: $(ls -lh "$SUBMISSION" | awk '{print $5}')"
echo "Lines: $(wc -l < "$SUBMISSION" | tr -d ' ')"
echo "First 3 lines:"
head -3 "$SUBMISSION"
echo ""
echo "Last 2 lines:"
tail -2 "$SUBMISSION"
echo ""

# Check for common issues
LINES=$(wc -l < "$SUBMISSION" | tr -d ' ')
if [ "$LINES" -lt 2 ]; then
  echo "ERROR: Submission has < 2 lines (header + at least 1 row required)"
  exit 2
fi

# Check header
HEADER=$(head -1 "$SUBMISSION")
if [ -z "$HEADER" ]; then
  echo "ERROR: Empty header line"
  exit 2
fi
echo "Header: $HEADER"
echo ""

# Check for empty cells
EMPTY_COUNT=$(awk -F',' 'NR>1 {for(i=1;i<=NF;i++) if($i=="") print NR":"i}' "$SUBMISSION" | wc -l | tr -d ' ')
if [ "$EMPTY_COUNT" -gt 0 ]; then
  echo "WARNING: $EMPTY_COUNT empty cells found in submission"
fi
echo ""

# Check if mlebench is available
echo "=== Running mlebench grade-sample ==="
if ! command -v mlebench >/dev/null 2>&1 && ! python3 -c "import mlebench" 2>/dev/null; then
  echo "WARNING: mlebench not installed. Skipping grade step."
  echo "Install with: pip install mle-bench"
  exit 3
fi

# Run grade-sample (takes positional args: submission, competition_id)
if command -v mlebench >/dev/null 2>&1; then
  GRADE_CMD="mlebench grade-sample"
else
  GRADE_CMD="python3 -m mlebench.grade_sample"
fi

echo "Command: $GRADE_CMD $SUBMISSION $COMPETITION"
echo ""

if $GRADE_CMD "$SUBMISSION" "$COMPETITION" 2>&1; then
  echo ""
  echo "=== GRADE PASS ==="
  exit 0
else
  GRADE_EXIT=$?
  echo ""
  echo "=== GRADE FAIL (exit $GRADE_EXIT) ==="
  exit 1
fi