#!/bin/bash
# Meta-Optimize — Memory Health Analyzer
# Scans memory/ for: staleness, contradictions, missing entries, large files.
# Output: human-readable report with suggested actions.
# Read-only — to apply changes, use /meta-apply (which goes through cross-model jury).
#
# Usage:
#   bash meta_optimize.sh
#   bash meta_optimize.sh --json
#   bash meta_optimize.sh --memory-dir=path/to/memory
#
# Output sections:
#   1. Staleness — files not modified in N days
#   2. Size warnings — files > 500 lines (500-line rule)
#   3. Missing index entries — files not in MEMORY.md
#   4. Contradiction hints — principles with overlapping topics
#   5. Coverage gaps — experiment topics not extracted as skills
#   6. Dead-end staleness — feedback_no_recheck entries > 1 year old
#   7. Cross-link health — broken internal links

set -e

MODE="text"
MEMORY_DIR="memory"

for arg in "$@"; do
  case "$arg" in
    --json) MODE="json" ;;
    --memory-dir=*) MEMORY_DIR="${arg#*=}" ;;
    --help|-h)
      echo "Usage: $0 [--json] [--memory-dir=PATH]"
      echo ""
      echo "Analyzes memory/ and reports:"
      echo "  - Staleness (> 90 days untouched)"
      echo "  - Oversized files (> 500 lines)"
      echo "  - Files not in MEMORY.md"
      echo "  - Possible contradictions"
      echo "  - Coverage gaps"
      exit 0
      ;;
  esac
done

# Resolve to absolute path
MEMORY_DIR=$(cd "$MEMORY_DIR" 2>/dev/null && pwd || echo "$MEMORY_DIR")
[ ! -d "$MEMORY_DIR" ] && {
  echo "ERROR: $MEMORY_DIR is not a directory"
  exit 1
}

# Output buffer
if [ "$MODE" = "json" ]; then
  echo "{"
fi

print_section() {
  local title="$1"
  if [ "$MODE" = "json" ]; then
    echo "  \"$(echo "$title" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')\": ["
  else
    echo ""
    echo "═══ $title ═══"
  fi
}

print_item() {
  local item="$1"
  if [ "$MODE" = "json" ]; then
    echo "    $item,"
  else
    echo "  $item"
  fi
}

print_section_end() {
  if [ "$MODE" = "json" ]; then
    echo "  ],"
  fi
}

# ─── 1. Staleness (> 90 days untouched) ──────────────────────────────────────
print_section "Staleness (> 90 days untouched)"
STALE_COUNT=0
while IFS= read -r -d '' FILE; do
  DAYS=$(( ($(date +%s) - $(stat -f %m "$FILE")) / 86400 ))
  if [ "$DAYS" -gt 90 ]; then
    STALE_COUNT=$((STALE_COUNT+1))
    REL=${FILE#$(pwd)/}
    if [ "$MODE" = "json" ]; then
      print_item "{\"file\": \"$REL\", \"days\": $DAYS}"
    else
      print_item "$REL — $DAYS days"
    fi
  fi
done < <(find "$MEMORY_DIR" -type f \( -name "*.md" \) -print0 2>/dev/null)

[ "$STALE_COUNT" -eq 0 ] && {
  if [ "$MODE" = "text" ]; then
    print_item "(none)"
  fi
}
print_section_end

# ─── 2. Size warnings (> 500 lines, 500-line rule) ──────────────────────────
print_section "Oversized Files (> 500 lines)"
OVERSIZE_COUNT=0
while IFS= read -r FILE; do
  LINES=$(wc -l < "$FILE" | tr -d ' ')
  if [ "$LINES" -gt 500 ]; then
    OVERSIZE_COUNT=$((OVERSIZE_COUNT+1))
    REL=${FILE#$(pwd)/}
    if [ "$MODE" = "json" ]; then
      print_item "{\"file\": \"$REL\", \"lines\": $LINES}"
    else
      print_item "$REL — $LINES lines (split into resources/)"
    fi
  fi
done < <(find "$MEMORY_DIR" -type f -name "*.md" 2>/dev/null)

[ "$OVERSIZE_COUNT" -eq 0 ] && {
  if [ "$MODE" = "text" ]; then
    print_item "(none)"
  fi
}
print_section_end

# ─── 3. Missing index entries (not in MEMORY.md) ────────────────────────────
print_section "Files Not in MEMORY.md"
MISSING_COUNT=0
if [ -f "$MEMORY_DIR/MEMORY.md" ]; then
  while IFS= read -r FILE; do
    REL=${FILE#$(pwd)/}
    REL=${REL#./}
    BASENAME=$(basename "$REL" .md)
    [ "$BASENAME" = "MEMORY" ] && continue
    [ "$BASENAME" = "TEMPLATE" ] && continue
    [ "$BASENAME" = "exp_template" ] && continue
    [ "$BASENAME" = "competition_template" ] && continue
    [ "$BASENAME" = "feedback_no_recheck_confirmed_dead" ] && continue

    # Check if mentioned in MEMORY.md
    if ! grep -q "$BASENAME" "$MEMORY_DIR/MEMORY.md" 2>/dev/null; then
      MISSING_COUNT=$((MISSING_COUNT+1))
      if [ "$MODE" = "json" ]; then
        print_item "{\"file\": \"$REL\"}"
      else
        print_item "$REL"
      fi
    fi
  done < <(find "$MEMORY_DIR" -type f -name "*.md" 2>/dev/null)
fi

[ "$MISSING_COUNT" -eq 0 ] && {
  if [ "$MODE" = "text" ]; then
    print_item "(all files indexed)"
  fi
}
print_section_end

# ─── 4. Contradiction hints ──────────────────────────────────────────────────
print_section "Possible Contradictions"
CONTRADICTIONS=0
if [ -d "$MEMORY_DIR/principles" ]; then
  # Look for opposing language in principles
  for f in "$MEMORY_DIR/principles"/*.md; do
    [ ! -f "$f" ] && continue
    if grep -qiE "never|don'?t|forbidden" "$f" 2>/dev/null; then
      REL=${f#$(pwd)/}
      if [ "$MODE" = "json" ]; then
        print_item "{\"file\": \"$REL\", \"hint\": \"contains never/don't/forbidden — verify not contradicted by newer principle\"}"
      else
        print_item "$REL — review for potential contradiction"
      fi
      CONTRADICTIONS=$((CONTRADICTIONS+1))
    fi
  done
fi

[ "$CONTRADICTIONS" -eq 0 ] && {
  if [ "$MODE" = "text" ]; then
    print_item "(no obvious contradictions)"
  fi
}
print_section_end

# ─── 5. Coverage gaps (experiment topics not extracted as skills) ───────────
print_section "Coverage Gaps (experiment topics without extracted skills)"
GAPS=0
if [ -d "$MEMORY_DIR/experiments" ]; then
  for exp in "$MEMORY_DIR/experiments"/*.md; do
    [ ! -f "$exp" ] && continue
    BASENAME=$(basename "$exp" .md)
    [ "$BASENAME" = "exp_template" ] && continue
    # Has this been extracted to a skill?
    if [ -d "$MEMORY_DIR/skills" ]; then
      HIT=0
      for skill in "$MEMORY_DIR/skills"/*.md; do
        [ ! -f "$skill" ] && continue
        # Check if experiment name appears in skill content
        if grep -qi "$BASENAME" "$skill" 2>/dev/null; then
          HIT=1
          break
        fi
      done
      if [ "$HIT" -eq 0 ]; then
        # Check if experiment was a "Breakthrough" outcome (only if checkbox [x] is checked, not just word mention)
        if grep -qiE "^\s*-?\s*\[x\]\s*\*?\*?breakthrough" "$exp" 2>/dev/null; then
          GAPS=$((GAPS+1))
          if [ "$MODE" = "json" ]; then
            print_item "{\"experiment\": \"$BASENAME\", \"action\": \"extract skill\"}"
          else
            print_item "$BASENAME — Breakthrough but no skill extracted"
          fi
        fi
      fi
    fi
  done
fi

[ "$GAPS" -eq 0 ] && {
  if [ "$MODE" = "text" ]; then
    print_item "(no obvious gaps)"
  fi
}
print_section_end

# ─── 6. Dead-end staleness (> 365 days) ──────────────────────────────────────
print_section "Stale Dead-Ends (> 365 days, may be obsolete)"
DEAD_STALE=0
if [ -f "$MEMORY_DIR/feedback_no_recheck_confirmed_dead.md" ]; then
  while IFS= read -r LINE; do
    DATE=$(echo "$LINE" | grep -oE '\b20[0-9]{2}-[0-9]{2}-[0-9]{2}\b' | head -1)
    if [ -n "$DATE" ]; then
      DAYS=$(( ($(date +%s) - $(date -j -f "%Y-%m-%d" "$DATE" +%s 2>/dev/null || echo 0)) / 86400 ))
      if [ "$DAYS" -gt 365 ]; then
        DEAD_STALE=$((DEAD_STALE+1))
        if [ "$MODE" = "json" ]; then
          print_item "{\"date\": \"$DATE\", \"line_preview\": \"$(echo "$LINE" | cut -c1-80)\"}"
        else
          print_item "$DATE — $LINE"
        fi
      fi
    fi
  done < "$MEMORY_DIR/feedback_no_recheck_confirmed_dead.md"
fi

[ "$DEAD_STALE" -eq 0 ] && {
  if [ "$MODE" = "text" ]; then
    print_item "(none)"
  fi
}
print_section_end

# ─── 7. Cross-link health (broken internal links) ───────────────────────────
print_section "Broken Internal Links"
BROKEN=0
if [ -f "$MEMORY_DIR/MEMORY.md" ]; then
  # Find markdown links
  while IFS= read -r -d '' FILE; do
    while IFS= read -r LINK; do
      # Extract target
      TARGET=$(echo "$LINK" | sed -E 's/.*\(([^)]+)\).*/\1/')
      [ -z "$TARGET" ] && continue
      [ "${TARGET:0:1}" = "#" ] && continue  # internal anchor
      [ "${TARGET:0:4}" = "http" ] && continue  # external

      # Resolve relative to file's directory
      FILE_DIR=$(dirname "$FILE")
      RESOLVED="$FILE_DIR/$TARGET"
      # Strip anchors
      RESOLVED="${RESOLVED%#*}"
      if [ ! -f "$RESOLVED" ] && [ ! -d "$RESOLVED" ]; then
        BROKEN=$((BROKEN+1))
        REL_FILE=${FILE#$(pwd)/}
        if [ "$MODE" = "json" ]; then
          print_item "{\"file\": \"$REL_FILE\", \"broken_link\": \"$TARGET\"}"
        else
          print_item "$REL_FILE → $TARGET (not found)"
        fi
      fi
    done < <(grep -oE '\[[^]]*\]\([^)]+\)' "$FILE" 2>/dev/null)
  done < <(find "$MEMORY_DIR" -type f -name "*.md" -print0 2>/dev/null)
fi

[ "$BROKEN" -eq 0 ] && {
  if [ "$MODE" = "text" ]; then
    print_item "(no broken links)"
  fi
}
print_section_end

# ─── Summary ─────────────────────────────────────────────────────────────────
if [ "$MODE" = "text" ]; then
  echo ""
  echo "═══ Summary ═══"
  echo "  Stale (>90d):      $STALE_COUNT"
  echo "  Oversized (>500L): $OVERSIZE_COUNT"
  echo "  Missing index:     $MISSING_COUNT"
  echo "  Contradictions:    $CONTRADICTIONS"
  echo "  Coverage gaps:     $GAPS"
  echo "  Stale dead-ends:   $DEAD_STALE"
  echo "  Broken links:      $BROKEN"
  echo ""
  if [ $((STALE_COUNT + OVERSIZE_COUNT + MISSING_COUNT + CONTRADICTIONS + GAPS + DEAD_STALE + BROKEN)) -eq 0 ]; then
    echo "✓ Memory is healthy. No action required."
  else
    echo "→ To apply changes, use /meta-apply (cross-model review required)"
  fi
elif [ "$MODE" = "json" ]; then
  echo "  \"summary\": {"
  echo "    \"stale\": $STALE_COUNT,"
  echo "    \"oversized\": $OVERSIZE_COUNT,"
  echo "    \"missing_index\": $MISSING_COUNT,"
  echo "    \"contradictions\": $CONTRADICTIONS,"
  echo "    \"coverage_gaps\": $GAPS,"
  echo "    \"stale_dead_ends\": $DEAD_STALE,"
  echo "    \"broken_links\": $BROKEN"
  echo "  }"
  echo "}"
fi