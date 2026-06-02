#!/bin/bash
# Retail EDA Pipeline — 5-stage implementation
# Used by ml-agent-code-template's /eda command
# See: skills/examples/retail-eda-framework/SKILL.md
#
# Usage:
#   bash eda_pipeline.sh <data-path> [options]
#   bash eda_pipeline.sh data/train.csv
#   bash eda_pipeline.sh data/train.csv --segment=customer_type
#   bash eda_pipeline.sh data/train.csv --time-series=date_col

set -e

DATA_PATH="${1:-data/train.csv}"
SEGMENT=""
TIME_SERIES=""

# Parse args
for arg in "$@"; do
  case "$arg" in
    --segment=*) SEGMENT="${arg#*=}" ;;
    --time-series=*) TIME_SERIES="${arg#*=}" ;;
  esac
done

if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: $DATA_PATH not found"
  echo "Usage: $0 <data.csv> [--segment=col] [--time-series=col]"
  exit 1
fi

mkdir -p eda

echo "=== EDA Pipeline ==="
echo "  Data: $DATA_PATH"
echo "  Segment: ${SEGMENT:-<none>}"
echo "  Time-series: ${TIME_SERIES:-<none>}"
echo ""

# ── Stage 1: Data Quality ──
echo "=== Stage 1: Data Quality Audit ==="
if python3 -c "import ydata_profiling" 2>/dev/null; then
  python3 - <<PYEOF
import ydata_profiling, pandas as pd
df = pd.read_csv("$DATA_PATH")
profile = ydata_profiling.ProfileReport(df, title="Data Quality Report", minimal=True)
profile.to_file("eda/data_quality.html")
print(f"  ✓ Generated eda/data_quality.html ({df.shape[0]:,} rows × {df.shape[1]} cols)")
PYEOF
else
  echo "  ⚠️ ydata-profiling not installed, skipping (pip install ydata-profiling)"
fi

if python3 -c "import missingno" 2>/dev/null; then
  python3 - <<PYEOF
import missingno as msno, pandas as pd, matplotlib
matplotlib.use('Agg')
df = pd.read_csv("$DATA_PATH")
msno.matrix(df).figure.savefig("eda/missing_matrix.png", bbox_inches='tight')
msno.heatmap(df).figure.savefig("eda/missing_heatmap.png", bbox_inches='tight')
msno.dendrogram(df).figure.savefig("eda/missing_dendrogram.png", bbox_inches='tight')
print("  ✓ Generated missing_matrix.png, missing_heatmap.png, missing_dendrogram.png")
PYEOF
else
  echo "  ⚠️ missingno not installed, skipping (pip install missingno)"
fi
echo ""

# ── Stage 2: Train vs Test ──
echo "=== Stage 2: Train vs Test Comparison ==="
TEST_PATH="${DATA_PATH//train/test}"
if [ -f "$TEST_PATH" ] && python3 -c "import sweetviz" 2>/dev/null; then
  python3 - <<PYEOF
import sweetviz as sv, pandas as pd
train = pd.read_csv("$DATA_PATH")
test = pd.read_csv("$TEST_PATH")
target = 'target' if 'target' in train.columns else None
report = sv.compare([train, "Train"], [test, "Test"], target_feat=target)
report.show_html("eda/train_vs_test.html")
print("  ✓ Generated eda/train_vs_test.html")
PYEOF
elif [ ! -f "$TEST_PATH" ]; then
  echo "  ⚠️ No test set at $TEST_PATH, skipping"
else
  echo "  ⚠️ sweetviz not installed, skipping (pip install sweetviz)"
fi
echo ""

# ── Stage 3: Domain-Specific ──
echo "=== Stage 3: Domain-Specific EDA ==="
if [ -n "$SEGMENT" ]; then
  if python3 -c "import ydata_profiling" 2>/dev/null; then
    python3 - <<PYEOF
import ydata_profiling, pandas as pd
df = pd.read_csv("$DATA_PATH")
for seg in df["$SEGMENT"].dropna().unique()[:10]:  # cap at 10 segments
    subset = df[df["$SEGMENT"] == seg]
    if len(subset) < 100: continue
    profile = ydata_profiling.ProfileReport(subset, minimal=True, title=f"Segment: {seg}")
    profile.to_file(f"eda/segment_{seg}.html")
    print(f"  ✓ Segment '{seg}': {len(subset):,} rows → eda/segment_{seg}.html")
PYEOF
  else
    echo "  ⚠️ ydata-profiling not installed, skipping"
  fi
else
  echo "  (--segment not specified, skipping per-segment EDA)"
fi
echo ""

# ── Stage 4: Time-Series ──
if [ -n "$TIME_SERIES" ]; then
  echo "=== Stage 4: Time-Series EDA ==="
  if python3 -c "import tslumen" 2>/dev/null; then
    python3 - <<PYEOF
import tslumen, pandas as pd, matplotlib
matplotlib.use('Agg')
df = pd.read_csv("$DATA_PATH", parse_dates=["$TIME_SERIES"])
tslumen.from_ts(df.set_index("$TIME_SERIES").select_dtypes(include='number').iloc[:, 0]).plot()
import matplotlib.pyplot as plt
plt.savefig("eda/time_series_overview.png", bbox_inches='tight')
print("  ✓ Generated eda/time_series_overview.png")
PYEOF
  else
    echo "  ⚠️ tslumen not installed, falling back to basic statsmodels"
    python3 - <<PYEOF
import pandas as pd, matplotlib
matplotlib.use('Agg')
df = pd.read_csv("$DATA_PATH", parse_dates=["$TIME_SERIES"])
df.set_index("$TIME_SERIES").plot(subplots=True, figsize=(12, 8))
import matplotlib.pyplot as plt
plt.savefig("eda/time_series_overview.png", bbox_inches='tight')
print("  ✓ Generated eda/time_series_overview.png (basic)")
PYEOF
  fi
  echo ""
fi

# ── Stage 5: Summary ──
echo "=== Stage 5: EDA Summary ==="
SUMMARY_FILE="eda/EDA_SUMMARY.md"
cat > "$SUMMARY_FILE" <<EOL
# EDA Summary

Generated: $(date '+%Y-%m-%d %H:%M:%S')
Data: \`$DATA_PATH\`
Rows: $(wc -l < "$DATA_PATH" | tr -d ' ')

## Files Generated
$(ls -la eda/ 2>/dev/null | tail -n +2 | awk '{print "- \`" $NF "\`"}')

## Manual Review Required
- [ ] Open eda/data_quality.html
- [ ] Check missing pattern (heatmap)
- [ ] Review train vs test distribution shift
- [ ] Note any segment-specific patterns
- [ ] (If time series) check stationarity and seasonality

## Next Steps
After reviewing EDA, run:
- \`/dev-docs <next-step>\` to plan
- \`/grade\` after first submission
- \`/meta-optimize\` to review memory health
EOL
echo "  ✓ Generated $SUMMARY_FILE"
echo ""

echo "=== EDA Complete ==="
echo "Open: $SUMMARY_FILE"
ls -la eda/ 2>/dev/null
