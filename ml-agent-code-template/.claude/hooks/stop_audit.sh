#!/bin/bash
# Stop Hook — Modified Files Security Audit (global, git-aware)
# Only flags production code, skips ML research / experimental scripts.
# Exit 0 = clean, Exit 2 = issues found (session pauses for confirmation)

INPUT=$(cat)

# Get current working directory
CWD=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cwd',''))" 2>/dev/null)
if [ -z "$CWD" ] || [ ! -d "$CWD" ]; then
    CWD="$(pwd)"
fi

cd "$CWD" || exit 0

# Only run in git repos
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    exit 0
fi

# Get modified files
MODIFIED=$(git diff --name-only HEAD 2>/dev/null)
UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null)
ALL_FILES=$(echo -e "${MODIFIED}\n${UNTRACKED}" | grep -v '^$' | grep -v -E 'node_modules|__pycache__|\.git|\.venv|venv|dist|build|\.min\.|venv37' | sort -u)

if [ -z "$ALL_FILES" ]; then
    exit 0
fi

declare -a ISSUES
ISSUE_COUNT=0
FOUND=0

# ── Always flag: secret / key leaks (any file type) ──────────────────
SECRET_PATTERNS="api[_-]?key\s*[=:]\s*['\"][a-zA-Z0-9]{16,}['\"]|secret\s*[=:]\s*['\"][a-zA-Z0-9]{16,}['\"]|token\s*[=:]\s*['\"][a-zA-Z0-9_\-]{20,}['\"]|bearer\s+[a-zA-Z0-9\-_\.]{20,}|sk-[a-zA-Z0-9]{20,}|-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----|aws[_-]?(access[_-]?key|secret)"

# ── Flag console.* in production JS/TS (frontend + backend services) ──
# Frontend
FRONTEND_PATTERNS="console\.(log|warn|error|debug|info|table|group)\s*\("
# Backend: Express / FastAPI / Flask routes
BACKEND_PATTERNS="console\.(log|warn|error|debug)\s*\("

# ── Skip console.* in these contexts ──
# ML/data science (always has print/console for training logs)
SKIP_DIRS="mle-bench|ML-Agent|kaggle|competitions|research|notebooks|sandbox|tmp|/tmp/|kaggle|\.ml\b"
# Known safe patterns in any file
SKIP_LINES="//.*console\.|#.*console\.|# print\(|# logging|tqdm\.set_|logger\.info|logger\.debug|log_file|log_path|log_dir|logging\.|\.log\.(info|debug|warn)"

# Production file extensions (flag console.* here)
PROD_EXTENSIONS="\.jsx?$|\.tsx?$|\.vue$|\.ts$|\.js$|\.mjs$|\.cjs$|\.php$|\.rb$"
# Backend-only JS patterns (e.g. Express route files)
BACKEND_EXTENSIONS="route|handler|controller|service|api|middleware|server|app\."

while IFS= read -r FILE; do
    FULL_PATH="${CWD}/${FILE}"
    if [ ! -f "$FULL_PATH" ]; then
        continue
    fi

    # ── Always check secrets (in ALL file types) ──
    SECRET_MATCHES=$(grep -nE "$SECRET_PATTERNS" "$FULL_PATH" 2>/dev/null | \
        grep -vE '\.env|\.sample|test|mock|fake|sandbox|staging|stub|placeholder|example|SANDBOX|TODO|FIXME|YOUR_' | head -5)
    if [ -n "$SECRET_MATCHES" ]; then
        while IFS= read -r LINE; do
            ISSUES+=("  [SECRET] ${FILE}:${LINE}")
            ISSUE_COUNT=$((ISSUE_COUNT+1))
        done <<< "$SECRET_MATCHES"
        FOUND=$((FOUND+1))
        continue  # Found secrets → skip other checks for this file
    fi

    # ── Skip known ML / research directories ──
    if echo "$FILE" | grep -qE "$SKIP_DIRS"; then
        continue
    fi

    # ── Skip non-production file extensions ──
    if ! echo "$FILE" | grep -qE "$PROD_EXTENSIONS"; then
        continue
    fi

    # ── Check console.* in production JS/TS ──
    CONSOLE_MATCHES=$(grep -nE "$FRONTEND_PATTERNS" "$FULL_PATH" 2>/dev/null | \
        grep -vE "$SKIP_LINES" | head -5)
    if [ -n "$CONSOLE_MATCHES" ]; then
        while IFS= read -r LINE; do
            ISSUES+=("  [console] ${FILE}:${LINE}")
            ISSUE_COUNT=$((ISSUE_COUNT+1))
        done <<< "$CONSOLE_MATCHES"
        FOUND=$((FOUND+1))
    fi

    # ── Check console.* in backend JS/TS (route/controller/service files) ──
    if echo "$FILE" | grep -qE "$BACKEND_EXTENSIONS"; then
        BACKEND_MATCHES=$(grep -nE "$BACKEND_PATTERNS" "$FULL_PATH" 2>/dev/null | \
            grep -vE "$SKIP_LINES" | head -3)
        if [ -n "$BACKEND_MATCHES" ]; then
            while IFS= read -r LINE; do
                ISSUES+=("  [console] ${FILE}:${LINE}")
                ISSUE_COUNT=$((ISSUE_COUNT+1))
            done <<< "$BACKEND_MATCHES"
            FOUND=$((FOUND+1))
        fi
    fi

    # Early exit if too many issues
    if [ "$ISSUE_COUNT" -ge 20 ]; then
        break
    fi
done <<< "$ALL_FILES"

if [ "$FOUND" -gt 0 ]; then
    echo "" >&2
    echo "═══ Stop Hook: ${ISSUE_COUNT} issue(s) in ${FOUND} file(s) ═══" >&2
    printf '%s\n' "${ISSUES[@]}" | head -30 >&2
    if [ "$ISSUE_COUNT" -gt 30 ]; then
        echo "  ... and $((ISSUE_COUNT-30)) more" >&2
    fi
    echo "" >&2
    exit 2
fi

exit 0