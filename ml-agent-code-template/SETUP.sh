#!/bin/bash
# ML Agent Template — One-time Setup
# Run after cloning: `bash SETUP.sh`
#
# What this does:
# 1. Verify Python 3 (>= 3.9) and required packages
# 2. Create venv and install ML packages
# 3. Install hooks to ~/.claude/hooks/ (merge with existing)
# 4. Update ~/.claude/settings.json to register hooks (merge)
# 5. Initialize memory/ structure if missing
# 6. Install Claude Code skill stubs
# 7. Print what's been installed
set -e

echo "=== ML Agent Template Setup ==="
echo ""

# Detect if we're in a git repo
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  REPO_NAME=$(basename "$(git rev-parse --show-toplevel 2>/dev/null)")
else
  REPO_NAME="ml-agent-project"
fi

echo "Project: $REPO_NAME"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── Step 1: Python environment ──────────────────────────────────────────────
echo "[1/6] Verifying Python environment..."

PYTHON=""
for cmd in python3.11 python3.12 python3.10 python3.9 python3; do
  if command -v "$cmd" >/dev/null 2>&1; then
    PYTHON=$(command -v "$cmd")
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "ERROR: Python 3 not found. Install via Homebrew or python.org"
  exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_OK=$($PYTHON -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" && echo "yes" || echo "no")

if [ "$PY_OK" = "no" ]; then
  echo "WARNING: Python $PY_VERSION < 3.9. Some packages may not work."
fi

echo "  Python: $PY_VERSION at $PYTHON"

# ── Step 2: venv + packages ──────────────────────────────────────────────────
echo ""
echo "[2/6] Setting up venv..."

if [ ! -d ".venv" ]; then
  $PYTHON -m venv .venv
  echo "  Created .venv"
else
  echo "  .venv already exists"
fi

# shellcheck source=/dev/null
source .venv/bin/activate

# Install core ML packages
echo "  Installing core packages..."
pip install --quiet --upgrade pip 2>/dev/null || true
pip install --quiet \
  numpy pandas scikit-learn scipy lightgbm

# Verify
$PYTHON -c "import numpy, pandas, sklearn, lightgbm, scipy; print('  All packages OK')" 2>/dev/null || {
  echo "  WARNING: Some packages failed to install. Re-run manually: pip install numpy pandas scikit-learn lightgbm scipy"
}

# ── Step 3: Create submissions and logs dirs ────────────────────────────────
echo ""
echo "[3/6] Creating output directories..."
mkdir -p submissions logs dev/active dev/archive
echo "  submissions/, logs/, dev/active/, dev/archive/ ready"

# ── Step 4: Install hooks ────────────────────────────────────────────────────
echo ""
echo "[4/6] Installing hooks to ~/.claude/hooks/..."

mkdir -p ~/.claude/hooks
HOOK_INSTALLED=0
HOOK_SKIPPED=0

for hook_file in .claude/hooks/*.sh; do
  if [ ! -f "$hook_file" ]; then continue; fi
  HOOK_NAME=$(basename "$hook_file")
  TARGET=~/.claude/hooks/$HOOK_NAME

  if [ -f "$TARGET" ]; then
    # Compare — skip if identical, backup if different
    if diff -q "$hook_file" "$TARGET" >/dev/null 2>&1; then
      HOOK_SKIPPED=$((HOOK_SKIPPED+1))
    else
      cp "$TARGET" "${TARGET}.bak.$(date +%Y%m%d-%H%M%S)"
      cp "$hook_file" "$TARGET"
      chmod +x "$TARGET"
      HOOK_INSTALLED=$((HOOK_INSTALLED+1))
    fi
  else
    cp "$hook_file" "$TARGET"
    chmod +x "$TARGET"
    HOOK_INSTALLED=$((HOOK_INSTALLED+1))
  fi
done

echo "  Installed: $HOOK_INSTALLED, Skipped (already present): $HOOK_SKIPPED"

# ── Step 5: Update ~/.claude/settings.json ──────────────────────────────────
echo ""
echo "[5/6] Registering hooks in ~/.claude/settings.json..."

SETTINGS=~/.claude/settings.json

# Build the hooks config we want
build_hooks_config() {
  cat <<'JSON'
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {"type": "command", "command": "${HOME}/.claude/hooks/git_risk_hook.sh"}
        ]
      }
    ],
    "PostToolUse": [],
    "UserPromptSubmit": [
      {
        "matcher": "*",
        "hooks": [
          {"type": "command", "command": "${HOME}/.claude/hooks/skill_activation_hook.sh"}
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "*",
        "hooks": [
          {"type": "command", "command": "${HOME}/.claude/hooks/session_start_hook.sh"}
        ]
      }
    ],
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {"type": "command", "command": "${HOME}/.claude/hooks/stop_audit.sh", "timeout": 30}
        ]
      }
    ]
  }
}
JSON
}

if [ ! -f "$SETTINGS" ]; then
  # No existing settings — write fresh
  build_hooks_config | python3 -m json.tool > "$SETTINGS"
  echo "  Created new $SETTINGS"
else
  # Existing settings — merge hooks block
  HOOK_BLOCK=$(build_hooks_config | python3 -c "
import json, sys
config = json.load(sys.stdin)
print(json.dumps(config.get('hooks', {}), indent=2))
")

  python3 <<PYEOF
import json, sys
from pathlib import Path

settings_path = Path("$SETTINGS")
new_hooks = json.loads('''$HOOK_BLOCK''')

# Backup
backup = settings_path.with_suffix(f".json.bak.{Path(str(__import__('datetime').datetime.now())).stem.replace(' ', '-').replace(':', '')}")
import shutil
shutil.copy(settings_path, backup)
print(f"  Backup: {backup}")

with open(settings_path) as f:
    try:
        current = json.load(f)
    except json.JSONDecodeError:
        print(f"  WARNING: $SETTINGS is not valid JSON. Skipping merge.")
        sys.exit(0)

# Merge hooks — for each event, append new hooks if not already present
existing_hooks = current.setdefault("hooks", {})
for event, hook_groups in new_hooks.items():
    existing_event = existing_hooks.setdefault(event, [])

    # Index existing commands
    existing_cmds = set()
    for group in existing_event:
        for hook in group.get("hooks", []):
            cmd = hook.get("command", "")
            if cmd:
                existing_cmds.add(cmd)

    # Append new hook groups that don't duplicate
    for group in hook_groups:
        new_cmds = [h.get("command", "") for h in group.get("hooks", [])]
        if any(c not in existing_cmds for c in new_cmds):
            existing_event.append(group)

with open(settings_path, "w") as f:
    json.dump(current, f, indent=2)
print(f"  Updated $SETTINGS")
PYEOF
fi

# ── Step 6: Install skills to ~/.claude/skills/ ─────────────────────────────
echo ""
echo "[6/6] Installing skills to ~/.claude/skills/..."

SKILLS_INSTALLED=0
SKILLS_SKIPPED=0
mkdir -p ~/.claude/skills

if [ -d ".claude/skills" ]; then
  for skill_dir in .claude/skills/*/; do
    [ ! -d "$skill_dir" ] && continue
    SKILL_NAME=$(basename "$skill_dir")
    [ "$SKILL_NAME" = "SKILL.md.template" ] && continue

    # Only copy if SKILL.md exists
    if [ ! -f "$skill_dir/SKILL.md" ]; then
      continue
    fi

    TARGET=~/.claude/skills/$SKILL_NAME
    if [ -d "$TARGET" ]; then
      SKILLS_SKIPPED=$((SKILLS_SKIPPED+1))
    else
      mkdir -p "$TARGET"
      cp "$skill_dir/SKILL.md" "$TARGET/SKILL.md"
      SKILLS_INSTALLED=$((SKILLS_INSTALLED+1))
    fi
  done
fi

echo "  Installed: $SKILLS_INSTALLED, Skipped: $SKILLS_SKIPPED"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Installed:"
echo "  - Python venv with numpy, pandas, sklearn, lightgbm, scipy"
echo "  - Hooks in ~/.claude/hooks/ (auto-loads on next session)"
echo "  - Hooks registered in ~/.claude/settings.json"
echo "  - Skills in ~/.claude/skills/ (auto-activates on relevant prompts)"
echo ""
echo "Test it:"
echo "  1. Start a new Claude Code session in this directory"
echo "  2. Type: 'Let me try a new feature engineering approach'"
echo "  3. You should see auto-suggested skills in the response"
echo ""
echo "Customize:"
echo "  - .claude/skill-rules.json — adjust pattern→skill mappings"
echo "  - .claude/hooks/*.sh — modify hook behavior"
echo "  - memory/MEMORY.md — add your domain knowledge"
echo "  - CLAUDE.md — project-specific instructions"
echo ""
echo "Next steps:"
echo "  1. Read CLAUDE.md — global instructions"
echo "  2. Read memory/MEMORY.md — project memory index"
echo "  3. Review .claude/skill-rules.json — adjust patterns"
echo "  4. Run your first experiment with the agent"