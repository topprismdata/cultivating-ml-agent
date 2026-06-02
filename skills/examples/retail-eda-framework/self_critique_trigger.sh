#!/bin/bash
# Self-Critique Checkpoint Trigger (UserPromptSubmit)
# Detects "about to execute" prompts and injects a 5-question self-critique.
# Per memory/skills/self-critique-checkpoint.md
#
# Exit 0 = allow, additionalContext injected for Claude

set -e

INPUT=$(cat)

# Parse input using python (write to temp file to avoid escaping issues)
PROMPT_TMP=$(mktemp 2>/dev/null || echo "/tmp/.self-critique-prompt-fallback-$$")
echo "$INPUT" > "$PROMPT_TMP"

# Do all the work in Python
PROMPT_FILE="$PROMPT_TMP" python3 <<'PYEOF'
import os, re, sys, json

prompt_file = os.environ['PROMPT_FILE']
try:
    with open(prompt_file) as f:
        input_str = f.read()
    d = json.loads(input_str)
    prompt = d.get('prompt', d.get('user_prompt', ''))
except Exception as e:
    sys.exit(0)

# Skip empty / very short prompts
if not prompt or len(prompt) < 10:
    sys.exit(0)

# Ignore patterns (questions, hypotheticals)
ignore_patterns = [
    r'[^?]*\?\s*$',
    r'^\s*(hi|hello|thanks?|okay)\b',
    r'^\s*what\s+(is|are|if)\b',
    r'^\s*can\s+you\b',
    r'^\s*show\s+me\b',
    r'^\s*how\s+(do|can|should|would)\b',
    r'^\s*why\b',
]
for pat in ignore_patterns:
    try:
        if re.search(pat, prompt, re.IGNORECASE):
            sys.exit(0)
    except re.error:
        continue

# Execution patterns (about to commit to a direction)
exe_patterns = [
    r"\blet['\u2019]?s\s+(try|use|train|do|run|build|commit)\b",
    r"\blet\s+me\s+(try|use|train|do|run|build|commit)\b",
    r"\b(I['\u2019]?m\s+going\s+to|going\s+to|will|gonna)\s+(try|use|train|do|run|build|commit)\b",
    r"\b(about\s+to|planning\s+to|ready\s+to)\s+(try|use|train|do|run|build)\b",
    r"\b(my\s+plan|the\s+plan|i['\u2019]?ll\s+use|i['\u2019]?ll\s+try)\b",
    r"\b(starting|starting\s+to)\s+(training|building|running)\b",
    r"\bi\s+(want|plan|need)\s+to\s+(try|use|train|do|run|build)\b",
]

matched = False
for pat in exe_patterns:
    try:
        if re.search(pat, prompt, re.IGNORECASE):
            matched = True
            break
    except re.error:
        continue

if not matched:
    sys.exit(0)

# Build context
preview = prompt[:200].replace('\n', ' ')
context = (
    "🪞 SELF-CRITIQUE CHECKPOINT (P0): You're about to commit to a direction.\n\n"
    "Before proceeding, answer the 5 questions from `memory/skills/self-critique-checkpoint.md`:\n\n"
    "1. **Hypothesis**: 'Doing X will achieve Y because Z' — state explicitly\n"
    "2. **Evidence**: Cite prior experiments or domain knowledge\n"
    "3. **Expected range**: Best case / worst case / probability of each\n"
    "4. **Failure mode**: What could go wrong? Check memory/feedback_no_recheck_confirmed_dead.md\n"
    "5. **Plan B**: If it fails, what's next?\n\n"
    "If any answer is 'no' or 'I don't know' → resolve before executing.\n\n"
    f"Your prompt: `{preview}`"
)

# Output JSON
out = {
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": context
    }
}
print(json.dumps(out, ensure_ascii=False))
PYEOF

# Cleanup
rm -f "$PROMPT_TMP"

exit 0