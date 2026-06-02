#!/bin/bash
# Cross-Model Review — invoke external LLM as critic
# Auto-detects available backend: agy (antigravity) > gemini > codex > ollama
#
# Usage:
#   bash cross_review.sh "<prompt>"
#   bash cross_review.sh "<prompt>" --backend=agy
#
# Output: prints the LLM's response to stdout

set -e

PROMPT="${1:-}"
BACKEND="${2:-auto}"

if [ -z "$PROMPT" ]; then
  echo "Usage: $0 '<prompt>' [--backend=<agy|gemini|codex|ollama>]" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $0 'Review my approach to TPS May 2022'" >&2
  echo "  $0 'What am I missing?' --backend=gemini" >&2
  exit 2
fi

# Auto-detect backend
detect_backend() {
  if [ -x "/Users/mac/.local/bin/agy" ] || command -v agy >/dev/null 2>&1; then
    echo "agy"
    return
  fi
  if command -v gemini >/dev/null 2>&1; then
    echo "gemini"
    return
  fi
  if command -v codex >/dev/null 2>&1; then
    echo "codex"
    return
  fi
  if command -v ollama >/dev/null 2>&1; then
    echo "ollama"
    return
  fi
  echo "none"
}

if [ "$BACKEND" = "auto" ]; then
  BACKEND=$(detect_backend)
fi

echo "Using backend: $BACKEND" >&2

# System prompt for review mode
REVIEW_PROREAM="You are a critical reviewer. Your job is to find problems, blind spots, and missing considerations in the proposed approach. Be specific. Be adversarial. Cite evidence. Do NOT just agree. Format your response as:

1. CLAIMS — List the key claims in the proposed approach
2. RISKS — What could go wrong
3. GAPS — What's missing
4. COUNTER-EVIDENCE — What prior work contradicts this
5. VERDICT — Should we proceed? If not, what would change your mind?

Here is the approach to review:

$PROMPT"

case "$BACKEND" in
  agy)
    /Users/mac/.local/bin/agy --print "$REVIEW_PROREAM" 2>&1 || \
    command -v agy >/dev/null 2>&1 && agy --print "$REVIEW_PROREAM" 2>&1
    ;;

  gemini)
    gemini -p "$REVIEW_PROREAM" 2>&1
    ;;

  codex)
    codex review "$REVIEW_PROREAM" 2>&1 || \
    codex exec "$REVIEW_PROREAM" 2>&1
    ;;

  ollama)
    ollama run llama3 "$REVIEW_PROREAM" 2>&1
    ;;

  none)
    echo "ERROR: No LLM CLI found. Install one of:" >&2
    echo "  - Antigravity: brew install antigravity-cli" >&2
    echo "  - Gemini:      brew install gemini-cli" >&2
    echo "  - Codex:       brew install codex" >&2
    echo "  - Ollama:      brew install ollama" >&2
    exit 3
    ;;

  *)
    echo "ERROR: Unknown backend: $BACKEND" >&2
    echo "Valid: agy, gemini, codex, ollama" >&2
    exit 2
    ;;
esac