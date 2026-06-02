---
description: Get an independent second opinion from an external LLM (agy/gemini/codex/ollama) on a decision
---

# /review — Cross-Model Review

> Use an external LLM as a critic. Breaks self-play blind spots. Auto-detects available backend.

## Usage

```
/review <topic>
/review <topic> --backend=agy
```

Examples:
- `/review should I add CatBoost to the TPS May 2022 stack?`
- `/review is this the right CV strategy for time-series data?`
- `/review before I submit my final submission for jigsaw-toxic`

## What This Does

1. Summarizes your current approach + key claims
2. Invokes `cross_review.sh` (auto-detects: agy > gemini > codex > ollama)
3. Captures the critique
4. Synthesizes: what to keep, what to reject, what's new
5. Saves the review to `memory/cross-reviews/<topic-slug>.md`
6. Adds an entry to MEMORY.md

## Backend Auto-Detection

The script checks in this order:
1. `agy` (Antigravity CLI) — preferred
2. `gemini` (Google Gemini CLI)
3. `codex` (OpenAI Codex CLI)
4. `ollama` (local models, no API)

If none installed, falls back to adversarial self-check (5-question rubric).

## When to Use This

- Before submitting a final submission
- Before making an architectural decision
- After 3+ failed attempts at the same problem
- When you suspect your reasoning is going in circles

## When NOT to Use

- Trivial decisions (not worth the latency)
- Decisions where you've already gotten external input
- When the answer is clearly defined (e.g., "use GroupKFold for groups")

## Anti-Patterns

- ❌ Using /review to validate (asking for confirmation, not critique)
- ❌ Ignoring the critique
- ❌ Not documenting the outcome
