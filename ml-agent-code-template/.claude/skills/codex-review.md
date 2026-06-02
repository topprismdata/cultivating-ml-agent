---
name: codex-review
description: Use an external LLM (Codex, Gemini, Antigravity, Ollama) as a critic on your work. Breaks self-play blind spots.
type: learned
---

# Cross-Model Review

> Use a different LLM as a critic on your work. Two models agreeing is more reliable than one.

## When to Use

**Before high-stakes decisions:**
- Submitting a final submission
- Choosing between two strong approaches
- Concluding a direction is "impossible"
- Making architectural changes
- Publishing a retrospective

**When you have a blind spot:**
- After 3+ attempts at the same problem (tunnel vision)
- After getting an unexpectedly good/bad result
- When you can't find the bug (your model is hallucinating a fix)

## The Rule

**Always get an independent second opinion before committing.**

Two models agreeing is not certainty, but it dramatically reduces the chance of subtle errors.

## How to Apply

```
1. State your current approach + key claims clearly
2. Pass to external LLM with: "Review this approach. What am I missing? What's wrong?"
3. Read the critique carefully — even adversarial feedback is signal
4. Synthesize: keep what's validated by both, reject what isn't
5. Document the review in memory/cross-reviews/<topic>.md
```

## Backend Selection

Multiple LLM CLIs are supported. Use whichever is installed:

| Backend | Binary | When to use |
|---------|--------|-------------|
| **Antigravity** | `agy` | Default; Google's agent platform |
| **Gemini** | `gemini` | Direct Gemini API access |
| **Codex** | `codex` | OpenAI's Codex CLI |
| **Ollama** | `ollama` | Local models, no API needed |

The cross-review script `cross_review.sh` auto-detects which is available.

## CLI Invocations

```bash
# Antigravity (preferred)
agy --print "Review this approach: <approach>. What am I missing?"

# Gemini
gemini -p "Review this approach: <approach>. What am I missing?"

# Codex
codex review "Review this approach: <approach>. What am I missing?"

# Ollama (local)
ollama run llama3 "Review this approach: <approach>..."
```

If no CLI is installed, fall back to:
- Web search + manual review
- Adversarial self-check (rubric below)

## Adversarial Self-Check (No External Model)

If you can't get an external model, run this checklist:

1. **What could be wrong with this?** (be specific)
2. **What did I assume without verifying?** (cite evidence)
3. **What would a hostile reviewer say?** (steel-man the critique)
4. **Is this consistent with prior experiments?** (cite counter-examples)
5. **If this fails, what would explain it?** (predicted failure modes)

If you can't answer 4+5 confidently, get external review.

## Empirical Evidence

**ARIS (wanshuiyin, 11.1k stars)** uses cross-model review as a core design principle:
> "Two is the minimum needed to break self-play blind spots"
> "Claude Code drives execution; Codex MCP, Oracle, Gemini, etc. act as critical reviewer"

**ARIS anti-self-poisoning**: When the model is the only source of validation, it can reinforce its own errors. External review catches this.

## Anti-Patterns

- ❌ Using external LLM as a "yes-man" (asking "is this right?" gets sycophantic answers)
- ❌ Skipping review because "this is obvious" (obvious is where blind spots hide)
- ❌ Reviewing trivial decisions (waste of cycles)
- ❌ Ignoring the critique to preserve ego (ego > signal = bad science)

## Output Format

When you get a review, document it:

```markdown
# Cross-Review: <topic>

## Primary Approach
<what you were going to do>

## Critique from <backend>
<verbatim response, key points>

## Synthesis
- Keep: <validated by both>
- Reject: <rejected after review>
- Add: <new ideas from review>

## Outcome
<what you did differently>
```

Save to `memory/cross-reviews/<topic>.md` and reference from MEMORY.md.
