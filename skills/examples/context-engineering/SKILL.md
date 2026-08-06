---
name: context-engineering
description: |
  Use when context window is getting crowded, when dealing with long training logs, or when preparing prompts with mixed-priority content (system / task / skills / examples). Triggers when assembling prompts >4000 chars, or when observing "Lost in the Middle" symptoms (model ignores mid-prompt content).
---

# Context Engineering

## Context
Even with 1M-2M token context windows, stuffing everything degrades performance ("Lost in the Middle" — Liu et al. 2023). This skill encodes Anthropic's 2025 best practices: dynamic compression, intelligent log truncation, and priority-aware token budgeting.

The core insight: **context is RAM, treat it like memory management**, not a buffer.

## Guidance

### Three Tools

```python
from framework.src.context_engineering import (
    ContextCompressor, LogTruncator, TokenEconomy,
    CompressionLevel, BudgetPriority,
)

# 1. Compress long text (LLM outputs, paper abstracts)
compressor = ContextCompressor()
result = compressor.compress(long_paper_abstract,
                              level=CompressionLevel.MEDIUM,
                              focus_query="time series forecasting")
# → AGGRESSIVE for very long, MEDIUM for 4K-20K, LIGHT for <4K

# 2. Truncate training logs (preserve errors + samples)
truncated = LogTruncator(max_lines=200).truncate(training_log)
# → 500 lines → 50 lines, all ERROR/WARN kept, METRIC sampled

# 3. Priority-aware token budget
econ = TokenEconomy()
econ.add_critical(system_prompt, "system")
econ.add_high(current_task, "task")
econ.add(relevant_skill_text, BudgetPriority.MEDIUM, "skills")
econ.add(optional_context, BudgetPriority.LOW, "context")
prompt = econ.assemble()  # auto-drops LOW when over budget
```

### Decision Workflow

```
Assembling a prompt?
  ↓
1. Are there training logs > 100 lines?
   → LogTruncator first (reduce noise 5-10x)
  ↓
2. Are there paper abstracts / long docs?
   → Compressor + MEDIUM (keep relevant 50%)
  ↓
3. Total still > 8000 chars?
   → TokenEconomy (prioritize, auto-drop LOW)
  ↓
4. Send to LLM
```

## Why This Matters

| Without Context Engineering | With It |
|---|---|
| Lost in the Middle (model misses mid-prompt) | Key info always prioritized |
| 40% token waste on redundant/irrelevant content | 50-70% token savings |
| Inconsistent outputs from over-stuffed prompts | Stable, focused outputs |
| Can't fit new context when LLM call limit hit | Always room for one more thing |

Real ROI: agy research confirms **20% accuracy improvement** on code generation when context is properly engineered.

## When to Apply

### When to Use
- Any prompt > 4000 chars
- Training logs in context
- Mixing system + task + skills + examples
- "Model keeps ignoring this instruction"
- Multi-turn conversations with growing history

### When NOT to Use
- Short prompts (<1000 chars) — overhead not worth it
- When every detail must be preserved (use verbatim)
- Streaming scenarios (compress after stream ends)

## Notes
- **Combine with Memory Hierarchy** (don't compete): Memory decides what enters context; Engineering decides how to fit it
- **Measure first**: log token counts before/after to validate savings
- **Test with edge cases**: 50K chars, all-metrics log, mixed priorities
- See also: `memory-hierarchy-management`, `runtime-reflexion`

## References
- Implementation: `framework/src/context_engineering/`
- Inspired by: Anthropic Context Engineering Best Practices (2025)
- Research: Liu et al. "Lost in the Middle" (2023), Anthropic prompt caching (2024)