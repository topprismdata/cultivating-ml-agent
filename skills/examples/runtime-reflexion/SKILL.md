---
name: runtime-reflexion
description: |
  Use when running Python code via sandbox, when an experiment fails with a runtime error, when building self-correcting ML agents, or when debugging auto-generated code. Triggers on subprocess failures, ImportError, CUDA OOM, shape mismatch, or any "agent gets stuck on syntax error" pattern.
---

# Runtime Reflexion (Error → Diagnose → Retry)

## Context
Static skills freeze knowledge. Reflexion (Shinn et al. NeurIPS 2023) + Code Sandbox makes ML agents self-correct: **80%+ of runtime errors auto-fixable** when errors are precisely classified. agy verified this as a P0 must-have for ML agents (MLE-bench Kaggle medal correlation).

The core insight: **errors aren't random — they fall into ~12 categories**, each with known fixes.

## Guidance

### Single Execution with Diagnosis

```python
from framework.src.reflexion import CodeSandbox, ErrorAnalyzer, execute_with_analysis

sandbox = CodeSandbox(timeout_sec=30)
result, diagnosis = execute_with_analysis(generated_code, sandbox)

if not result.success:
    print(f"Error: {diagnosis.category.value}")
    print(f"Suggested fix: {diagnosis.suggested_fix}")
    print(f"Related skills: {diagnosis.related_skills}")
    # → "category=memory, fix=reduce batch_size, skills=[gpu-readiness-assessment]"
```

### Self-Correction Loop

```python
from framework.src.reflexion import ReflexionLoop, CodeSandbox

def my_llm_fix(prompt: str) -> str:
    """Call Claude/your LLM to fix code given the reflexion prompt"""
    return llm_client.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text

loop = ReflexionLoop(
    sandbox=CodeSandbox(timeout_sec=60),
    llm_fix=my_llm_fix,
)
result = loop.run(initial_code, max_attempts=5)

if result.success:
    deploy(result.final_code)
else:
    log_to_vault(result.final_diagnosis)  # remember for next time
```

### Error Categories (12)

| Category | Trigger | Fix |
|---|---|---|
| `syntax` | SyntaxError | Check indentation |
| `import` | ModuleNotFoundError | pip install |
| `shape` | shape mismatch | reshape/transpose |
| `memory` | CUDA OOM | reduce batch_size |
| `overfitting` | val_loss diverging | dropout/early stop |
| `data_leakage` | "leakage" mentioned | check temporal features |
| `timeout` | timed out | reduce data/optimize |
| ... | ... | ... |

## Why This Matters

Without Reflexion: agent gets stuck on syntax error → user manually fixes → no learning. **Repeat next competition.**

With Reflexion: agent auto-fixes 80%+ errors → escalates to LLM with precise diagnosis → logs to vault for future prevention.

agy: "Reflexion + Sandbox is the **only reliable deployment pattern** for code-generation agents in 2025-2026."

## When to Apply

### When to Use
- Auto-generated Python code from LLM
- ML pipeline runs that may fail
- Kaggle competition automation
- Agent benchmarks (MLE-bench, AIDE)
- Any "agent gets stuck and gives up" pattern

### When NOT to Use
- Single-shot queries (overkill)
- Production stable code (just monitor)
- When LLM fix is unavailable (record-only mode)

## Notes
- **Combine with sandbox isolation**: never run untrusted code outside subprocess
- **Set timeouts aggressively**: 30-60s default; ML training needs more
- **Log failures to vault**: future prevention (`memory-hierarchy-management`)
- **Track fix success rate**: <50% means your prompts need improvement
- See also: `multi-agent-roles` (Critic uses Reflexion), `self-evolving-skills`

## References
- Implementation: `framework/src/reflexion/`
- Research: Shinn et al. "Reflexion" (NeurIPS 2023), MLE-bench (OpenAI 2024)
- Pairing skill: `multi-agent-roles` (Continuity-Critic uses ErrorAnalyzer)