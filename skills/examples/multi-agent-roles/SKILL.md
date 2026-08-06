---
name: multi-agent-roles
description: |
  Use when designing a multi-agent system, when a single agent has too many responsibilities, when planning complex ML pipelines, or when observing "agent attention fragmentation" symptoms. Triggers when a task involves planning + coding + review steps.
---

# Multi-Agent Role Architecture

## Context
A single agent stuffed with all system prompts + 43+ skills gets confused (attention fragmentation). agy verified that splitting into **Architect / Coder / Critic / Researcher / Integrator** roles reduces hallucination and improves complex pipeline stability. Inspired by AutoGen, CrewAI, and MetaGPT.

The core insight: **single agent = single bottleneck**; role separation enables parallel work and specialized expertise.

## Guidance

### Built-in Roles

```python
from framework.src.agents import Orchestrator

orch = Orchestrator(llm_call=my_llm_function)

# 标准 ML pipeline: Architect → Coder → Critic with revision loop
pipeline = orch.create_standard_ml_pipeline()
outputs = orch.run_pipeline("standard-ml",
                             "Predict Spaceship Titanic Transported",
                             max_revisions=2)
# → Architect outputs plan → Coder writes code → Critic reviews
# → If REJECT, Architect revises with Critic feedback
```

### Available Roles

| Role | Responsibility | Forbidden |
|---|---|---|
| `data-architect` | Plan EDA/FE/model strategy | Write code, submit |
| `ml-coder` | Implement plans as Python | Modify plan, submit |
| `continuity-critic` | Catch leakage/overfitting | Fix code, submit |
| `knowledge-researcher` | Find papers + Kaggle insights | Recommend implementations |

### Custom Pipeline

```python
from framework.src.agents import Pipeline

custom = (Pipeline(name="research-then-code", description="...")
          .add_step("knowledge-researcher", input_from=None, output_to="data-architect")
          .add_step("data-architect", input_from="knowledge-researcher", output_to="ml-coder")
          .add_step("ml-coder", input_from="data-architect"))

orch.register_pipeline(custom)
orch.run_pipeline("research-then-code", "Find best technique for X")
```

## Why This Matters

| Single Agent | Multi-Agent |
|---|---|
| 1 confused "do everything" prompt | 4 focused "do one thing" prompts |
| Hallucinates across domains | Specializes per role |
| One failure breaks everything | Failure isolated per step |
| Hard to debug | Each step traceable |

agy: "Domain-Specific Multi-Agent Roles is a **true trend**; single-prompt generalization is limited."

## When to Apply

### When to Use
- Complex ML pipeline (EDA + FE + model + submission)
- Tasks where planning ≠ execution ≠ review
- Building production ML agents
- When a single agent's context has >3 distinct responsibilities

### When NOT to Use
- Simple Q&A (overkill)
- Single-step tasks (no benefit)
- When LLM API budget is constrained (multi-agent = more calls)

## Notes
- **Mock mode for testing**: `Orchestrator(llm_call=None)` returns echo responses for pipeline logic testing
- **Pass roles explicitly**: don't reuse one role for multiple pipelines
- **Combine with Reflexion**: Coder outputs run through `runtime-reflexion` loop
- **Revision budget**: 2-3 max, more = diminishing returns + cost
- See also: `runtime-reflexion`, `self-evolving-skills`

## References
- Implementation: `framework/src/agents/`
- Inspired by: AutoGen (Microsoft), CrewAI, MetaGPT, Data Interpreter (AAAI 2025)
- Standard pipeline: Architect → Coder → Critic (agy recommended)