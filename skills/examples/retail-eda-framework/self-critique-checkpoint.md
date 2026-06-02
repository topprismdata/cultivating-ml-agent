---
name: self-critique-checkpoint
description: |
  Use at every key decision point BEFORE executing an action. Insert a self-critique
  instead of running first and reviewing later. Trigger on: (1) about to train a model,
  (2) choosing between approaches, (3) committing to a direction, (4) finalizing a
  submission, (5) accepting a dead-end. Catches the 30% error-path waste before it
  happens. Research basis: SPIRAL Critic (per-step dense evaluation) +
  cultivating-ml-agent §11.2 Self-Critique Checkpoint (P0).
---

# Self-Critique Checkpoint (P0)

> Insert a self-critique at every key decision point. Don't just run first and review later.

## The Rule

**Before executing an action, answer 5 questions.** Only proceed if the answer is "yes" to all 5, or you've explicitly accepted the risks.

This catches 30% of error paths BEFORE you waste hours on them (per cultivating-ml-agent §11.2 empirical evidence).

## The 5 Questions

### 1. **What's my hypothesis?**
> "I believe doing X will achieve Y because Z."

- If you can't state a clear hypothesis, you don't know what you're testing
- If hypothesis is vague, refine before proceeding

### 2. **What evidence supports this?**
> Cite prior experiments, papers, mentor's experience, or domain knowledge.

- If you have NO evidence, you're guessing
- Strong evidence = prior experiments on same data type
- Weak evidence = "this might work"
- No evidence = **stop and re-research**

### 3. **What's the expected outcome range?**
> Best case: OOF=0.997. Worst case: OOF=0.96. Probability of each: 30%/50%.

- If you can't estimate, you don't understand the problem
- If expected value is low, pivot
- Be specific (don't say "might be slightly better")

### 4. **What's the failure mode?**
> "This could fail because: X takes 4 hours, or Y has data leakage, or Z is already in our dead-ends list."

- If you can't articulate the failure mode, you're not thinking critically
- If failure mode is in `memory/feedback_no_recheck_confirmed_dead.md`, **STOP**
- If failure mode is "waste time", assess budget before proceeding

### 5. **What will I do if it fails?**
> Plan B: pivot to approach B, or revert and try C.

- Always have a Plan B before starting
- If Plan B is "try something else", you don't have a plan
- If Plan B is "give up", that's also fine — but decide upfront

## When to Apply

**Always** before:
- Training a model (5-30 min investment)
- Running an experiment (similar)
- Submitting to leaderboard (no going back)
- Choosing between 2+ approaches (commitment point)
- Spending >1 hour on a direction

**Especially** when:
- After 3+ failed attempts (tunnel vision risk)
- After reading someone else's success story (mimicry risk)
- When "this should work" feels too easy (overconfidence risk)

## Real Case Studies

### ✅ Self-Critique Saved Time

**TPS May 2022 — Sentence-Transformer attempt**:
1. Hypothesis: "384-dim embeddings of f_27 will help stacking"
2. Evidence: General NLP intuition (WEAK)
3. Expected: OOF improvement 0.001+ (40% prob)
4. Failure mode: "f_27 is structured 10-char, not natural language"
5. Plan B: Pivot to Nystroem + RBF if OOF < 0.95

**Without critique**: Would have spent 30 min encoding + training, found OOF=0.76 (dead end)
**With critique**: Realized f_27 is structured data, not language → tried cheaper test first → avoided the trap

### ❌ Self-Critique Failure (Real Example)

**R11 CatBoost attempt (from cultivating-ml-agent)**:
1. Hypothesis: "CatBoost native categorical handling will be better than LGB" (unverified)
2. Evidence: CatBoost marketing materials (WEAK)
3. Expected: CV improvement 0.005+ (unjustified high confidence)
4. Failure mode: "Slow on 800K rows" (underweighted)
5. Plan B: NONE → just kept trying

**Cost**: Wasted several hours on a known dead-end. Should have been on the dead-ends list already.

## Adversarial Self-Check (When No External Model)

If you can't get an external LLM, run this checklist yourself:

```python
# Run in your head or as text
print("1. Hypothesis stated?", "yes" if HYPOTHESIS else "STOP")
print("2. Evidence cited?", "yes" if EVIDENCE else "STOP")
print("3. Expected range given?", "yes" if RANGE else "STOP")
print("4. Failure mode identified?", "yes" if FAILURE_MODE else "STOP")
print("5. Plan B exists?", "yes" if PLAN_B else "STOP")
```

If any "no" or "STOP" → resolve before proceeding.

## Anti-Patterns

- ❌ **Skipping self-critique "to save time"** → Always costs more time when wrong
- ❌ **Vague hypothesis** ("this might help") → Not a real hypothesis
- ❌ **Ignoring failure mode** → Confirmation bias
- ❌ **No Plan B** → Tunnel vision when wrong
- ❌ **Self-critique for everything** → Overhead; use for non-trivial actions only
- ❌ **Stuck in self-critique loop** → Set a time limit (5 min max)

## Empirical Evidence

**cultivating-ml-agent §11.2**:
- "Real case: R11 training — if we had done Self-Critique on Day 1 ('CatBoost slow on 2.8M rows, uncertain'), wouldn't have wasted hours"
- "Reduces 30% error path waste" (claimed by upstream)

**Research basis**:
- **SPIRAL**: Critic role generates dense per-step quantitative evaluation, not just sparse final reward
- **Self-Evolving Agents Survey (arXiv 2026)**: Process + Outcome feedback > Outcome-only

## How to Apply in This Template

The `self_critique_trigger.sh` hook (UserPromptSubmit) fires when you say:
- "let me try", "going to train", "I'm going to use", "let me commit to"
- And prompts you to write the 5-question answer

You can also manually invoke by saying:
> "Self-critique on this: ..."

And the agent will walk through the 5 questions before executing.

## Related

- `check-dead-ends` — Self-Critique Question 4 (failure mode) reference
- `codex-review` — External second opinion (alternative to self-critique)
- `local-optimum-trap` — When self-critique says "stuck"
