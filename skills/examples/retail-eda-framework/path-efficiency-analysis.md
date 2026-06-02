---
name: path-efficiency-analysis
description: |
  Analyze the EFFICIENCY of how a session was executed, not just the outcome.
  Use during /claudeception retrospectives or when reviewing session logs.
  Computes: Recovery Rate, Repetitiveness Rate, Tool Productivity,
  Costliest Wrong Path. Research basis: cultivating-ml-agent §11.7 Path
  Efficiency Analysis (P2).
---

# Path Efficiency Analysis (P2)

> Don't just ask "what was learned?" — ask "how efficiently was it learned?"

## The Rule

**At every retrospective, compute 4 metrics.** A 95% accurate model built with 10x the work isn't a 95% accurate model — it's a 50% efficient one.

## The 4 Metrics

### 1. Recovery Rate (target: 100%)
> Of the errors/wrong paths encountered, what fraction was corrected?

- High recovery = good (catches mistakes fast)
- Low recovery = bad (stays on wrong path)
- Recovery Rate = errors_corrected / total_errors
- "Errors corrected" = subsequent actions fixed or pivoted away from a mistake
- "Total errors" = all detected wrong paths in session

### 2. Repetitiveness Rate (target: < 10%)
> What fraction of actions were repeated without learning?

- High repetitiveness = bad (going in circles)
- Low = good
- Repetitiveness Rate = repeated_actions / total_actions
- "Repeated" = same command/operation 3+ times without state change
- Use this to catch "trying the same thing and hoping it works"

### 3. Tool Productivity (target: > 80%)
> Of all tool calls, what fraction produced useful work?

- High = good (efficient tool use)
- Low = bad (lots of failed/exploratory tool calls)
- Tool Productivity = useful_calls / total_calls
- "Useful" = resulted in a file change, real output, or useful state
- "Unuseful" = error, empty response, repeated read

### 4. Costliest Wrong Path (target: minimize)
> What's the longest/most-expensive sequence of actions that led to a wrong outcome?

- Identify the most expensive mistake
- What early warning signals were missed?
- What checkpoint would have caught it earlier?

## How to Apply

```bash
# Run on current session log
bash ml-agent-code-template/.claude/hooks/path_efficiency.sh <log-file>

# Run on a project session
bash ml-agent-code-template/.claude/hooks/path_efficiency.sh \
  --project=ml-agent-test-tps-may \
  --since=2026-06-01
```

## Real Case: TPS May 2022 (Path Inefficiency Examples)

### ❌ Inefficient Pattern (Wasted ~3 hours)

```
12:00 - Tried CatBoost (know slow)
12:30 - CatBoost fold 1 done (0.7 AUC, dead end)
13:00 - Tried CatBoost with different params
13:30 - Same result
14:00 - Read CatBoost docs
14:30 - Tried CatBoost v2
15:00 - Realized it's on dead-ends list
15:30 - Pivoted to LGB
```

- **Recovery Rate**: 50% (caught at hour 3.5)
- **Repetitiveness**: 4 CatBoost attempts (80% of actions)
- **Tool Productivity**: 30% (mostly wasted on CatBoost)
- **Costliest Wrong Path**: 3 hours on CatBoost
- **Early warning missed**: CatBoost on dead-ends list from Day 1

### ✅ Efficient Pattern (Caught early)

```
14:00 - Self-critique: "CatBoost on 800K rows, slow, weak"
14:05 - Read dead-ends list — confirmed
14:10 - Pivoted to LGB
14:30 - LGB running
```

- **Recovery Rate**: 100% (caught in 5 min)
- **Repetitiveness**: 0%
- **Tool Productivity**: 100%
- **Costliest Wrong Path**: 5 minutes

## What to Do With Metrics

| Scenario | Action |
|----------|--------|
| Recovery Rate < 70% | Add more mid-task self-critique checkpoints |
| Repetitiveness > 20% | Add dead-end check at start of any new direction |
| Tool Productivity < 60% | Investigate why — usually stuck in research loop |
| Costliest Wrong Path > 1h | Document in dead-ends, add to skill activation |

## Anti-Patterns

- ❌ Only measuring outcome (final score) without measuring process
- ❌ Ignoring the "how" when reflecting on what worked
- ❌ Forgetting that you can have 100% accuracy in 10x the time

## Empirical Evidence

**cultivating-ml-agent §11.7**:
- "Path Efficiency Analysis (P2 — 增强 Claudeception)"
- Metrics: Recovery Rate / Repetitiveness Rate / Tool Productivity / Costliest Wrong Path
- "Identifying 重复无效动作 / 昂贵的错误路径 / 早期预警信号"

## Related

- `self-critique-checkpoint` — Reduces all 4 metrics' badness
- `check-dead-ends` — Targets the "Costliest Wrong Path" issue
- `meta-optimize` — Detects memory issues over time

## Output Format

```
=== Path Efficiency Report ===
Session: <name>
Duration: 2h 30min
Total actions: 47

1. Recovery Rate:    85%  (11/13 errors caught)
2. Repetitiveness:   8%   (4/47 repeated actions)
3. Tool Productivity: 78%  (37/47 useful)
4. Costliest Wrong Path: 45 min on "CatBoost tuning"
   Early warning signal: dead-ends list (not consulted)

Recommendation: Run self-critique at start of any new direction
```
