# Multi-Agent ML Competition System Design

> Date: 2026-05-24
> Status: Approved for implementation

## 1. Overview

A persistent multi-agent system for running Kaggle competitions. Each competition gets a dedicated agent team that works across sessions, accumulating knowledge and improving over time.

**Goals**:
1. **Speed** — Parallel training + shared feature cache eliminates redundant computation
2. **Quality** — Multiple model strategies compete, ensemble picks the best
3. **Scale** — Mentor orchestrates multiple agent teams across competitions
4. **Knowledge** — Every experiment's insight persists to Obsidian for future agents

**Core principle**: Mentor does not train. It coordinates. Trainers do not talk to each other. They write to shared storage and notify via Obsidian.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│              Mentor (主会话, long-lived)              │
│                                                    │
│   ~/obsidian/ml-agent-memory/teams/{competition}/ │
└────────────────────────┬──────────────────────────┘
                         │ Task tool (parallel dispatch)
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
    │ Feature │     │  Model  │     │  Model  │
    │Engineer │     │Trainer A│     │Trainer B│
    │ (1 per  │     │ (GKFold)│     │(StratKF)│
    │  comp)  │     └────┬────┘     └────┬────┘
    └────┬────┘          │               │
         │               │               │
         ▼               │               │
   ~/shared/{comp}/     │               │
   ├── v{N}/            │               │
   │   └── features/     │               │
   └── oof/             │               │
                   ┌─────┴───────────────┘
                   │  Mentor polls file mtimes
             ┌─────▼─────┐
             │ Ensemble  │
             │Specialist │
             └─────┬─────┘
                   │
              Kaggle API
                   │
                   ▼
      ~/obsidian/ml-agent-memory/
      teams/{comp}/insights/{exp_name}.md
```

---

## 3. Directory Structure

### Shared Storage (per competition)

```
~/shared/{competition}/              # e.g., ~/shared/s6e5/
├── v1/                              # Feature version N
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── X_orig.npy
│   ├── y_train.npy
│   ├── test_ids.npy
│   ├── orig_y.npy
│   ├── feature_cols.json           # {"cols": [...], "cat_cols": [...], "num_cols": [...]}
│   ├── domain_stats.json           # Serialized domain statistics
│   ├── train_fe_full.pkl           # Full DataFrame for OOF target encoding
│   └── version_notes.md            # What features changed, why
├── v2/
│   └── ...
└── oof/                            # OOF predictions from all trainers
    ├── gkf/
    │   ├── oof.npy
    │   └── test_preds.npy
    ├── skf/
    │   ├── oof.npy
    │   └── test_preds.npy
    └── external/
        ├── oof.npy
        └── test_preds.npy
```

### Obsidian Team Memory (persistent, synced via iCloud)

```
~/obsidian/ml-agent-memory/
└── teams/
    └── {competition}/               # e.g., s6e5, s6e4
        ├── team_config.json       # {"competition": "s6e5", "members": [...], "active": true}
        ├── tasks/
        │   └── {agent_name}/
        │       ├── status.json    # {"state": "idle|running|done|error", "version": "v1", ...}
        │       └── log.md          # Execution log (append-only)
        └── insights/
            ├── {exp_id}_summary.md  # Brief result summary for quick reference
            └── {exp_id}_full.md     # Detailed experiment report
```

**File polling path**: Mentor reads `status.json` mtime to detect completion. No `.done` files needed.

---

## 4. Agent Roles

### 4.1 Mentor (主会话)

**Responsibilities**:
- Read Obsidian at session start to understand competition state
- Decide experiment plan based on CV/LB gap, past experiments
- Dispatch agents via Task tool
- Poll `status.json` files for completion
- Trigger Ensemble Specialist when all trainers done
- Write insights to Obsidian after each experiment
- Decide when to pivot or continue

**Never does**: Feature engineering, model training, ensembling logic

**Decision triggers** (from Obsidian):

| Situation | Action |
|-----------|--------|
| New competition | Start Feature Engineer, then Model Trainers |
| CV < LB by >0.01 | Trigger adversarial validation agent |
| CV plateau (<0.0001 for 3 iterations) | Pivot to different strategy |
| Experiment done | Run Ensemble Specialist, then decide next experiment |

### 4.2 Critic Agent (独立，对抗式)

**Trigger**: Mentor 调度，或 Model Trainer 结果出来后自动触发

**Responsibilities**:
- 在关键决策点提供对抗性审查
- 评估实验结果是否可信（避免噪音、假阳性）
- 检查是否存在局部最优陷阱
- 验证提交格式和数据泄漏

**检查时机**：

| 时机 | 触发条件 | 检查内容 |
|------|---------|---------|
| **实验规划** | Mentor 决定实验方向时 | "这个方向有历史证据吗？最大风险是什么？" |
| **CV 结果** | Model Trainer 完成时 | "这个提升是真的还是噪音？3 次类似结果了吗？" |
| **提交前** | Ensemble Specialist 生成 submission 时 | "格式正确吗？概率范围合理吗？有泄漏吗？" |
| **策略瓶颈** | 3 次迭代 <0.0001 提升时 | "是否陷入局部最优？应该 pivot 吗？" |

**Output**:

```json
// ~/obsidian/ml-agent-memory/teams/{comp}/tasks/critic/review_{exp_id}.json
{
  "exp_id": "v18_feature_exp",
  "trigger": "cv_result",
  "verdict": "caution",
  "confidence": 0.75,
  "concerns": [
    "CV improvement 0.0012 is within noise range (std=0.0011)",
    "Only 1 of 3 seeds showed improvement"
  ],
  "recommendations": [
    "Run 3 more seeds before submitting",
    "Check if improvement is stable across folds"
  ],
  "max_risk": "Submitting unreliable result wastes daily quota"
}
```

**Behavior**:
- Never says "all good" without checking — always outputs structured review
- Uses opposing evidence from Obsidian (`16-principles.md`, past experiments)
- Asks: "What would disprove this result?"
- Writes both JSON (machine-readable) and markdown (human-readable) to `review/`

**Mentor 的反应**：

```python
critic_review = json.load(f"teams/s6e5/tasks/critic/review_v18.json")

if critic_review['verdict'] == 'stop':
    print("Critic says STOP. Review: " + critic_review['concerns'])
    # Don't submit, pivot strategy
elif critic_review['verdict'] == 'caution':
    print("Critic says CAUTION. Recommendations: " + critic_review['recommendations'])
    # Apply recommended checks, then proceed
elif critic_review['verdict'] == 'proceed':
    print("Critic says PROCEED. Confidence: " + str(critic_review['confidence']))
    # Safe to submit
```

### 4.3 Feature Engineer

**Trigger**: Mentor dispatches via Task tool
**Input**: Competition data paths, competition metric, target column
**Output**: Versioned feature directory + domain_stats.json
**Behavior**:
- Creates `v{N}/` directory (increments from latest)
- Saves all artifacts as numpy arrays + JSON metadata
- Writes `status.json` with `"state": "done"` on completion
- Writes `version_notes.md` explaining what changed

**Versioning rules**:
- `v1/` is created on first run, never deleted
- `v{N+1}/` created only when experiment requires different features
- Each version is fully self-contained (can reproduce experiment)

### 4.4 Model Trainer A / B / C (parameterized)

**Trigger**: Mentor dispatches via Task tool, specifying:
- `strategy`: gkf | skf | external
- `version`: v1 | v2 | ...
- `model_type`: lgb | xgb | cb | all
- `seeds`: [42, 123, 456]

**Input**: Reads from `~/shared/{comp}/{version}/features/`
**Output**: `oof/{strategy}/oof.npy` + `test_preds.npy`
**Behavior**:
- Loads cached features (no recomputation)
- Runs 3 seeds × 5 folds with specified strategy
- Computes OOF AUC for CV-LB gap tracking
- Writes `status.json` with `"oof_auc": X.XXXX` on completion

**Parallelization**: Trainers run independently. Same version can have multiple strategies running simultaneously (A=B=skf is fine).

### 4.5 Ensemble Specialist

**Trigger**: Mentor, after all Model Trainers report done
**Input**: `oof/*/test_preds.npy` from all strategies
**Output**: Submission file + Obsidian insight
**Behavior**:
- Loads all OOF predictions
- Computes correlation matrix (check signal dilution)
- Runs hill-climbing or weighted blend
- Submits to Kaggle
- Writes `insights/{exp_id}_full.md` with: CV/LB, blend weights, what worked, what didn't

---

## 5. Communication Protocol

### Mentor → Agent

Passed via Task tool prompt at dispatch time:
```
Task prompt includes:
- Working directory
- Feature version to use
- Competition metric
- What to write (output paths)
- Where to write status.json
```

### Agent → Mentor

Via filesystem (not SendMessage):
```
~/obsidian/ml-agent-memory/teams/{comp}/tasks/{agent}/status.json

{
  "state": "done",
  "version": "v1",
  "output_paths": {
    "oof": "~/shared/s6e5/oof/gkf/oof.npy",
    "test_preds": "~/shared/s6e5/oof/gkf/test_preds.npy"
  },
  "oof_auc": 0.9487,
  "elapsed_minutes": 40,
  "timestamp": "2026-05-24T12:30:00"
}
```

### Mentor Poll Loop

```python
# After dispatching agents, Mentor polls:
def poll_agents(agents, timeout_seconds=300):
    while True:
        for agent in agents:
            status = json.load(open(f"~/obsidian/ml-agent-memory/teams/s6e5/tasks/{agent}/status.json"))
            if status['state'] == 'done':
                results[agent] = status
            elif status['state'] == 'error':
                raise AgentError(f"{agent} failed: {status['error']}")
        if all_done(results, agents):
            break
        time.sleep(30)  # Poll every 30s

    return results
```

---

## 6. Experiment Lifecycle

```
[Experiment Planning]
       │
       ▼
┌──────────────────┐
│  Mentor reads    │  ← Obsidian, dashboard, past experiments
│  Obsidian         │
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Decide strategy │  ← GroupKFold? External data? Feature change?
│  (Mentor alone)  │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────┐
│  Critic reviews plan             │  ← "这个方向有证据吗？风险多大？"
│  (adversarial check)            │
└──────┬───────────────────────────┘
       │ critic verdict
       ▼
┌──────────────────────────────────┐
│  Dispatch via Task tool           │  ← Parallel
│  Feature Engineer                 │  ← Creates v{N}/features/
│  Model Trainer A (gkf)            │  ← Reads v{N}/features/
│  Model Trainer B (skf)            │  ← Reads v{N}/features/
└──────┬────────────┬─────────────────┘
       │          │
       ▼          ▼
   [Run independently]    [Mentor polls status.json every 30s]
       │          │
       ▼          ▼
┌──────────────────────────────────┐
│  All done?                       │  ← Check mtime of status.json
│  Critic reviews CV results       │  ← "这个提升是噪音还是真的？"
└──────┬────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  Ensemble Specialist              │  ← Reads oof/*/, blends, submits
│  Critic reviews submission       │  ← "格式正确吗？概率范围合理吗？"
└──────┬────────────────────────────┘
       │
       ▼
┌──────────────────┐
│  Write insights  │  ← ~/obsidian/ml-agent-memory/teams/s6e5/insights/
│  to Obsidian     │
└──────┬───────────┘
       ▼
   [Next Experiment]
```

---

## 7. Knowledge Persistence

### What each experiment writes

```
insights/{exp_id}_summary.md:
  ## {exp_name}
  - **CV/LB**: 0.9487 / 0.9512
  - **Key change**: GroupKFold by race_year
  - **Verdict**: No improvement over StratKFold
  - **Next**: Try external predictions integration
```

```
insights/{exp_id}_full.md:
  ## {exp_name} — Full Report
  ### Setup
  - Version: v2/features
  - CV Strategy: GroupKFold (n_splits=5)
  - Models: LGB + XGB + CB, 3 seeds each

  ### Results
  - GroupKFold OOF AUC: 0.9487
  - StratKFold OOF AUC: 0.9533
  - LB Score: 0.9512

  ### Analysis
  - CV gap between GKFold and StratKFold: 0.0046
  - GroupKFold is more conservative
  - LB closer to StratKFold → StratKFold is actually better aligned

  ### Key Insight
  - groupkfold_too_conservative_for_f1_pit_stop

  ### Next Action
  - Try external predictions (S6E4 strategy)
```

### What each agent writes

```
tasks/{agent}/status.json  — Machine-readable state
tasks/{agent}/log.md      — Human-readable execution log
```

---

## 8. Multi-Competition Scaling

Mentor can manage multiple competition agent teams:

```python
# Mentor decides which competition to advance
teams = ["s6e5", "s6e4", "store_sales"]

for competition in teams:
    obsidian_path = f"~/obsidian/ml-agent-memory/teams/{competition}"
    team_config = json.load(open(f"{obsidian_path}/team_config.json"))

    if team_config["active"]:
        # Dispatch tasks for this competition
        # Poll independently
        # Write insights to this competition's Obsidian folder
```

**Constraint**: Only one active experiment per competition at a time (avoids resource conflicts).

---

## 9. Agent Script Template

All agent scripts follow this pattern:

```python
"""
Agent: {name}
Competition: {competition}
Strategy: {strategy}
Version: {version}

Reads from: ~/shared/{comp}/{version}/features/
Writes to: ~/shared/{comp}/oof/{strategy}/
Status to: ~/obsidian/ml-agent-memory/teams/{comp}/tasks/{name}/status.json
"""

import json, time, numpy as np, os

COMP = os.environ.get("SHARED_DIR", f"~/shared/{comp}")
OBSIDIAN = os.environ.get("OBSIDIAN_DIR", "~/obsidian/ml-agent-memory")
STATUS_PATH = f"{OBSIDIAN}/teams/{comp}/tasks/{name}/status.json"

def write_status(state, **kwargs):
    status = {"state": state, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **kwargs}
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)

def main():
    write_status("running")

    # Load cached features
    X_train = np.load(f"{COMP}/{version}/features/X_train.npy")
    # ... train, predict, save ...

    write_status("done", oof_auc=final_auc, output_paths={...})

if __name__ == "__main__":
    main()
```

---

## 10. Implementation Priority

| Phase | What | Why |
|-------|------|-----|
| **P0** | Mentor dispatch + status polling | Core loop |
| **P1** | Feature Engineer versioned cache | Speed gains |
| **P2** | Model Trainer (parameterized) | Parallel training |
| **P3** | Ensemble Specialist | Automated submission |
| **P4** | Insight writing automation | Knowledge accumulation |
| **P5** | Multi-competition support | Scale |

---

## 11. Anti-Patterns to Avoid

1. **Mentor does training** — Bottleneck + loses coordination role
2. **Agents use SendMessage for state** — Messages lost on restart
3. **No versioning** — Can't reproduce or compare experiments
4. **Agents talk to each other directly** — Creates coupling, hard to debug
5. **Single shared DataFrame** — Concurrency corruption, use numpy files

---

## 12. Open Questions (Future)

- [ ] How does new agent (fresh session) find the team's latest state? → Mentor reads Obsidian at start
- [ ] What if Feature Engineer and Model Trainer have different feature expectations? → version_notes.md + JSON schema for feature_cols.json
- [ ] How to handle GPU memory limits with parallel trainers? → Reserve by model_type (LGB=CPU, XGB=GPU slot 1, CB=GPU slot 2)

---

*Document created via brainstorming. Approved for implementation.*