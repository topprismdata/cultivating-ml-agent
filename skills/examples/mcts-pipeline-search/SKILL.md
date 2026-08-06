---
name: mcts-pipeline-search
description: |
  Use when exploring hyperparameter combinations for ML models, when doing Top-1% push on Kaggle, when grid search is too expensive, or when needing systematic exploration of pipeline variants. Triggers for neural architecture search, ensemble weight tuning, or feature subset selection.
---

# MCTS Pipeline Search

## Context
Grid/random search wastes budget exploring dead branches. MCTS (Monte Carlo Tree Search) uses UCB1 to balance exploitation (best so far) and exploration (uncertain branches). agy recommends MCTS as P2 for Top-1% scenarios: used by AIDE for ML pipeline exploration, AlphaGo for game trees.

The core insight: **UCB1 prevents both "stuck on suboptimal" and "random walk"**, getting best of both.

## Guidance

### Basic Usage

```python
from framework.src.mcts import MCTSSearch, grid_expansion

# 网格定义
grid = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 500, 1000],
}

# 评估函数: 真实 CV 或 proxy
def evaluator(config):
    return train_and_score(config)  # 返回 0-1 score

# 扩展: 网格生成子配置
expansion = grid_expansion(grid)

# 搜索
search = MCTSSearch(evaluator, expansion, max_depth=3)
result = search.search(initial_config={"num_leaves": 31}, iterations=100)

print(f"Best: {result.best_node.config}")
print(f"Score: {result.best_score:.3f}")
print(f"Tree size: {result.tree_size}")
```

### Custom Expansion (Beyond Grid)

```python
def my_expansion(config):
    """对当前 config 生成邻居(参数 ±10%)"""
    neighbors = []
    for delta in [-0.1, 0.1]:
        new = dict(config)
        new["learning_rate"] = max(0.001, config["learning_rate"] * (1 + delta))
        neighbors.append(new)
    return neighbors
```

### Pipeline Search DAG

```python
# 探索不同模型 pipeline 组合
def pipeline_evaluator(config):
    # config 包含: model_type, features, ensemble_method
    if config["model_type"] == "lgbm":
        return train_lgbm(config)
    elif config["model_type"] == "xgb":
        return train_xgb(config)
    elif config["model_type"] == "ensemble":
        return train_ensemble(config)

def pipeline_expansion(config):
    """从当前 pipeline 探索下一个组件"""
    next_steps = []
    for model in ["lgbm", "xgb", "catboost", "ensemble"]:
        new = dict(config)
        new["model_type"] = model
        next_steps.append(new)
    return next_steps

search = MCTSSearch(pipeline_evaluator, pipeline_expansion, max_depth=5)
result = search.search({"features": "basic"}, iterations=200)
```

## Why This Matters

| Grid Search | MCTS |
|---|---|
| O(grid_size) — slow | O(iterations × branching) — adaptive |
| Wastes time on bad branches | UCB1 prunes via visits |
| No learning between configs | Backprop updates scores |
| Single shot | Iterative refinement |

agy: "MCTS Pipeline Search suitable for **Top-1% competition push**; expensive but can find non-obvious winners."

## When to Apply

### When to Use
- Hyperparameter search with 4+ dimensions
- ML pipeline architecture selection
- Ensemble weight tuning
- Final push on leaderboard (when baseline works)
- When grid is too large for brute force

### When NOT to Use
- Small grids (just enumerate)
- When evaluation is super expensive (MCTS = many evals)
- Initial baseline (start simple)
- Real-time applications (search takes minutes-hours)

## Notes
- **Iteration count**: 50-200 typical; more = better but slower
- **UCB1 weight**: 1.414 (sqrt(2)) default; increase for more exploration
- **Depth limit**: 3-5 typical; deeper = combinatorial explosion
- **Combine with warm start**: seed MCTS with known good config
- **Cache evaluations**: same config = same score (memoize)
- See also: `ml-sweet-spot`, `optuna-integration` (alternative: Bayesian)

## References
- Implementation: `framework/src/mcts/`
- Inspired by: AIDE ML pipeline search, MLZero MCTS (NeurIPS 2025)
- Alternative: Optuna (Bayesian), Hyperopt (random search)
- Algorithm: UCB1 (Auer et al. 2002), applied to ML by AIDE