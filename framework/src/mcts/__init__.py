"""
mcts package — Monte Carlo Tree Search for ML Pipeline

agy 建议:MCTS 适合 Top 1% 冲刺场景,多分支探索 + UCB 剪枝

典型用法:
    探索多个 ML pipeline 分支:
        branch 1: LightGBM + walk-forward
        branch 2: XGBoost + K-fold
        branch 3: Neural Net + ensemble
    MCTS 根据 validation score 动态选最有前途的分支
"""
from .node import PipelineNode
from .search import MCTSSearch, SearchResult, grid_expansion, identity_evaluator

__all__ = [
    "PipelineNode",
    "MCTSSearch", "SearchResult",
    "grid_expansion", "identity_evaluator",
]