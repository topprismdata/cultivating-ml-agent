"""
Pipeline Node — MCTS 搜索树节点

每个节点代表一个 ML pipeline 配置(数据 / 特征 / 模型 / 超参 / 验证)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PipelineNode:
    """MCTS 树节点"""
    id: str
    config: Dict = field(default_factory=dict)   # pipeline 配置
    parent: Optional["PipelineNode"] = None
    children: List["PipelineNode"] = field(default_factory=list)
    visits: int = 0
    score_sum: float = 0.0
    score: float = 0.0           # 最近一次评估的分数
    is_terminal: bool = False
    depth: int = 0

    @property
    def avg_score(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.score_sum / self.visits

    def ucb1(self, exploration_weight: float = 1.414) -> float:
        """UCB1 公式: exploitation + exploration"""
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.avg_score
        exploitation = self.avg_score
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def add_child(self, child: "PipelineNode") -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def best_child(self, exploration_weight: float = 0.0) -> Optional["PipelineNode"]:
        """选择 best 子节点(0 探索 = 纯 exploitation)"""
        if not self.children:
            return None
        return max(
            self.children,
            key=lambda c: c.ucb1(exploration_weight)
            if exploration_weight > 0 else c.avg_score,
        )

    def __repr__(self):
        cfg_str = ", ".join(f"{k}={v}" for k, v in list(self.config.items())[:3])
        return (
            f"PipelineNode(id={self.id}, visits={self.visits}, "
            f"avg_score={self.avg_score:.3f}, cfg=({cfg_str}))"
        )