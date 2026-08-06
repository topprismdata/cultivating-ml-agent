"""
MCTS Search — 蒙特卡洛树搜索 ML Pipeline

agy 验证:Tree-of-Thought / MCTS 适合 Top 1% 冲刺场景。

简化实现:用 validation score 作为 simulation 结果(而非完整训练)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .node import PipelineNode


# 评估函数签名: config → score (0-1,越大越好)
EvaluatorFunc = Callable[[Dict], float]


# Pipeline 修改函数签名: config → List[config] (生成候选子节点)
ExpansionFunc = Callable[[Dict], List[Dict]]


@dataclass
class SearchResult:
    """MCTS 搜索最终结果"""
    best_node: PipelineNode
    iterations: int
    best_score: float
    tree_size: int

    def __repr__(self):
        return (
            f"SearchResult(best={self.best_node.id}, "
            f"score={self.best_score:.3f}, "
            f"iterations={self.iterations}, "
            f"tree_size={self.tree_size})"
        )


class MCTSSearch:
    """MCTS 搜索器"""

    def __init__(self, evaluator: EvaluatorFunc, expansion: ExpansionFunc,
                 exploration_weight: float = 1.414,
                 max_depth: int = 5):
        """
        Args:
            evaluator: 评估函数(config → score)
            expansion: 扩展函数(config → List[config])
            exploration_weight: UCB1 探索权重
            max_depth: 最大深度
        """
        self.evaluator = evaluator
        self.expansion = expansion
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth

    def search(self, initial_config: Dict,
               iterations: int = 50,
               root_id: str = "root") -> SearchResult:
        """运行 MCTS 搜索

        Args:
            initial_config: 初始 pipeline 配置
            iterations: MCTS 迭代次数
            root_id: 根节点 ID
        """
        root = PipelineNode(id=root_id, config=initial_config)
        root.visits = 1  # 根节点先访问一次

        for _ in range(iterations):
            # 1. Selection: 从根沿 UCB1 选到叶子
            node = self._select(root)

            # 2. Expansion: 叶子扩展子节点(深度未超限)
            if not node.is_terminal and node.depth < self.max_depth:
                self._expand(node)

            # 3. Simulation: 评估新节点
            if node.children:
                # 选刚扩展的第一个子节点 simulation
                sim_node = node.children[-1]
            else:
                sim_node = node

            score = self.evaluator(sim_node.config)

            # 4. Backpropagation: 沿 parent路径回溯更新
            self._backpropagate(sim_node, score)

        # 找 best
        best = max(
            [n for n in self._iter_nodes(root) if n.visits > 0],
            key=lambda n: n.avg_score,
            default=root,
        )
        return SearchResult(
            best_node=best,
            iterations=iterations,
            best_score=best.avg_score,
            tree_size=sum(1 for _ in self._iter_nodes(root)),
        )

    def _select(self, node: PipelineNode) -> PipelineNode:
        """Selection: 沿 UCB1 选子节点直到叶子"""
        while node.children and not node.is_terminal:
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                return random.choice(unvisited)
            node = node.best_child(self.exploration_weight)
        return node

    def _expand(self, node: PipelineNode) -> None:
        """扩展节点:生成子节点候选"""
        candidate_configs = self.expansion(node.config)
        for i, cfg in enumerate(candidate_configs):
            child = PipelineNode(
                id=f"{node.id}.{i}",
                config=cfg,
                is_terminal=(node.depth + 1 >= self.max_depth),
            )
            node.add_child(child)

    def _backpropagate(self, node: PipelineNode, score: float) -> None:
        """回溯更新 visits + score_sum"""
        current: Optional[PipelineNode] = node
        while current is not None:
            current.visits += 1
            current.score_sum += score
            current.score = score
            current = current.parent

    def _iter_nodes(self, root: PipelineNode):
        """DFS 遍历所有节点"""
        yield root
        for child in root.children:
            yield from self._iter_nodes(child)


# ---- 简化使用示例的工具函数 ----

def grid_expansion(grid: Dict[str, List], base: Optional[Dict] = None
                   ) -> ExpansionFunc:
    """网格搜索式扩展:对每个参数取一个值,组合生成"""
    import itertools
    keys = list(grid.keys())
    values = list(grid.values())
    base = base or {}

    def expand(config: Dict) -> List[Dict]:
        results = []
        for combo in itertools.product(*values):
            child = dict(base)
            child.update(config)
            for k, v in zip(keys, combo):
                child[k] = v
            results.append(child)
        return results
    return expand


def identity_evaluator() -> EvaluatorFunc:
    """默认评估器:用 config 自身的某个 'score' 字段"""
    def evaluate(config: Dict) -> float:
        return float(config.get("score", 0.0))
    return evaluate