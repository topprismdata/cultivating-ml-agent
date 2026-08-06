"""
召回策略 — 多策略融合(MemGPT 风格的"contextual retrieval")

未来扩展点:
    - 可升级为 embedding 召回(用 sentence-transformers)
    - 可加 LLM reranker(用 Claude 做最后排序)
    - 可加时间衰减(老经验自然降权)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from .hierarchy import MemoryItem, ArchivalStore


@dataclass
class RecallConfig:
    """召回配置 — 决定如何融合多种策略"""
    use_keyword: bool = True
    use_recency: bool = True
    use_importance: bool = True
    use_access_frequency: bool = True
    recency_decay_days: float = 30.0   # 30 天前记忆半衰
    final_top_k: int = 5


def score_item(item: MemoryItem, query_terms: set, config: RecallConfig) -> float:
    """综合评分 = keyword * importance * recency * frequency"""
    score = 0.0

    if config.use_keyword:
        c_terms = set(ArchivalStore._tokenize(item.content))
        if not c_terms:
            return 0.0
        overlap = len(query_terms & c_terms)
        if overlap == 0:
            return 0.0
        score += overlap / math.sqrt(len(query_terms) * len(c_terms))

    if config.use_importance:
        # importance 0.5 → 1.0x, 1.0 → 1.5x
        score *= (1.0 + (item.importance - 0.5))

    if config.use_recency:
        try:
            last = datetime.fromisoformat(item.last_accessed or item.created_at)
            age_days = (datetime.now() - last).total_seconds() / 86400
            decay = math.exp(-age_days / config.recency_decay_days)
            score *= decay
        except Exception:
            pass

    if config.use_access_frequency:
        # 越常用越排前,但平滑
        score *= (1 + math.log1p(item.access_count) * 0.2)

    return score


def multi_strategy_recall(store: ArchivalStore, query: str,
                          config: Optional[RecallConfig] = None,
                          k: Optional[int] = None,
                          type_filter: Optional[str] = None) -> List[MemoryItem]:
    """多策略召回:keyword + importance + recency + frequency"""
    cfg = config or RecallConfig()
    final_k = k or cfg.final_top_k
    q_terms = set(ArchivalStore._tokenize(query))
    if not q_terms:
        return []

    scored: List[tuple[float, MemoryItem]] = []
    for item in store.items.values():
        if type_filter and item.type != type_filter:
            continue
        s = score_item(item, q_terms, cfg)
        if s > 0:
            scored.append((s, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:final_k]]