"""
Skill Extractor — 从实验/对话提取候选 skill

agy 验证:自我封装新 skill 让 agent 持续进化。

提取规则(基于经验):
    1. 同一解决方案在 ≥2 个不同项目中使用 → 提取
    2. 失败模式被解决(尝试 ≥3 次才成功)→ 提取 anti-pattern
    3. 用户/agent 显式 "记住这个" → 强制提取
    4. 实验日志显示新 insight(非平凡观察)→ 提取为 principle

输出:SkillCandidate,供 Validator 验证后再注册。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path


@dataclass
class SkillCandidate:
    """候选 skill"""
    name: str
    description: str
    content: str
    type: str = "skill"  # skill / principle / anti-pattern
    tags: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)  # 证据引用
    importance: float = 0.5
    source: str = "auto-extract"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_markdown(self) -> str:
        """渲染为 SKILL.md 格式"""
        tags_str = ", ".join(self.tags) if self.tags else "auto"
        frontmatter = (
            "---\n"
            f"name: {self.name}\n"
            f"description: |\n"
            f"  {self.description}\n"
            f"type: {self.type}\n"
            f"importance: {self.importance}\n"
            f"tags: {tags_str}\n"
            f"source: {self.source}\n"
            f"created: {self.created_at}\n"
            "---\n\n"
        )
        body = f"# {self.name}\n\n{self.content}\n"
        if self.evidence:
            body += "\n## Evidence\n\n"
            for ev in self.evidence:
                body += f"- {ev}\n"
        return frontmatter + body


# 内置提取规则
EXTRACTION_TRIGGERS = [
    (re.compile(r"(?:we should|remember that|note:|key insight:?|lesson learned:?)\s*(.+?)(?:\.|$)", re.IGNORECASE),
     "lesson"),
    (re.compile(r"(?:don't|never|avoid)\s+(.+?)(?:\.|$)", re.IGNORECASE),
     "anti-pattern"),
    (re.compile(r"(?:always|make sure|ensure|remember to)\s+(.+?)(?:\.|$)", re.IGNORECASE),
     "principle"),
]


class SkillExtractor:
    """从文本/对话提取候选 skill"""

    def __init__(self, min_evidence: int = 1):
        self.min_evidence = min_evidence

    def extract_from_text(self, text: str, source: str = "conversation"
                          ) -> List[SkillCandidate]:
        """从单段文本提取候选"""
        candidates: List[SkillCandidate] = []
        for pattern, kind in EXTRACTION_TRIGGERS:
            for m in pattern.finditer(text):
                content = m.group(1).strip()
                if len(content) < 20 or len(content) > 500:
                    continue
                # 计算 importance(基于关键词)
                importance = self._score_importance(content, kind)
                if importance < 0.4:
                    continue
                candidates.append(SkillCandidate(
                    name=self._make_name(content),
                    description=f"[Auto-extracted from {source}]: {content[:100]}",
                    content=f"## Context\n{content}",
                    type=kind if kind != "lesson" else "skill",
                    evidence=[f"Source: {source}"],
                    importance=importance,
                    source=f"auto-extract:{source}",
                ))
        return candidates

    def extract_from_experiment_log(self, log_path: Path) -> List[SkillCandidate]:
        """从实验日志(LLM.md 格式)提取"""
        if not log_path.exists():
            return []
        text = log_path.read_text(encoding="utf-8", errors="replace")
        candidates = self.extract_from_text(text, source=f"experiment:{log_path.name}")

        # 特殊规则: 提升关键洞察的 importance
        for c in candidates:
            if any(kw in c.content.lower() for kw in [
                "data leakage", "overfitting", "cv-lb gap",
                "walk-forward", "saturation", "ensemble",
            ]):
                c.importance = min(0.9, c.importance + 0.2)
                c.tags.append("ml-critical")
        return candidates

    def extract_from_experiments_md(self, experiments_md: str
                                    ) -> List[SkillCandidate]:
        """从大实验记录(EXPERIMENTS.md 风格)提取"""
        candidates = self.extract_from_text(experiments_md, source="experiments")
        # 按 section 聚合:同一 section 内的发现合并
        # 简化: 保留 top-N
        candidates.sort(key=lambda c: c.importance, reverse=True)
        return candidates[:20]

    # ---- helpers ----

    def _score_importance(self, content: str, kind: str) -> float:
        """打分 importance"""
        score = 0.5
        # 类型加成
        if kind == "anti-pattern":
            score += 0.1
        elif kind == "principle":
            score += 0.15
        # 关键词加成
        keywords_high = [
            "always", "never", "critical", "data leakage",
            "overfitting", "validation", "saturation",
        ]
        keywords_med = ["should", "important", "key", "remember"]
        text_lower = content.lower()
        if any(kw in text_lower for kw in keywords_high):
            score += 0.2
        if any(kw in text_lower for kw in keywords_med):
            score += 0.1
        return min(1.0, score)

    def _make_name(self, content: str) -> str:
        """生成 kebab-case skill 名"""
        words = re.findall(r"[a-z]+|[\u4e00-\u9fff]+", content.lower())
        slug = "-".join(words[:5])[:50]
        return f"auto-{slug}" if slug else "auto-skill"