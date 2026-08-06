"""
Skill Validator — 验证候选 skill 质量

agy 验证:Self-Evolving Skills 需要质量门控,否则会污染库。

检查:
    1. 模板合规(frontmatter + 必要字段)
    2. 长度合理(20-2000 字符)
    3. 与现有 skills 不重复(similarity >0.85 视为重复)
    4. description 含 "Use when..." 触发器
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .extractor import SkillCandidate


class ValidationVerdict(Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


@dataclass
class ValidationResult:
    verdict: ValidationVerdict
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    similarity_to_existing: float = 0.0  # 0-1

    def __repr__(self):
        return f"ValidationResult({self.verdict.value}, {len(self.issues)} issues)"


# 必要 frontmatter 字段
REQUIRED_FIELDS = {"name", "description"}


class SkillValidator:
    """Skill 验证器"""

    def __init__(self, min_content_length: int = 20, max_content_length: int = 2000):
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length

    def validate(self, candidate: SkillCandidate,
                 existing_skills: List[str] = None) -> ValidationResult:
        """验证候选 skill

        Args:
            candidate: 候选 skill
            existing_skills: 现有 skill 的描述列表(用于去重)
        """
        issues: List[str] = []
        suggestions: List[str] = []

        # 1. 模板合规
        md = candidate.to_markdown()
        if not self._has_frontmatter(md):
            issues.append("Missing YAML frontmatter")
        else:
            missing = self._missing_fields(md)
            if missing:
                issues.append(f"Missing required fields: {missing}")

        # 2. 长度
        if len(candidate.content) < self.min_content_length:
            issues.append(f"Content too short ({len(candidate.content)} chars)")
        if len(candidate.content) > self.max_content_length:
            issues.append(f"Content too long ({len(candidate.content)} chars)")

        # 3. description 触发器
        if not candidate.description.startswith("Use when"):
            suggestions.append(
                'Description should start with "Use when..." for proper triggering'
            )

        # 4. 与现有 skills 相似度
        max_similarity = 0.0
        if existing_skills:
            for existing in existing_skills:
                sim = self._similarity(candidate.content, existing)
                max_similarity = max(max_similarity, sim)

        if max_similarity > 0.85:
            issues.append(f"Too similar to existing skill (similarity={max_similarity:.2f})")

        # 判定
        if any("Missing" in i or "Too similar" in i or "too short" in i for i in issues):
            verdict = ValidationVerdict.REJECT
        elif issues:
            verdict = ValidationVerdict.REVISE
        else:
            verdict = ValidationVerdict.APPROVE

        return ValidationResult(
            verdict=verdict,
            issues=issues,
            suggestions=suggestions,
            similarity_to_existing=max_similarity,
        )

    # ---- helpers ----

    def _has_frontmatter(self, md: str) -> bool:
        return md.startswith("---\n") and "\n---\n" in md

    def _missing_fields(self, md: str) -> List[str]:
        m = re.search(r"^---\n(.*?)\n---", md, re.DOTALL)
        if not m:
            return list(REQUIRED_FIELDS)
        fields = set()
        for line in m.group(1).splitlines():
            if ":" in line:
                fields.add(line.split(":", 1)[0].strip())
        return list(REQUIRED_FIELDS - fields)

    def _similarity(self, text1: str, text2: str) -> float:
        """简单 Jaccard 相似度"""
        tokens1 = set(self._tokenize(text1))
        tokens2 = set(self._tokenize(text2))
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z][a-z0-9_]+", text.lower())