"""
Context Compressor — 智能压缩长文本到关键信息

策略:
    LIGHT    — 去除空白/重复,保留 90%+ 原文
    MEDIUM   — 提取关键句子(TF-IDF 启发),保留 ~50%
    AGGRESSIVE — 摘要式压缩(用 LLM 或规则),保留 ~20%

升级点:接 LLM 后用 Anthropic prompt caching
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional


class CompressionLevel(Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    AGGRESSIVE = "aggressive"


@dataclass
class CompressionResult:
    original: str
    compressed: str
    level: CompressionLevel
    original_chars: int
    compressed_chars: int

    @property
    def ratio(self) -> float:
        return self.compressed_chars / max(self.original_chars, 1)

    def __repr__(self):
        return (
            f"CompressionResult({self.level.value}: "
            f"{self.original_chars} → {self.compressed_chars} chars, "
            f"ratio={self.ratio:.2%})"
        )


class ContextCompressor:
    """智能压缩长文本"""

    def __init__(self, max_chars_per_block: int = 4000):
        self.max_chars_per_block = max_chars_per_block

    def compress(self, text: str, level: CompressionLevel = CompressionLevel.MEDIUM,
                 focus_query: Optional = None) -> CompressionResult:
        """压缩文本

        Args:
            text: 输入文本
            level: 压缩级别
            focus_query: 聚焦 query(MEDIUM/AGGRESSIVE 时,优先保留相关内容)
        """
        if level == CompressionLevel.LIGHT:
            compressed = self._compress_light(text)
        elif level == CompressionLevel.MEDIUM:
            compressed = self._compress_medium(text, focus_query)
        else:
            compressed = self._compress_aggressive(text, focus_query)

        return CompressionResult(
            original=text,
            compressed=compressed,
            level=level,
            original_chars=len(text),
            compressed_chars=len(compressed),
        )

    # ---- 三档压缩策略 ----

    def _compress_light(self, text: str) -> str:
        """轻压缩:去多余空白/重复行"""
        lines = text.splitlines()
        seen = set()
        unique = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in seen:
                continue
            seen.add(stripped)
            unique.append(line)
        # 折叠多个空行
        result = "\n".join(unique)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result

    def _compress_medium(self, text: str, focus_query: Optional = None) -> str:
        """中等压缩:TF + 位置权重 + 聚焦加权"""
        sentences = self._split_sentences(text)
        if len(sentences) <= 5:
            return "\n".join(sentences)

        # 计算每句得分
        freq = self._term_frequency(sentences)
        focus_terms = set(self._tokenize(focus_query)) if focus_query else set()

        scored = []
        n = len(sentences)
        for i, sent in enumerate(sentences):
            terms = set(self._tokenize(sent))
            if not terms:
                continue
            # TF score
            tf = sum(freq.get(t, 0) for t in terms) / len(terms)
            # 位置权重(开头/结尾加权)
            position_weight = 1.5 if i < 3 or i >= n - 3 else 1.0
            # 聚焦加权
            focus_weight = 2.0 if focus_terms and (terms & focus_terms) else 1.0
            # 长度惩罚(过短/过长)
            len_penalty = min(1.0, len(sent) / 100)
            score = tf * position_weight * focus_weight * len_penalty
            scored.append((score, i, sent))

        # 保留 top 50%
        scored.sort(key=lambda x: x[0], reverse=True)
        keep = max(3, len(scored) // 2)
        kept = sorted(scored[:keep], key=lambda x: x[1])
        return "\n".join(s[2] for s in kept)

    def _compress_aggressive(self, text: str, focus_query: Optional = None) -> str:
        """激进压缩:保留标题/关键句/数据点"""
        lines = text.splitlines()
        kept: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # 标题/列表项/数据 → 保留
            if (stripped.startswith(("#", "-", "*", "1.", "2.")) or
                re.match(r"^\*\*[^*]+\*\*:", stripped) or  # **Key**: value
                re.search(r"\d+\.\d+", stripped)):  # 数字
                kept.append(line)
                continue
            # 长段落保留首句
            if len(stripped) > 200:
                first_sentence = re.split(r"[.!?。]", stripped)[0]
                if focus_query and any(t in first_sentence.lower() for t in focus_query.lower().split()):
                    kept.append(first_sentence + ".")
                continue
            # 短句聚焦保留
            if focus_query:
                if any(t in stripped.lower() for t in focus_query.lower().split()):
                    kept.append(line)
            # 数字行/数据点
            elif re.search(r"\d", stripped):
                kept.append(line)

        result = "\n".join(kept)
        # 兜底:仍太长就只保留前 N 字符
        if len(result) > self.max_chars_per_block:
            result = result[:self.max_chars_per_block] + "\n...(truncated)"
        return result

    # ---- helpers ----

    def _split_sentences(self, text: str) -> List[str]:
        """按句子切分(中英文混合)"""
        text = re.sub(r"\s+", " ", text)
        parts = re.split(r"(?<=[.!?。])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").lower()
        return re.findall(r"[a-z][a-z0-9_]+|[\u4e00-\u9fff]", text)

    def _term_frequency(self, sentences: List[str]) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for sent in sentences:
            for term in set(self._tokenize(sent)):
                freq[term] = freq.get(term, 0) + 1
        return freq