"""
Token Economy — 优先级感知的 token 预算管理

类比:操作系统内存管理。
    HIGH   预算 → 必须塞进 context(当前任务关键信息)
    MEDIUM 预算 → 尽量塞,塞不下就压缩
    LOW    预算 → 仅在还有余量时塞入

每次 LLM 调用前,根据 budget 选择哪些 context 进入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


class BudgetPriority(IntEnum):
    """优先级(数字越大越优先)"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ContextBlock:
    """一段待塞入 context 的内容"""
    content: str
    priority: BudgetPriority
    label: str = ""
    estimated_tokens: int = 0

    def __post_init__(self):
        if self.estimated_tokens == 0:
            # 粗估:英文 4 chars/token,中文 1.5 chars/token
            self.estimated_tokens = self._estimate_tokens(self.content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # 简化估算:每 4 个字符约 1 token
        return max(1, len(text) // 4)


@dataclass
class TokenBudget:
    """一次调用的总预算"""
    max_tokens: int = 8000
    reserved_for_response: int = 2000  # 给模型回复留的空间

    @property
    def available(self) -> int:
        return self.max_tokens - self.reserved_for_response


class TokenEconomy:
    """优先级感知的 token 分配器"""

    def __init__(self, budget: Optional[TokenBudget] = None):
        self.budget = budget or TokenBudget()
        self.blocks: List[ContextBlock] = []

    def add(self, content: str, priority: BudgetPriority,
            label: str = "") -> "TokenEconomy":
        """添加一块 context"""
        self.blocks.append(ContextBlock(
            content=content, priority=priority, label=label,
        ))
        return self

    def add_high(self, content: str, label: str = "") -> "TokenEconomy":
        return self.add(content, BudgetPriority.HIGH, label)

    def add_critical(self, content: str, label: str = "") -> "TokenEconomy":
        return self.add(content, BudgetPriority.CRITICAL, label)

    def assemble(self, auto_compress: bool = True) -> str:
        """按优先级组装 context,预算超限时自动压缩低优先级块

        Returns:
            组装好的 context 字符串
        """
        available = self.budget.available
        # 按优先级降序
        sorted_blocks = sorted(self.blocks, key=lambda b: -b.priority.value)

        parts: List[str] = []
        used = 0

        for block in sorted_blocks:
            if used + block.estimated_tokens <= available:
                parts.append(self._format_block(block))
                used += block.estimated_tokens
            elif auto_compress and block.priority.value >= BudgetPriority.MEDIUM.value:
                # 中高优先级:压缩后塞入
                remaining = available - used
                compressed = self._compress_to_fit(block.content, remaining)
                if compressed:
                    parts.append(self._format_block(block, note=f"compressed to {len(compressed)} chars"))
                    used += len(compressed) // 4
                else:
                    parts.append(f"# [SKIPPED: {block.label or 'block'} — over budget]")
            else:
                # 低优先级:跳过
                parts.append(f"# [SKIPPED: {block.label or 'block'} — low priority]")

        header = (
            f"# Context ({used} tokens used / {available} available)\n"
            f"# Budget: max={self.budget.max_tokens}, "
            f"reserved_for_response={self.budget.reserved_for_response}\n\n"
        )
        return header + "\n\n".join(parts)

    def stats(self) -> Dict:
        """返回 budget 使用统计"""
        used = sum(b.estimated_tokens for b in self.blocks)
        return {
            "blocks": len(self.blocks),
            "estimated_tokens": used,
            "available_tokens": self.budget.available,
            "over_budget": used > self.budget.available,
            "by_priority": {
                p.name: sum(b.estimated_tokens for b in self.blocks if b.priority == p)
                for p in BudgetPriority
            },
        }

    def _compress_to_fit(self, content: str, max_chars: int) -> str:
        """压缩到不超过 max_chars"""
        if len(content) <= max_chars:
            return content
        # 简单截断(可升级为调用 ContextCompressor)
        ratio = max_chars / len(content)
        if ratio > 0.5:
            return content[:max_chars] + "\n..."
        else:
            # 激进:保留首尾
            half = max_chars // 2 - 3
            return content[:half] + "\n...\n" + content[-half:]

    def _format_block(self, block: ContextBlock, note: str = "") -> str:
        label = block.label or "block"
        priority = block.priority.name
        if note:
            return f"## [{label}] ({priority}, {note})\n{block.content}"
        return f"## [{label}] ({priority}, ~{block.estimated_tokens} tokens)\n{block.content}"