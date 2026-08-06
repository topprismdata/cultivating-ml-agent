"""
context_engineering package — 自适应上下文压缩(Token Economy)

agy 研究结论:即使 context 扩到 1M-2M tokens,全量投喂仍导致"Lost in the Middle"
和推理延迟。本模块提供三层压缩:
    1. Compressor — 长文档/摘要智能压缩
    2. LogTruncator — 训练日志智能截断
    3. TokenEconomy — 优先级感知的 token 预算管理
"""
from .compressor import ContextCompressor, CompressionLevel
from .log_truncator import LogTruncator, truncate_log
from .token_economy import TokenEconomy, BudgetPriority

__all__ = [
    "ContextCompressor", "CompressionLevel",
    "LogTruncator", "truncate_log",
    "TokenEconomy", "BudgetPriority",
]