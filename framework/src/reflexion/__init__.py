"""
reflexion package — 运行时自我反思(Reflexion + Sandbox 闭环)

agy 研究结论:Reflexion + Code Sandbox 是 2025-2026 ML agent 拿 Kaggle 牌的关键。
本模块:
    1. Sandbox — 子进程执行 Python 代码,捕获 stderr
    2. ErrorAnalyzer — 智能分类错误类型
    3. ReflexionLoop — 错误 → 反思 → 重试的循环
"""
from .sandbox import CodeSandbox, ExecutionResult
from .error_analyzer import ErrorAnalyzer, ErrorCategory, ErrorDiagnosis
from .loop import ReflexionLoop, ReflexionResult, execute_with_analysis

__all__ = [
    "CodeSandbox", "ExecutionResult",
    "ErrorAnalyzer", "ErrorCategory", "ErrorDiagnosis",
    "ReflexionLoop", "ReflexionResult", "execute_with_analysis",
]