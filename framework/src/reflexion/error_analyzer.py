"""
Error Analyzer — 智能分类 Python 错误 + 给出修复建议

根据 agy 研究:Reflexion 能自动修复 80%+ 错误,前提是错误能被精确分类。

分类:
    SYNTAX         — SyntaxError / IndentationError
    IMPORT         — ModuleNotFoundError / ImportError
    NAME           — NameError / AttributeError
    TYPE           — TypeError / ValueError
    SHAPE          — 维度不匹配
    MEMORY         — MemoryError / OOM
    LOGIC        — AssertionError / RuntimeError(业务逻辑)
    IO             — FileNotFoundError / PermissionError
    DATA_LEAKAGE   — 数据泄露(ML 特有)
    OVERFITTING    — 过拟合信号(ML 特有)
    UNKNOWN        — 其他
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ErrorCategory(Enum):
    SYNTAX = "syntax"
    IMPORT = "import"
    NAME = "name"
    TYPE = "type"
    SHAPE = "shape"
    MEMORY = "memory"
    LOGIC = "logic"
    IO = "io"
    DATA_LEAKAGE = "data_leakage"
    OVERFITTING = "overfitting"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class ErrorDiagnosis:
    """错误诊断结果"""
    category: ErrorCategory
    error_type: str       # 原始异常类名
    message: str          # 关键错误信息
    line_number: Optional[int] = None
    suggested_fix: str = ""
    related_skills: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 分类置信度

    def __repr__(self):
        return (
            f"ErrorDiagnosis({self.category.value}, "
            f"{self.error_type}, confidence={self.confidence:.2f})"
        )


# 错误模式库
ERROR_PATTERNS = [
    # (category, regex, suggested_fix, related_skills)
    (ErrorCategory.SYNTAX,
     re.compile(r"SyntaxError|IndentationError|invalid syntax"),
     "检查代码缩进和语法,可使用 IDE 高亮辅助",
     []),

    (ErrorCategory.IMPORT,
     re.compile(r"ModuleNotFoundError|ImportError|cannot import name"),
     "用 pip install 装包,或检查 import 路径拼写",
     []),

    (ErrorCategory.NAME,
     re.compile(r"NameError.*is not defined|AttributeError.*has no attribute"),
     "检查变量名/属性名拼写,或确认对象是否初始化",
     []),

    (ErrorCategory.SHAPE,
     re.compile(r"shapes?.*not aligned|dimension|mismatch|broadcast|reshape|(\d+).*vs.*(\d+)"),
     "检查张量/数组维度,用 .shape 调试;尝试 reshape/transpose",
     []),

    (ErrorCategory.MEMORY,
     re.compile(r"MemoryError|CUDA out of memory|OOM|RuntimeError.*out of memory"),
     "减小 batch_size / 序列长度;启用 gradient checkpointing;降低模型规模",
     ["gpu-readiness-assessment"]),

    (ErrorCategory.OVERFITTING,
     re.compile(r"val_loss.*increasing|train.*acc.*val.*acc|overfitting", re.IGNORECASE),
     "train vs val 差距过大:加 dropout / 减小模型 / 早停 / 数据增强",
     ["feature-engineering-saturation-detection"]),

    (ErrorCategory.DATA_LEAKAGE,
     re.compile(r"data leakage|target leakage|future.*feature", re.IGNORECASE),
     "检查特征是否包含未来信息;确保聚合只用 train 阶段数据",
     ["time-series-walk-forward-validation", "feature-engineering-saturation-detection"]),

    (ErrorCategory.TIMEOUT,
     re.compile(r"TimeoutError|timed out|timeout expired", re.IGNORECASE),
     "代码运行超时:减少数据量、优化算法、增加 timeout",
     []),

    (ErrorCategory.IO,
     re.compile(r"FileNotFoundError|PermissionError|OSError.*\[Errno"),
     "检查文件路径、权限、磁盘空间",
     []),

    (ErrorCategory.LOGIC,
     re.compile(r"AssertionError|RuntimeError|ValueError"),
     "检查业务逻辑:断言条件、循环边界、数据范围",
     []),

    (ErrorCategory.TYPE,
     re.compile(r"TypeError|unsupported operand|cannot concatenate"),
     "类型不匹配:用 isinstance 检查,显式 cast(str/int/float)",
     []),
]


class ErrorAnalyzer:
    """错误分析器"""

    def analyze(self, stderr_text: str, code: Optional = None) -> ErrorDiagnosis:
        """分析错误输出

        Args:
            stderr_text: Python 抛出的 stderr(包含 Traceback)
            code: 源代码(可选,用于上下文分析)

        Returns:
            ErrorDiagnosis
        """
        if not stderr_text:
            return ErrorDiagnosis(
                category=ErrorCategory.UNKNOWN,
                error_type="Empty",
                message="No error output",
            )

        # 1. 提取关键信息
        error_type, message, line_no = self._extract_traceback(stderr_text)

        # 2. 模式匹配分类
        for category, pattern, fix, skills in ERROR_PATTERNS:
            if pattern.search(stderr_text) or pattern.search(message):
                return ErrorDiagnosis(
                    category=category,
                    error_type=error_type,
                    message=message,
                    line_number=line_no,
                    suggested_fix=fix,
                    related_skills=skills,
                    confidence=0.85,
                )

        # 3. 兜底
        return ErrorDiagnosis(
            category=ErrorCategory.UNKNOWN,
            error_type=error_type or "Unknown",
            message=message or stderr_text[:500],
            line_number=line_no,
            suggested_fix="查看完整 Traceback 定位;复制错误信息搜索解决方案",
            confidence=0.3,
        )

    def _extract_traceback(self, stderr: str) -> tuple[Optional[str], str, Optional[int]]:
        """从 Traceback 中提取异常类型、消息、行号"""
        # Python Traceback 最后一行通常是: ErrorType: message
        lines = [l for l in stderr.splitlines() if l.strip()]
        if not lines:
            return None, "", None

        last = lines[-1]
        # 形如 "ValueError: too many values to unpack"
        m = re.match(r"^([A-Za-z_]+(?:Error|Exception|Warning))(?::\s*(.*))?$", last)
        if m:
            error_type = m.group(1)
            message = m.group(2) or ""
        else:
            error_type = None
            message = last

        # 提取行号: "File \"...\", line 42"
        line_match = re.search(r'File\s+"[^"]+",\s+line\s+(\d+)', stderr)
        line_no = int(line_match.group(1)) if line_match else None

        return error_type, message, line_no