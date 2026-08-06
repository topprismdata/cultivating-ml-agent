"""
Reflexion Loop — 错误 → 反思 → 重试 的闭环

agy 验证:Reflexion + Sandbox 能修复 80%+ 运行时错误。

工作流程:
    1. 执行代码(在 sandbox)
    2. 如果成功 → 返回
    3. 如果失败 → 调用 ErrorAnalyzer 诊断
    4. 用 LLM(或规则)生成修复建议
    5. 把错误 + 诊断 + 修复建议作为 context
    6. 重新生成代码
    7. 最多 N 轮

使用:
    loop = ReflexionLoop(sandbox, analyzer, llm_client=...)
    result = loop.run(initial_code, max_attempts=5)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .sandbox import CodeSandbox, ExecutionResult
from .error_analyzer import ErrorAnalyzer, ErrorDiagnosis, ErrorCategory


@dataclass
class Attempt:
    """一次尝试的记录"""
    code: str
    execution: ExecutionResult
    diagnosis: Optional[ErrorDiagnosis] = None
    duration_sec: float = 0.0


@dataclass
class ReflexionResult:
    """整个反思循环的最终结果"""
    success: bool
    final_code: str
    final_output: str
    attempts: List[Attempt] = field(default_factory=list)
    total_duration_sec: float = 0.0
    final_diagnosis: Optional[ErrorDiagnosis] = None

    @property
    def attempts_count(self) -> int:
        return len(self.attempts)

    @property
    def last_error_category(self) -> Optional[ErrorCategory]:
        if self.attempts and self.attempts[-1].diagnosis:
            return self.attempts[-1].diagnosis.category
        return None


# 类型: LLM 调用函数签名
LLMCallable = Callable[[str], str]  # input(prompt) → output(revised code)


class ReflexionLoop:
    """Reflexion 自我修正循环"""

    def __init__(self,
                 sandbox: Optional[CodeSandbox] = None,
                 analyzer: Optional[ErrorAnalyzer] = None,
                 llm_fix: Optional[LLMCallable] = None):
        """
        Args:
            sandbox: 代码执行沙箱
            analyzer: 错误分析器
            llm_fix: LLM 修复函数,接收 (code + error + diagnosis),返回修正后的 code
                    可选;若不提供,只能记录错误,无法自动修复
        """
        self.sandbox = sandbox or CodeSandbox()
        self.analyzer = analyzer or ErrorAnalyzer()
        self.llm_fix = llm_fix

    def run(self, code: str, max_attempts: int = 3) -> ReflexionResult:
        """运行 Reflexion 循环

        Args:
            code: 初始代码
            max_attempts: 最大尝试次数(包括第一次)
        """
        attempts: List[Attempt] = []
        current_code = code
        start = time.time()
        final_success = False
        final_output = ""

        for attempt_idx in range(max_attempts):
            t0 = time.time()
            exec_result = self.sandbox.execute(current_code)
            exec_result.duration_sec = time.time() - t0

            diagnosis = None
            if not exec_result.success:
                diagnosis = self.analyzer.analyze(exec_result.stderr, current_code)
                # 尝试用 LLM 修复
                if self.llm_fix:
                    current_code = self.llm_fix(self._build_reflexion_prompt(
                        current_code, exec_result, diagnosis
                    ))

            attempts.append(Attempt(
                code=current_code if attempt_idx > 0 else code,
                execution=exec_result,
                diagnosis=diagnosis,
                duration_sec=exec_result.duration_sec,
            ))

            if exec_result.success:
                final_success = True
                final_output = exec_result.stdout
                break

        return ReflexionResult(
            success=final_success,
            final_code=current_code,
            final_output=final_output,
            attempts=attempts,
            total_duration_sec=time.time() - start,
            final_diagnosis=attempts[-1].diagnosis if attempts else None,
        )

    def _build_reflexion_prompt(self, code: str,
                                 exec_result: ExecutionResult,
                                 diagnosis: ErrorDiagnosis) -> str:
        """构建给 LLM 的反思 prompt"""
        return f"""[Reflexion: Previous Attempt Failed]

Code:
```python
{code}
```

Execution Result:
- Error Type: {diagnosis.error_type}
- Category: {diagnosis.category.value}
- Message: {diagnosis.message}
- Line: {diagnosis.line_number}
- Suggested Fix: {diagnosis.suggested_fix}
- Related Skills: {', '.join(diagnosis.related_skills) or '(none)'}

Full Traceback:
```
{exec_result.stderr[:2000]}
```

Please provide a corrected version of the code that fixes the above error.
Output ONLY the corrected Python code, no explanations."""


# 便捷函数:无 LLM 的纯执行版本
def execute_with_analysis(code: str,
                          sandbox: Optional[CodeSandbox] = None,
                          analyzer: Optional[ErrorAnalyzer] = None
                          ) -> tuple[ExecutionResult, Optional[ErrorDiagnosis]]:
    """单次执行 + 分析(不重试)"""
    sb = sandbox or CodeSandbox()
    az = analyzer or ErrorAnalyzer()
    result = sb.execute(code)
    diagnosis = az.analyze(result.stderr, code) if not result.success else None
    return result, diagnosis