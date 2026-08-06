"""
Code Sandbox — 安全执行 Python 代码并捕获错误

设计:
    - 使用 subprocess 而非 exec,保证崩溃隔离
    - 默认 30s timeout, 防止死循环
    - 捕获 stdout / stderr / returncode
    - 可选 working_dir 和额外环境变量

未来扩展:
    - Docker sandbox(更安全)
    - E2B / Modal remote sandbox
    - Resource limits(memory, CPU)
"""
from __future__ import annotations

import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ExecutionResult:
    """一次代码执行的结果"""
    success: bool
    stdout: str
    stderr: str
    returncode: int
    duration_sec: float
    timed_out: bool = False
    error_type: str = ""
    error_line: Optional[int] = None

    def __repr__(self):
        status = "OK" if self.success else f"FAIL ({self.error_type})"
        return f"ExecutionResult({status}, {self.duration_sec:.2f}s)"


class CodeSandbox:
    """Python 代码沙箱"""

    def __init__(self, timeout_sec: int = 30, python_executable: str = "python"):
        self.timeout_sec = timeout_sec
        self.python_executable = python_executable

    def execute(self, code: str, working_dir: Optional = None,
                env: Optional[Dict] = None) -> ExecutionResult:
        """执行 Python 代码

        Args:
            code: 完整 Python 代码(必须可独立运行)
            working_dir: 工作目录
            env: 额外环境变量

        Returns:
            ExecutionResult
        """
        # 写到临时文件执行(避免命令行长度限制)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            merged_env = None
            if env:
                merged_env = {**os.environ, **env}

            proc = subprocess.run(
                [self.python_executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                cwd=working_dir,
                env=merged_env,
                errors="replace",
            )
            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                duration_sec=0.0,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {self.timeout_sec}s",
                returncode=-1,
                duration_sec=float(self.timeout_sec),
                timed_out=True,
                error_type="TimeoutError",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                returncode=-1,
                duration_sec=0.0,
                error_type=type(e).__name__,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def execute_script(self, script_path: str, args: Optional = None,
                       working_dir: Optional = None) -> ExecutionResult:
        """执行已存在的脚本文件"""
        cmd = [self.python_executable, script_path]
        if args:
            cmd.extend(args)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout_sec, cwd=working_dir, errors="replace",
            )
            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                duration_sec=0.0,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False, stdout="", stderr="Timed out", returncode=-1,
                duration_sec=float(self.timeout_sec), timed_out=True,
                error_type="TimeoutError",
            )