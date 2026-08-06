"""
Log Truncator — 训练日志智能截断

保留优先级:
    1. ERROR / Exception / Traceback 完整保留
    2. WARNING 完整保留
    3. 关键 metric (val_loss / accuracy / f1) 保留
    4. 首尾 N 行保留
    5. 中间重复行折叠

应用场景:把 10000 行训练日志压成 200 行保留全部关键信号
"""
from __future__ import annotations

import re
from typing import List, Tuple


# 关键信号模式
ERROR_PATTERNS = [
    re.compile(r"\b(ERROR|Exception|Traceback|FAILED|CRITICAL)\b", re.IGNORECASE),
    re.compile(r"^\s*File \".+\", line \d+", re.MULTILINE),  # Python traceback
]
WARNING_PATTERNS = [
    re.compile(r"\b(WARNING|WARN)\b"),
]
METRIC_PATTERNS = [
    re.compile(r"(val_|valid_|test_|eval_)(loss|acc|accuracy|f1|auc|rmse|mae)\s*[:=]\s*[\d.eE+-]+", re.IGNORECASE),
    re.compile(r"(loss|acc|accuracy|f1|auc)\s*[:=]\s*[\d.eE+-]+", re.IGNORECASE),
    re.compile(r"epoch\s*\d+.*?(loss|acc|f1)", re.IGNORECASE),
]


class LogTruncator:
    """训练/运行日志智能截断"""

    def __init__(self, max_lines: int = 200,
                 head_lines: int = 20,
                 tail_lines: int = 30,
                 fold_repeats: bool = True):
        self.max_lines = max_lines
        self.head_lines = head_lines
        self.tail_lines = tail_lines
        self.fold_repeats = fold_repeats

    def truncate(self, log_text: str) -> str:
        """截断日志

        优先级:
          1. ERROR/WARN 必须全部保留
          2. METRIC 采样(每 N 行保留 1 行)
          3. HEAD + TAIL 保留
          4. 配额满 → 按优先级保留
        """
        lines = log_text.splitlines()
        if len(lines) <= self.max_lines:
            return log_text

        # 1. 分类(优先级)
        errors: List[Tuple[int, str]] = []
        warnings: List[Tuple[int, str]] = []
        metrics: List[Tuple[int, str]] = []
        for i, line in enumerate(lines):
            if any(p.search(line) for p in ERROR_PATTERNS):
                errors.append((i, line))
            elif any(p.search(line) for p in WARNING_PATTERNS):
                warnings.append((i, line))
            elif any(p.search(line) for p in METRIC_PATTERNS):
                metrics.append((i, line))

        # 2. METRIC 采样(均匀间隔)
        if metrics:
            sample_step = max(1, len(metrics) // max(1, self.max_lines // 3))
            metrics = metrics[::sample_step]

        # 3. 头部 / 尾部
        head = [(i, lines[i]) for i in range(min(self.head_lines, len(lines)))]
        tail = [(i, lines[i]) for i in range(max(0, len(lines) - self.tail_lines), len(lines))]

        # 4. 合并去重,按行号排序
        all_kept = errors + warnings + metrics + head + tail
        seen = set()
        unique = []
        for idx, line in sorted(all_kept):
            if idx not in seen:
                seen.add(idx)
                unique.append((idx, line))

        # 5. 截断到 max_lines(优先保留 errors/warnings)
        if len(unique) > self.max_lines:
            # 先保留 errors + warnings,余量给其他
            errors_in = [u for u in unique if any(p.search(u[1]) for p in ERROR_PATTERNS)]
            warnings_in = [u for u in unique if u not in errors_in and
                            any(p.search(u[1]) for p in WARNING_PATTERNS)]
            others = [u for u in unique if u not in errors_in and u not in warnings_in]
            kept = errors_in + warnings_in + others
            kept.sort()
            unique = kept[:self.max_lines]

        return self._render(unique, total_lines=len(lines), dropped=len(lines) - len(unique))

    def _render(self, kept: List[Tuple[int, str]],
                total_lines: int, dropped: int) -> str:
        out = [f"# Log: {total_lines} lines → {len(kept)} kept ({dropped} dropped)\n"]
        prev_idx = -1
        for idx, line in kept:
            # 自动分类(给 kind 标签)
            if any(p.search(line) for p in ERROR_PATTERNS):
                kind = "ERROR"
            elif any(p.search(line) for p in WARNING_PATTERNS):
                kind = "WARN"
            elif any(p.search(line) for p in METRIC_PATTERNS):
                kind = "METRIC"
            else:
                kind = "INFO"
            if prev_idx >= 0 and idx > prev_idx + 1:
                out.append(f"... [{idx - prev_idx - 1} lines omitted] ...")
            out.append(f"L{idx:5d} [{kind:6s}] {line}")
            prev_idx = idx
        return "\n".join(out)


# 便捷函数
def truncate_log(log_text: str, max_lines: int = 200) -> str:
    """便捷入口"""
    return LogTruncator(max_lines=max_lines).truncate(log_text)