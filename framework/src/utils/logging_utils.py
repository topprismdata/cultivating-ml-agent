"""Structured logging utilities.

Replaces scattered print() calls with timestamped, leveled logging.
Designed to work alongside MLflow for experiment tracking.

Usage:
    from utils.logging_utils import get_logger
    log = get_logger("R06_crossjoin")
    log.info("Training started")
    log.metric("cv_rmse", 21.55)
"""
import time
import sys
from typing import Optional


class ExperimentLogger:
    """Simple logger with timestamps and metric formatting."""

    def __init__(self, name: str = "experiment", level: str = "INFO"):
        self.name = name
        self.level = level
        self._start_time = time.time()

    def _format(self, level: str, msg: str) -> str:
        elapsed = time.time() - self._start_time
        return f"[{time.strftime('%H:%M:%S')} +{elapsed:.0f}s] [{self.name}] [{level}] {msg}"

    def info(self, msg: str) -> None:
        if self.level in ("INFO", "DEBUG"):
            print(self._format("INFO", msg), flush=True)

    def warn(self, msg: str) -> None:
        print(self._format("WARN", msg), flush=True, file=sys.stderr)

    def error(self, msg: str) -> None:
        print(self._format("ERROR", msg), flush=True, file=sys.stderr)

    def debug(self, msg: str) -> None:
        if self.level == "DEBUG":
            print(self._format("DEBUG", msg), flush=True)

    def metric(self, name: str, value: float, extra: str = "") -> None:
        """Log a metric with consistent formatting."""
        suffix = f" ({extra})" if extra else ""
        print(self._format("METRIC", f"{name}: {value:.4f}{suffix}"), flush=True)

    def separator(self, title: str = "") -> None:
        """Print a visual separator."""
        if title:
            print(f"\n{'=' * 70}", flush=True)
            print(f"  {title}", flush=True)
            print(f"{'=' * 70}", flush=True)
        else:
            print(f"{'─' * 70}", flush=True)

    def section(self, title: str) -> None:
        """Print a section header."""
        print(f"\n--- {title} ---", flush=True)

    def data_shape(self, name: str, df) -> None:
        """Log DataFrame shape info."""
        print(self._format("INFO", f"{name}: {df.shape[0]:,} rows x {df.shape[1]} cols"), flush=True)


def get_logger(name: str = "experiment", level: str = "INFO") -> ExperimentLogger:
    """Get an ExperimentLogger instance.

    Args:
        name: Experiment/run name.
        level: Log level (DEBUG, INFO, WARN, ERROR).

    Returns:
        Configured ExperimentLogger.
    """
    return ExperimentLogger(name=name, level=level)
