"""Path resolution utilities.

Eliminates the repeated Path(__file__).resolve().parent.parent pattern
found in every experiment script. Provides a single source of truth
for directory paths.

Usage:
    from utils.paths import get_competition_dirs
    dirs = get_competition_dirs(cfg)
    print(dirs.data_raw, dirs.outputs, dirs.submissions)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CompetitionDirs:
    """Resolved paths for a competition directory."""
    root: Path
    data_raw: Path
    data_processed: Path
    outputs: Path
    submissions: Path
    models: Path
    oof: Path
    scripts: Path
    notebooks: Path
    mlruns: Path

    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        for path in [self.data_raw, self.data_processed, self.outputs,
                     self.submissions, self.models, self.oof,
                     self.scripts, self.notebooks, self.mlruns]:
            path.mkdir(parents=True, exist_ok=True)


def get_competition_dirs(cfg_or_root) -> CompetitionDirs:
    """Resolve all competition directory paths from config or root path.

    Args:
        cfg_or_root: CompetitionConfig object or Path to competition root.

    Returns:
        CompetitionDirs with all paths resolved.
    """
    if hasattr(cfg_or_root, "project_root"):
        root = cfg_or_root.project_root
    elif isinstance(cfg_or_root, (str, Path)):
        root = Path(cfg_or_root)
    else:
        raise TypeError(f"Expected CompetitionConfig or Path, got {type(cfg_or_root)}")

    return CompetitionDirs(
        root=root,
        data_raw=root / "data" / "raw",
        data_processed=root / "data" / "processed",
        outputs=root / "outputs",
        submissions=root / "outputs" / "submissions",
        models=root / "outputs" / "models",
        oof=root / "outputs" / "oof",
        scripts=root / "scripts",
        notebooks=root / "notebooks",
        mlruns=root / "mlruns",
    )
