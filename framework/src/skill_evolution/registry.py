"""
Skill Registry — 把验证通过的 skill 注册到 MCP 库

agy 验证:Self-evolving skill 必须有持久化,否则重启后丢失。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

from .extractor import SkillCandidate
from .validator import SkillValidator, ValidationResult, ValidationVerdict


@dataclass
class RegisteredSkill:
    """已注册的 skill"""
    name: str
    file_path: Path
    registered_at: str
    importance: float
    type: str


class SkillRegistry:
    """Skill 注册表"""

    def __init__(self, skills_dir: str = "skills/examples",
                 state_file: Optional[str] = None):
        self.skills_dir = Path(skills_dir)
        self.state_file = Path(state_file or f"{skills_dir}/.registry.json")
        self.validator = SkillValidator()
        self.registered: Dict[str, RegisteredSkill] = self._load_state()

    def register(self, candidate: SkillCandidate,
                 existing_descriptions: Optional[List[str]] = None,
                 auto_approve: bool = False) -> ValidationResult:
        """注册一个 skill

        Args:
            candidate: 候选 skill
            existing_descriptions: 现有 skill 描述(用于去重)
            auto_approve: 是否跳过验证(默认 False,生产必 False)

        Returns:
            ValidationResult
        """
        if not auto_approve:
            if existing_descriptions is None:
                existing_descriptions = self._gather_existing_descriptions()
            result = self.validator.validate(candidate, existing_descriptions)
            if result.verdict == ValidationVerdict.REJECT:
                return result
        else:
            result = ValidationResult(verdict=ValidationVerdict.APPROVE)

        # 写到文件
        skill_dir = self.skills_dir / candidate.name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(candidate.to_markdown(), encoding="utf-8")

        # 更新 registry
        self.registered[candidate.name] = RegisteredSkill(
            name=candidate.name,
            file_path=skill_path,
            registered_at=datetime.now().isoformat(),
            importance=candidate.importance,
            type=candidate.type,
        )
        self._save_state()
        return result

    def list_registered(self) -> List[RegisteredSkill]:
        return list(self.registered.values())

    def is_registered(self, name: str) -> bool:
        return name in self.registered

    def _gather_existing_descriptions(self) -> List[str]:
        """收集现有 skill 描述(用于去重检测)"""
        descriptions = []
        if not self.skills_dir.exists():
            return descriptions
        for skill_dir in self.skills_dir.iterdir():
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                try:
                    text = skill_md.read_text(encoding="utf-8", errors="replace")
                    # 提取 description 字段
                    import re
                    m = re.search(r"description:\s*\|\s*\n((?:\s{2,}.+\n)+)", text)
                    if m:
                        descriptions.append(m.group(1).strip())
                    else:
                        descriptions.append(text[:500])
                except Exception:
                    pass
        return descriptions

    def _load_state(self) -> Dict[str, RegisteredSkill]:
        if not self.state_file.exists():
            return {}
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return {
                k: RegisteredSkill(
                    name=v["name"],
                    file_path=Path(v["file_path"]),
                    registered_at=v["registered_at"],
                    importance=v["importance"],
                    type=v["type"],
                )
                for k, v in data.items()
            }
        except Exception:
            return {}

    def _save_state(self):
        data = {
            k: {
                "name": v.name,
                "file_path": str(v.file_path),
                "registered_at": v.registered_at,
                "importance": v.importance,
                "type": v.type,
            }
            for k, v in self.registered.items()
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )