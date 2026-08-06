"""
skill_evolution package — Self-Evolving Skills(Voyager 风格)

agy 验证:Agent 解决新题型后自动总结封装 skill,可让 43+ skill 自动扩充至 100+。

本模块:
    1. Extractor — 从实验记录提取候选 skill
    2. Validator — 验证 skill 质量(模板合规 / 不重复 / 可执行)
    3. Registry — 注册到 MCP skill 库
"""
from .extractor import SkillExtractor, SkillCandidate
from .validator import SkillValidator, ValidationResult
from .registry import SkillRegistry

__all__ = [
    "SkillExtractor", "SkillCandidate",
    "SkillValidator", "ValidationResult",
    "SkillRegistry",
]