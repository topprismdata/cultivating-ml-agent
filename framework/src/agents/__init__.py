"""
agents package — Role-based Multi-Agent Orchestration

agy 验证:把 "数据预处理 / 模型拟合 / 代码审查" 分工给不同角色
能大幅减少幻觉,提升复杂 AutoML 流程稳定度。

本模块:
    1. Role — Agent 角色定义(系统 prompt + 工具 + 责任)
    2. Agent — 单个角色 agent
    3. Orchestrator — 多 agent 编排(数据流 / 反馈循环)
"""
from .role import Role, RoleType, ROLE_PRESETS, get_role
from .agent import Agent, AgentMessage
from .orchestrator import Orchestrator, Pipeline

__all__ = [
    "Role", "RoleType", "ROLE_PRESETS", "get_role",
    "Agent", "AgentMessage",
    "Orchestrator", "Pipeline",
]