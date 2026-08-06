"""
Agent — 单个角色的 agent 实例

简化实现:用 LLM callable 处理消息,产出消息响应。
不绑定具体 LLM SDK,接受任何 (prompt → response) 函数。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .role import Role


@dataclass
class AgentMessage:
    """agent 间的一条消息"""
    sender: str
    receiver: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


# LLM 调用签名: (system_prompt, user_messages, tools) → str
LLMCallable = Callable[[str, List[dict], List[str]], str]


class Agent:
    """单个角色 agent"""

    def __init__(self, role: Role, llm_call: Optional[LLMCallable] = None,
                 name: Optional[str] = None):
        self.role = role
        self.name = name or role.name
        self.llm_call = llm_call
        self.history: List[AgentMessage] = []

    def receive(self, message: AgentMessage) -> None:
        """接收一条消息"""
        self.history.append(message)

    def send(self, content: str, receiver: str = "orchestrator",
             metadata: Optional[dict] = None) -> AgentMessage:
        """构造并记录一条发送消息"""
        msg = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            metadata=metadata or {},
        )
        self.history.append(msg)
        return msg

    def act(self, input_content: str, context: Optional[List[dict]] = None
            ) -> AgentMessage:
        """基于角色 + 输入 + 历史,生成响应

        Args:
            input_content: 当前输入
            context: 额外的对话历史 [{role, content}, ...]
        """
        if self.llm_call is None:
            # Mock 模式:直接 echo(用于测试编排逻辑)
            response = f"[{self.name} mock response to]: {input_content[:100]}"
        else:
            messages = (context or []) + [
                {"role": "user", "content": input_content}
            ]
            response = self.llm_call(self.role.system_prompt, messages,
                                      self.role.tools)
        return self.send(response)

    def __repr__(self):
        return f"Agent({self.name})"