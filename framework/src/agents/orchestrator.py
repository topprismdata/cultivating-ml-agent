"""
Orchestrator — 多 agent 编排器

agy 验证模式: Architect → Coder → Critic 流水线

典型工作流:
    1. Architect 收到任务,产出计划
    2. Coder 根据计划写代码
    3. Critic 审查代码,反馈问题
    4. 如果 REJECT,Coder 重写
    5. 直到 APPROVE,Integrator 提交

支持自定义 Pipeline(DAG 风格)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any

from .role import Role, get_role
from .agent import Agent, AgentMessage, LLMCallable


@dataclass
class PipelineStep:
    """Pipeline 中的一步"""
    agent_name: str
    input_from: Optional[str] = None  # 上游 agent 名(None = 初始输入)
    output_to: Optional[str] = None
    transform: Optional[Callable[[AgentMessage], str]] = None  # 转换函数


@dataclass
class Pipeline:
    """一个 agent 流水线定义"""
    name: str
    steps: List[PipelineStep] = field(default_factory=list)
    description: str = ""

    def add_step(self, agent_name: str, input_from: Optional[str] = None,
                 output_to: Optional[str] = None,
                 transform: Optional[Callable] = None) -> "Pipeline":
        self.steps.append(PipelineStep(agent_name, input_from, output_to, transform))
        return self


class Orchestrator:
    """多 agent 协调器"""

    def __init__(self, llm_call: Optional[LLMCallable] = None):
        self.llm_call = llm_call
        self.agents: Dict[str, Agent] = {}
        self.pipelines: Dict[str, Pipeline] = {}

    def register_agent(self, agent: Agent) -> "Orchestrator":
        self.agents[agent.name] = agent
        return self

    def register_role(self, role_name: str, agent_name: Optional[str] = None
                      ) -> Agent:
        """从内置角色快速注册一个 agent"""
        role = get_role(role_name)
        agent = Agent(role, llm_call=self.llm_call,
                      name=agent_name or role_name)
        self.register_agent(agent)
        return agent

    def register_pipeline(self, pipeline: Pipeline) -> "Orchestrator":
        self.pipelines[pipeline.name] = pipeline
        return self

    def create_standard_ml_pipeline(self) -> Pipeline:
        """创建标准 ML pipeline:Architect → Coder → Critic 循环"""
        # 确保三个角色都已注册
        for role_name in ("data-architect", "ml-coder", "continuity-critic"):
            if role_name not in self.agents:
                self.register_role(role_name)

        pipeline = Pipeline(
            name="standard-ml",
            description="Architect → Coder → Critic with revision loop",
        )
        pipeline.add_step(
            agent_name="data-architect",
            input_from=None,  # 接收初始任务
            output_to="ml-coder",
        )
        pipeline.add_step(
            agent_name="ml-coder",
            input_from="data-architect",
            output_to="continuity-critic",
        )
        pipeline.add_step(
            agent_name="continuity-critic",
            input_from="ml-coder",
            output_to=None,  # 终态
        )
        self.register_pipeline(pipeline)
        return pipeline

    def run_pipeline(self, pipeline_name: str,
                     initial_input: str,
                     max_revisions: int = 2) -> List[AgentMessage]:
        """运行一个 pipeline

        Args:
            pipeline_name: pipeline 名
            initial_input: 初始输入
            max_revisions: Critic REJECT 后最多重试次数
        """
        pipeline = self.pipelines.get(pipeline_name)
        if not pipeline:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        all_outputs: List[AgentMessage] = []
        last_msg: Optional[AgentMessage] = None
        upstream_content = initial_input

        for revision_round in range(max_revisions + 1):
            for step in pipeline.steps:
                agent = self.agents.get(step.agent_name)
                if not agent:
                    raise ValueError(f"Agent not registered: {step.agent_name}")

                # 构造 input
                if step.transform and last_msg:
                    input_for_agent = step.transform(last_msg)
                elif step.input_from and last_msg:
                    input_for_agent = f"[From {last_msg.sender}]:\n{last_msg.content}"
                else:
                    input_for_agent = upstream_content

                # 执行
                response = agent.act(input_for_agent)
                all_outputs.append(response)
                last_msg = response

                # 简单启发式:如果 Critic 说 REJECT,触发 revision
                if "REJECT" in response.content and revision_round < max_revisions:
                    # 把 Critic 反馈作为下一轮 Architect → Coder 的输入补充
                    upstream_content = (
                        f"{initial_input}\n\n"
                        f"[Critic feedback]: {response.content}\n\n"
                        f"Please revise based on the feedback."
                    )
                    last_msg = None  # 重置,让 Architect 重新规划
                    break

        return all_outputs

    def broadcast(self, content: str, receiver: str = "all",
                  sender: str = "user") -> List[AgentMessage]:
        """广播一条消息给所有 agent"""
        responses = []
        for agent in self.agents.values():
            msg = AgentMessage(sender=sender, receiver=agent.name, content=content)
            agent.receive(msg)
            response = agent.act(content)
            responses.append(response)
        return responses