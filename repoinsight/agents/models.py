from pydantic import BaseModel, Field

from repoinsight.models.answer_model import RepoAnswerResult
from repoinsight.models.analysis_model import AnalysisRunResult
from repoinsight.models.rag_model import IndexResult


AgentPayloadValue = str | int | float | bool | list[str]


class AgentStepSpec(BaseModel):
    """表示多 Agent 编排中的一个固定角色步骤。"""

    # Agent 角色标识，例如 metadata_agent、router_agent。
    role: str = Field(..., description='Agent 角色标识')

    # 面向用户展示的 Agent 名称。
    display_name: str = Field(..., description='Agent 展示名称')

    # 当前 Agent 的职责说明。
    description: str = Field(..., description='Agent 职责说明')

    # 该 Agent 负责的底层阶段名称列表。
    stage_names: list[str] = Field(default_factory=list, description='负责的阶段列表')

    # 当前 Agent 依赖的上游 Agent 角色列表。
    depends_on: list[str] = Field(default_factory=list, description='依赖的 Agent 列表')

    # 是否具备未来并行执行的潜力；当前仅作为规划字段保留。
    can_run_in_parallel: bool = Field(default=False, description='是否可并行执行')


class AgentTaskPacket(BaseModel):
    """表示 planner_agent 分发给某个下游 Agent 的任务卡片。"""

    # 任务接收方角色。
    target_role: str = Field(..., description='任务目标 Agent')

    # 任务标题，便于在 Trace / UI 中快速识别。
    title: str = Field(..., description='任务标题')

    # 当前任务的核心目标。
    objective: str = Field(..., description='任务目标')

    # 该任务依赖的上游 Agent 角色。
    depends_on: list[str] = Field(default_factory=list, description='依赖的上游 Agent')

    # 执行当前任务前应重点关注的输入。
    required_inputs: list[str] = Field(default_factory=list, description='所需输入')

    # 当前任务期望产出的结构化结果。
    expected_outputs: list[str] = Field(default_factory=list, description='期望输出')

    # planner 提供给下游 Agent 的交接备注。
    handoff_notes: list[str] = Field(default_factory=list, description='交接说明')


class AgentRunRecord(BaseModel):
    """表示一次多 Agent 编排中单个 Agent 的执行记录。"""

    # Agent 角色标识，例如 retrieval_agent。
    role: str = Field(..., description='Agent 角色标识')

    # 面向用户展示的 Agent 名称。
    display_name: str = Field(..., description='Agent 展示名称')

    # 当前执行状态，例如 success、failed、pending、skipped。
    status: str = Field(..., description='执行状态')

    # 当前 Agent 负责的阶段名称列表。
    stage_names: list[str] = Field(default_factory=list, description='负责的阶段列表')

    # 当前 Agent 实际完成的阶段名称列表。
    completed_stage_names: list[str] = Field(default_factory=list, description='已完成的阶段列表')

    # 对本次执行结果的简短说明。
    detail: str | None = Field(default=None, description='执行说明')

    # 失败时的错误信息。
    error_message: str | None = Field(default=None, description='错误信息')

    # 当前 Agent 实际尝试执行的次数，便于观察是否发生重试。
    attempt_count: int = Field(default=1, description='尝试次数')

    # 当前 Agent 是否发生过重试。
    used_retry: bool = Field(default=False, description='是否使用重试')

    # 当前 Agent 的执行耗时，单位毫秒。
    duration_ms: int | None = Field(default=None, description='执行耗时（毫秒）')

    # 当前 Agent 的统一结构化产物，便于后续 CLI / UI / LangGraph 统一消费。
    structured_output: 'AgentStructuredOutput | None' = Field(default=None, description='结构化输出')


class AgentEvidenceItem(BaseModel):
    """表示 Agent 输出中的一条结构化证据。"""

    # 证据类型，例如 document、snippet、location、issue_tag。
    kind: str = Field(default='text', description='证据类型')

    # 证据标签，便于 CLI 或 UI 直接展示。
    label: str = Field(..., description='证据标签')

    # 证据对应的源文件路径。
    source_path: str | None = Field(default=None, description='来源文件路径')

    # 证据对应的源码位置或逻辑位置。
    location: str | None = Field(default=None, description='证据位置')

    # 证据摘要片段。
    snippet: str | None = Field(default=None, description='证据片段')


class AgentStructuredOutput(BaseModel):
    """表示多 Agent 协作中统一的结构化输出协议。"""

    # 当前 Agent 提炼出的核心结论。
    conclusions: list[str] = Field(default_factory=list, description='核心结论')

    # 当前 Agent 使用到的证据列表。
    evidence: list[AgentEvidenceItem] = Field(default_factory=list, description='证据列表')

    # 当前 Agent 主动暴露的不确定点或风险。
    uncertainties: list[str] = Field(default_factory=list, description='不确定点')

    # 建议下游 Agent 或调用方继续执行的动作。
    next_actions: list[str] = Field(default_factory=list, description='后续动作建议')

    # 结构化元数据，便于后续 UI / Trace / 评测直接读取。
    metadata: dict[str, AgentPayloadValue] = Field(default_factory=dict, description='结构化元数据')


class CoordinatedAnalysisResult(BaseModel):
    """表示一轮带多 Agent 视角的分析编排结果。"""

    # 原始分析结果，便于继续复用现有报告、索引和问答链路。
    analysis_result: AnalysisRunResult = Field(..., description='分析结果')

    # 当前编排使用的 Agent 计划。
    agent_plan: list[AgentStepSpec] = Field(default_factory=list, description='Agent 计划')

    # 当前编排的 Agent 执行记录。
    agent_trace: list[AgentRunRecord] = Field(default_factory=list, description='Agent 执行记录')

    # 若启用了 memory_agent，则记录知识入库结果。
    index_result: IndexResult | None = Field(default=None, description='知识入库结果')

    # 各 Agent 之间传递的共享上下文快照，便于观察状态流转。
    shared_context: dict[str, str | int | float | bool | list[str]] = Field(
        default_factory=dict,
        description='共享上下文',
    )


class AnswerRouteDecision(BaseModel):
    """表示问答链路中 router_agent 的路由决策。"""

    # 目标仓库标识，例如 owner/repo。
    repo_id: str = Field(..., description='仓库标识')

    # 用户原始问题。
    question: str = Field(..., description='用户问题')

    # 路由后的问题焦点，例如 overview、startup、implementation。
    focus: str = Field(..., description='问题焦点')

    # 实际检索使用的 top_k。
    retrieval_top_k: int = Field(..., description='实际检索数量')

    # 路由决策说明，便于调试与后续优化。
    reason: str = Field(..., description='路由说明')


class CodeTraceStep(BaseModel):
    """表示 code_agent 提炼出的单步源码追踪结果。"""

    # 当前步骤的类型，例如 route、function、class、callee。
    step_kind: str = Field(..., description='追踪步骤类型')

    # 当前步骤对应的符号或路由名称。
    label: str = Field(..., description='步骤标签')

    # 当前步骤对应的源码文件路径。
    source_path: str | None = Field(default=None, description='源码文件路径')

    # 当前步骤对应的源码位置，例如 app.py:L18-L33。
    location: str | None = Field(default=None, description='源码位置')

    # 当前步骤的简要说明。
    summary: str = Field(..., description='步骤说明')

    # 当前步骤截取出的源码片段。
    snippet: str | None = Field(default=None, description='源码片段')

    # 当前步骤在调用链中的深度，入口步骤为 0。
    depth: int = Field(default=0, description='调用链深度')

    # 当前步骤的上游标签，便于展示 route -> handler -> service 关系。
    parent_label: str | None = Field(default=None, description='上游步骤标签')

    # 当前步骤与上游之间的关系类型，例如 handle_route、delegate_service。
    relation_type: str | None = Field(default=None, description='与上游之间的关系类型')


class CodeRelationEdge(BaseModel):
    """表示关系链中的一条有类型的边。"""

    # 当前边的起点标签。
    source_label: str = Field(..., description='边起点标签')

    # 当前边的终点标签。
    target_label: str = Field(..., description='边终点标签')

    # 当前边的关系类型，例如 handle_route、delegate_service。
    relation_type: str | None = Field(default=None, description='边关系类型')


class CodeRelationChain(BaseModel):
    """表示一条结构化的代码关系链。"""

    # 保留原有无类型字符串，便于兼容旧展示逻辑。
    plain_text: str = Field(..., description='无类型关系链文本')

    # 带关系类型的展示文本，例如 A -[call]-> B。
    typed_text: str = Field(..., description='带关系类型的关系链文本')

    # 当前链路包含的节点标签列表。
    labels: list[str] = Field(default_factory=list, description='链路节点标签')

    # 当前链路包含的结构化边列表。
    edges: list[CodeRelationEdge] = Field(default_factory=list, description='链路边列表')


class CodeInvestigationResult(BaseModel):
    """表示 code_agent 基于代码级摘要提炼出的实现线索。"""

    # 当前调查对应的问题焦点，例如 implementation 或 api。
    focus: str = Field(..., description='调查焦点')

    # code_agent 对实现路径的简短总结。
    summary: str = Field(..., description='代码调查摘要')

    # 命中的函数、方法或类限定名列表。
    matched_symbols: list[str] = Field(default_factory=list, description='命中的代码符号')

    # 命中的接口或路由列表。
    matched_routes: list[str] = Field(default_factory=list, description='命中的接口路由')

    # 涉及到的源码文件路径列表。
    source_paths: list[str] = Field(default_factory=list, description='涉及的源码路径')

    # 关键源码位置列表，例如 app.py:L18-L33。
    evidence_locations: list[str] = Field(default_factory=list, description='关键源码位置')

    # 从命中文档中提炼出的下游调用符号列表。
    called_symbols: list[str] = Field(default_factory=list, description='下游调用符号')

    # 代码级追踪步骤，便于 CLI 展示或后续继续扩展。
    trace_steps: list[CodeTraceStep] = Field(default_factory=list, description='源码追踪步骤')

    # 从源码追踪步骤归纳出的代表性关系链，便于展示 route -> handler -> service 之类的路径。
    relation_chains: list[str] = Field(default_factory=list, description='代表性关系链')

    # 带关系类型的结构化关系链，便于回答和 CLI 更精细展示。
    relation_chain_details: list[CodeRelationChain] = Field(default_factory=list, description='结构化关系链')

    # 供 synthesis_agent 直接消费的实现说明列表。
    implementation_notes: list[str] = Field(default_factory=list, description='实现说明')

    # 本次调查主要消费到的文档类型列表。
    evidence_doc_types: list[str] = Field(default_factory=list, description='证据文档类型')

    # code_agent 对当前调查结果的相关性评分，范围约为 0~1。
    relevance_score: float = Field(default=0.0, description='相关性评分')

    # 对当前调查结果的置信度分级，例如 high、medium、low。
    confidence_level: str = Field(default='low', description='置信度等级')

    # 对当前质量判断的简短说明，便于 CLI 解释为什么可信或不可信。
    quality_notes: list[str] = Field(default_factory=list, description='质量说明')

    # 当前结果是否直接命中了 code_agent 的本地缓存。
    cache_hit: bool = Field(default=False, description='是否命中缓存')

    # 是否触发过低置信度后的自动扩检恢复。
    recovery_attempted: bool = Field(default=False, description='是否执行恢复扩检')

    # 自动扩检后是否拿到了更好的结果。
    recovery_improved: bool = Field(default=False, description='恢复是否提升结果')


class AnswerVerificationResult(BaseModel):
    """表示 verifier_agent 对最终回答做出的证据一致性检查结果。"""

    # 验证结论分级，例如 passed、warning、failed。
    verdict: str = Field(default='warning', description='验证结论')

    # 证据一致性评分，范围约为 0~1。
    support_score: float = Field(default=0.0, description='证据支撑评分')

    # 本次大致检查到的结论条目数量。
    checked_claim_count: int = Field(default=0, description='检查的结论数量')

    # 被判定为已有证据支撑的结论数量。
    supported_claim_count: int = Field(default=0, description='有证据支撑的结论数量')

    # 验证阶段发现的问题列表。
    issues: list[str] = Field(default_factory=list, description='验证问题列表')

    # 结构化失败原因标签，便于后续编排器按类型恢复。
    issue_tags: list[str] = Field(default_factory=list, description='失败原因标签')

    # 验证阶段给出的补充说明。
    notes: list[str] = Field(default_factory=list, description='验证说明')


class CoordinatedAnswerResult(BaseModel):
    """表示一轮带多 Agent 视角的问答编排结果。"""

    # 原始问答结果，继续兼容 CLI 和现有展示链路。
    answer_result: RepoAnswerResult = Field(..., description='问答结果')

    # 当前问答编排使用的 Agent 计划。
    agent_plan: list[AgentStepSpec] = Field(default_factory=list, description='Agent 计划')

    # 当前问答编排的 Agent 执行记录。
    agent_trace: list[AgentRunRecord] = Field(default_factory=list, description='Agent 执行记录')

    # router_agent 的路由决策。
    route_decision: AnswerRouteDecision = Field(..., description='路由决策')

    # retrieval_agent 实际使用的检索后端。
    retrieval_backend: str = Field(default='local', description='检索后端')

    # retrieval_agent 实际召回的结果数量。
    retrieval_hit_count: int = Field(default=0, description='检索结果数量')

    # retrieval_agent 排序后命中的文档类型列表。
    retrieved_doc_types: list[str] = Field(default_factory=list, description='命中文档类型列表')

    # 若执行了 code_agent，则记录代码调查结果。
    code_investigation: CodeInvestigationResult | None = Field(default=None, description='代码调查结果')

    # verifier_agent 对最终回答的证据一致性检查结果。
    verification_result: AnswerVerificationResult | None = Field(default=None, description='回答验证结果')

    # 各 Agent 之间传递的共享上下文快照，便于观察状态流转。
    shared_context: dict[str, str | int | float | bool | list[str]] = Field(
        default_factory=dict,
        description='共享上下文',
    )


AgentRunRecord.model_rebuild()
