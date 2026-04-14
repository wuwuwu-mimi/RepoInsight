import io
from contextlib import redirect_stdout

from repoinsight.agents.models import (
    AgentRunRecord,
    AgentStepSpec,
    AnswerRouteDecision,
    AnswerVerificationResult,
    CodeInvestigationResult,
    CodeTraceStep,
    CoordinatedAnswerResult,
)
from repoinsight.models.answer_model import AnswerEvidence, RepoAnswerResult
import repoinsight.cli.main as cli_main



def test_answer_command_renders_agent_trace_and_code_investigation() -> None:
    original_run_multi_agent_answer = cli_main.run_multi_agent_answer
    original_get_llm_settings = cli_main.get_llm_settings
    try:
        cli_main.get_llm_settings = lambda: None
        cli_main.run_multi_agent_answer = lambda **kwargs: CoordinatedAnswerResult(
            answer_result=RepoAnswerResult(
                repo_id='demo/sample',
                question='handle_login 是怎么实现的？',
                answer='结论：\n- handle_login 负责登录。\n依据：\n- function_summary 命中。\n不确定点：\n- 暂无。\n补充线索：\n- handle_login @ app.py:L18-L33 -> 函数 handle_login 承担当前实现逻辑',
                answer_mode='extractive',
                backend='local',
                fallback_used=False,
                llm_enabled=False,
                llm_attempted=False,
                llm_error=None,
                evidence=[
                    AnswerEvidence(
                        repo_id='demo/sample',
                        doc_type='function_summary',
                        source_path='app.py',
                        snippet='函数 handle_login 位于 app.py。',
                    )
                ],
            ),
            agent_plan=[
                AgentStepSpec(role='router_agent', display_name='Router Agent', description='route'),
                AgentStepSpec(role='retrieval_agent', display_name='Retrieval Agent', description='retrieve', depends_on=['router_agent']),
                AgentStepSpec(role='code_agent', display_name='Code Agent', description='code', depends_on=['retrieval_agent']),
                AgentStepSpec(role='synthesis_agent', display_name='Synthesis Agent', description='synthesis', depends_on=['retrieval_agent', 'code_agent']),
            ],
            agent_trace=[
                AgentRunRecord(role='router_agent', display_name='Router Agent', status='success', stage_names=['route_question'], completed_stage_names=['route_question'], detail='问题被路由为实现细节问题。'),
                AgentRunRecord(role='retrieval_agent', display_name='Retrieval Agent', status='success', stage_names=['search'], completed_stage_names=['search'], detail='命中 3 条文档。'),
                AgentRunRecord(role='code_agent', display_name='Code Agent', status='success', stage_names=['investigate_code_context'], completed_stage_names=['investigate_code_context'], detail='命中符号：AuthService.handle_login'),
                AgentRunRecord(role='synthesis_agent', display_name='Synthesis Agent', status='success', stage_names=['build_answer_result'], completed_stage_names=['build_answer_result'], detail='已完成回答合成。'),
            ],
            route_decision=AnswerRouteDecision(
                repo_id='demo/sample',
                question='handle_login 是怎么实现的？',
                focus='implementation',
                retrieval_top_k=5,
                reason='实现问题',
            ),
            retrieval_backend='local',
            retrieval_hit_count=3,
            retrieved_doc_types=['function_summary', 'class_summary'],
            code_investigation=CodeInvestigationResult(
                focus='implementation',
                summary='已定位到与问题最相关的实现符号 AuthService.handle_login，源码主要位于 app.py。',
                matched_symbols=['AuthService.handle_login'],
                matched_routes=[],
                source_paths=['app.py'],
                evidence_locations=['app.py:L18-L33'],
                called_symbols=['verify_password', 'create_session_token'],
                trace_steps=[
                    CodeTraceStep(
                        step_kind='entry',
                        label='AuthService.handle_login',
                        source_path='app.py',
                        location='app.py:L18-L33',
                        summary='函数 handle_login 承担当前实现逻辑',
                        snippet='L18: def handle_login(self, username, password):\nL19:     if not verify_password(username, password):',
                    )
                ],
                relation_chains=['POST /login -> AuthService.handle_login -> create_session_token'],
                implementation_notes=['函数 handle_login 承担当前实现逻辑。'],
                evidence_doc_types=['function_summary'],
            ),
            verification_result=AnswerVerificationResult(
                verdict='warning',
                support_score=0.5,
                checked_claim_count=2,
                supported_claim_count=1,
                issues=['部分结论缺少足够直接的证据支撑。'],
                issue_tags=['evidence_weak'],
                notes=['本次验证参考了 2 条主证据文档。'],
            ),
        )

        output = io.StringIO()
        with redirect_stdout(output):
            cli_main.answer('demo/sample', 'handle_login 是怎么实现的？', use_llm=False, stream=False)

        rendered = output.getvalue()
        assert '问答 Agent 轨迹' in rendered
        assert '代码调查' in rendered
        assert '源码追踪步骤' in rendered
        assert '源码片段：AuthService.handle_login' in rendered
        assert '回答验证' in rendered
        assert 'evidence_weak' in rendered
        assert '问题焦点' in rendered
        assert 'implementation' in rendered
        assert 'AuthService.handle_login' in rendered
        assert 'app.py:L18-L33' in rendered
        assert '关系链' in rendered
        assert 'POST /login -> AuthService.handle_login -> create_session_token' in rendered
    finally:
        cli_main.run_multi_agent_answer = original_run_multi_agent_answer
        cli_main.get_llm_settings = original_get_llm_settings
