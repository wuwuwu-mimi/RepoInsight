from __future__ import annotations

import re


_SERVICE_KEYWORDS = ('service', 'svc')
_REPOSITORY_KEYWORDS = ('repository', 'repo', 'dao')
_HANDLER_KEYWORDS = ('handler', 'controller', 'view', 'endpoint')


def normalize_python_call_target(target: str) -> str:
    """把 Python 调用目标归一化成更适合跨文件匹配的符号引用。"""
    normalized = target.strip()
    if not normalized:
        return ''

    if normalized.startswith('await '):
        normalized = normalized[len('await ') :].strip()

    normalized = re.sub(r'^(self|cls)\.', '', normalized)
    normalized = re.sub(r'\b([A-Za-z_][A-Za-z0-9_]*)\([^()]*\)\.', r'\1.', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def normalize_javascript_call_target(target: str) -> str:
    """把 JS/TS 调用目标归一化成更适合跨文件匹配的符号引用。"""
    normalized = target.strip()
    if not normalized:
        return ''

    normalized = normalized.replace('?.', '.')
    normalized = re.sub(r'^\bawait\s+', '', normalized)
    normalized = re.sub(r'^(this|self)\.', '', normalized)
    normalized = re.sub(r'\bnew\s+([A-Za-z_$][\w$]*)\s*\([^()]*\)\.', r'\1.', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def infer_code_symbol_role(symbol_ref: str | None) -> str | None:
    """根据符号命名粗略推断其业务角色。"""
    if not symbol_ref:
        return None

    lowered = symbol_ref.lower()
    if any(keyword in lowered for keyword in _REPOSITORY_KEYWORDS):
        return 'repository'
    if any(keyword in lowered for keyword in _SERVICE_KEYWORDS):
        return 'service'
    if any(keyword in lowered for keyword in _HANDLER_KEYWORDS):
        return 'handler'
    return None


def infer_call_relation_type(called_symbol: str) -> str:
    """根据被调目标的命名特征，把普通调用提升为更具体的业务关系。"""
    role = infer_code_symbol_role(called_symbol)
    if role == 'repository':
        return 'delegate_repository'
    if role == 'service':
        return 'delegate_service'
    return 'call'


def score_symbol_role_priority(symbol_ref: str) -> int:
    """为主链路选择提供一个轻量角色优先级。"""
    role = infer_code_symbol_role(symbol_ref)
    if role == 'repository':
        return 4
    if role == 'service':
        return 3
    if role == 'handler':
        return 2
    return 0
