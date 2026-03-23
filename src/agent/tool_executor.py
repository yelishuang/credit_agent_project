"""
工具执行器 — 直接调用 MCP server 中的底层 Python 函数。
输入：工具名 + 参数 dict，输出：JSON 字符串。
"""

import inspect
import json
import logging

logger = logging.getLogger(__name__)

# 可用工具名 -> 调用函数的映射（延迟导入，避免启动时加载重型依赖）
TOOL_REGISTRY = {}


def _ensure_registry():
    if TOOL_REGISTRY:
        return
    from src.mcp_server.server import (
        query_user_credit_data,
        predict_risk_score,
        search_knowledge_base,
    )
    TOOL_REGISTRY["query_user_credit_data"] = query_user_credit_data
    TOOL_REGISTRY["predict_risk_score"] = predict_risk_score
    TOOL_REGISTRY["search_knowledge_base"] = search_knowledge_base


def execute_tool(tool_name: str, arguments: dict) -> str:
    """执行指定工具，返回 JSON 字符串结果。"""
    _ensure_registry()

    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)

    fn = TOOL_REGISTRY[tool_name]
    try:
        # 过滤掉函数签名中不存在的参数（模型可能传多余参数如 top_k）
        sig = inspect.signature(fn)
        valid_args = {k: v for k, v in arguments.items() if k in sig.parameters}
        result = fn(**valid_args)
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"工具 {tool_name} 执行失败: {e}")
        return json.dumps({"error": f"工具执行失败: {str(e)}"}, ensure_ascii=False)
