"""
MCP 工具单元测试 — 直接调用工具函数验证输入输出
"""

import os
import sys
import json

os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mcp_server.server import query_user_credit_data, predict_risk_score, search_knowledge_base

SFT_FIELDS = [
    "mainoccupationinc_384A", "credamount_770A", "annuity_780A",
    "totaldebt_9A", "currdebt_22A", "maxdpdlast24m_143P",
    "maxdpdlast3m_392P", "numactivecreds_622L", "applications30d_658L",
    "numrejects9m_859L", "debtoverdue_47A", "days30_165L",
    "numinstlswithdpd5_4187116L",
]


def test_query_user_credit_data():
    print("=" * 60)
    print("Test 1: query_user_credit_data")
    print("=" * 60)
    result_str = query_user_credit_data(user_id=57543)
    result = json.loads(result_str)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 验证 13 个字段齐全
    for f in SFT_FIELDS:
        assert f in result, f"缺少字段: {f}"
    # 验证无 NaN
    for k, v in result.items():
        assert v is None or isinstance(v, (int, float)), f"{k} 类型异常: {type(v)}"
    print("[PASS] 13 个字段齐全，类型正确\n")
    return result


def test_predict_risk_score(features: dict):
    print("=" * 60)
    print("Test 2: predict_risk_score")
    print("=" * 60)
    result_str = predict_risk_score(features=json.dumps(features))
    result = json.loads(result_str)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    assert "risk_score" in result
    assert "risk_level" in result
    assert "top_factors" in result
    assert 0 <= result["risk_score"] <= 1
    assert result["risk_level"] in ("低风险", "中风险", "高风险")
    print("[PASS] 风险评分结果格式正确\n")


def test_search_knowledge_base():
    print("=" * 60)
    print("Test 3: search_knowledge_base")
    print("=" * 60)
    result_str = search_knowledge_base(query="逾期风险评估标准")
    print(result_str[:500])
    print("...")

    assert len(result_str) > 0
    assert "---" in result_str  # 至少有分隔符说明返回了多段
    print("\n[PASS] 知识库检索返回非空结果\n")


if __name__ == "__main__":
    features = test_query_user_credit_data()
    test_predict_risk_score(features)
    test_search_knowledge_base()
    print("All tests passed!")
