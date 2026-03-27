"""
CreditAgent MCP Server — stdio 通信
注册三个工具：query_user_credit_data, predict_risk_score, search_knowledge_base
"""

import os
import sys
import json
import math
import logging

# 离线模式，防止 HuggingFace 联网
os.environ["HF_HUB_OFFLINE"] = "1"

# 把项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import RAG_INDEX_PATH, RAG_CHUNKS_PATH, EMBEDDING_MODEL
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("CreditAgent Tools")

# ---------------------------------------------------------------------------
# 列名映射：SFT 期望字段 -> parquet 实际列名
# ---------------------------------------------------------------------------
FIELD_MAPPING = {
    "mainoccupationinc_384A": "max_mainoccupationinc_384A",
    "debtoverdue_47A": "max_debtoverdue_47A",
}

SFT_FIELDS = [
    "mainoccupationinc_384A",
    "credamount_770A",
    "annuity_780A",
    "totaldebt_9A",
    "currdebt_22A",
    "maxdpdlast24m_143P",
    "maxdpdlast3m_392P",
    "numactivecreds_622L",
    "applications30d_658L",
    "numrejects9m_859L",
    "debtoverdue_47A",
    "days30_165L",
    "numinstlswithdpd5_4187116L",
]

def _clean_value(v):
    """将 NaN/Inf 转为 None，float 整数转 int，保持 JSON 可序列化。"""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        if v == int(v):
            return int(v)
    return v


# ---------------------------------------------------------------------------
# Tool 1: query_user_credit_data
# ---------------------------------------------------------------------------
@mcp.tool()
def query_user_credit_data(user_id: int) -> str:
    """根据 user_id (case_id) 从测试集中查询客户的关键信贷特征。"""
    from src.credit_risk_model.feature_engineering import build_features_from_parquet

    df = build_features_from_parquet(case_ids=[user_id], split="test")
    if df.height == 0:
        return json.dumps({"error": f"未找到 user_id={user_id} 的数据"}, ensure_ascii=False)

    row = df.row(0, named=True)
    result = {}
    for field in SFT_FIELDS:
        col = FIELD_MAPPING.get(field, field)
        result[field] = _clean_value(row.get(col))

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2: predict_risk_score
# ---------------------------------------------------------------------------
@mcp.tool()
def predict_risk_score(features: dict) -> str:
    """输入客户特征字典，调用 LightGBM 模型预测违约概率。"""
    from src.credit_risk_model.predict import predict_credit_risk

    # 将 SFT 字段名映射回模型实际使用的列名（如 mainoccupationinc_384A → max_mainoccupationinc_384A）
    feat_dict = {FIELD_MAPPING.get(k, k): v for k, v in features.items()}
    result = predict_credit_risk(feat_dict)
    return json.dumps(result, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Tool 3: search_knowledge_base
# ---------------------------------------------------------------------------
_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.rag.build_rag_index import load_retriever
        _retriever = load_retriever(str(RAG_INDEX_PATH), str(RAG_CHUNKS_PATH), EMBEDDING_MODEL)
    return _retriever


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """从 FAISS 向量索引中检索与 query 相关的风控知识文档片段。"""
    search_fn = _get_retriever()
    results = search_fn(query, top_k=3)
    # 拼接 top 3 片段
    texts = [r["content"] for r in results]
    return "\n\n---\n\n".join(texts)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
