"""
类型 B：单步数据查询 — prompt 模板与变量池

生成目标：~360 条（遴选 ~225 条）
工具调用链：仅 query_user_credit_data（1 轮）
关键约束：只返回查询结果，不做深度分析
"""
import random
import json

from .type_d_explanation import SYSTEM_PROMPT, RISK_PROFILES

# ── 问法模板 ─────────────────────────────────────────────────────────
QUESTION_TEMPLATES = [
    "查一下客户{user_id}的收入和负债情况",
    "帮我看看客户{user_id}的基本信用数据",
    "客户{user_id}的逾期记录是什么",
    "查询客户{user_id}的信贷信息",
    "客户{user_id}目前的债务情况怎样",
    "帮我调一下客户{user_id}的征信数据",
    "客户{user_id}的月供和收入分别是多少",
    "看一下客户{user_id}有没有逾期",
    "客户{user_id}的活跃信用账户有几个",
    "查询客户{user_id}的贷款金额和还款信息",
    "客户{user_id}近期有申请其他贷款吗",
    "帮我查客户{user_id}的外部评分",
    "客户{user_id}的总负债是多少",
    "看看客户{user_id}最近有没有被拒记录",
    "客户{user_id}的收入来源和金额是什么",
]

# ── 查询侧重点（影响返回哪些字段）────────────────────────────────────
QUERY_FOCUSES = [
    "income_debt",       # 收入与负债
    "delinquency",       # 逾期记录
    "credit_info",       # 信贷信息
    "application_history", # 申请历史
    "overall",           # 综合概览
]

# ── Few-shot 示例 ────────────────────────────────────────────────────
FEW_SHOT = {
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "查一下客户100032的收入和负债情况"},
        {"role": "assistant", "content": "好的，我来查询该客户的信用数据。\n<tool_call>\n{\"name\": \"query_user_credit_data\", \"arguments\": {\"user_id\": 100032}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"mainoccupationinc_384A\": 55000, \"credamount_770A\": 120000, \"annuity_780A\": 4200, \"totaldebt_9A\": 95000, \"currdebt_22A\": 48000, \"maxdpdlast24m_143P\": 0, \"maxdpdlast3m_392P\": 0, \"numactivecreds_622L\": 2, \"applications30d_658L\": 0, \"debtoverdue_47A\": 0, \"numrejects9m_859L\": 0}\n</tool_response>"},
        {"role": "assistant", "content": "客户100032的收入和负债情况如下：\n\n- 主要职业收入：55,000元/月\n- 贷款金额：120,000元\n- 月供金额：4,200元/月\n- 总债务：95,000元\n- 当前债务：48,000元\n- 当前逾期金额：0元\n\n该客户目前有2个活跃信用账户，近24个月无逾期记录。"},
    ]
}


def _gen_credit_data(profile: dict) -> dict:
    """根据风险档位随机生成一组信用数据。"""
    data = {}
    for field, (lo, hi) in profile["credit_data_ranges"].items():
        if isinstance(lo, int) and isinstance(hi, int):
            data[field] = random.randint(lo, hi)
        else:
            data[field] = round(random.uniform(lo, hi), 2)
    return data


def build_prompt() -> tuple[str, dict]:
    """构建一条类型 B 的生成 prompt。"""
    profile_key = random.choice(["high", "medium", "low"])
    profile = RISK_PROFILES[profile_key]
    user_id = random.randint(100001, 999999)
    focus = random.choice(QUERY_FOCUSES)
    credit_data = _gen_credit_data(profile)

    question = random.choice(QUESTION_TEMPLATES).format(user_id=user_id)
    few_shot_json = json.dumps(FEW_SHOT, ensure_ascii=False, indent=2)

    prompt = f"""你是一个微调数据生成器。请严格按照 Qwen2.5 tool call 格式，生成一条"单步数据查询"类型的完整对话数据。

## 格式要求（极其重要，必须严格遵守）
1. 输出一个 JSON 对象，包含 "messages" 数组
2. messages[0] 的 role 为 "system"，content 为以下系统提示词（原样使用，不要修改）：
{json.dumps(SYSTEM_PROMPT, ensure_ascii=False)}

3. assistant 的思考过程直接写在 content 开头，然后跟 <tool_call> 标签
4. <tool_call> 内是单行 JSON，arguments 是 JSON 对象（不是字符串！）
5. 工具返回消息的 role 是 "user"（不是 "tool"），content 以 <tool_response> 开头
6. 最后一条 assistant 消息不包含 <tool_call> 标签
7. role 必须严格交替：system → user → assistant → user → assistant

## 场景参数
- 用户ID: {user_id}
- 用户问题: "{question}"
- 查询侧重: {focus}
- 客户信用数据（用于 query_user_credit_data 的模拟返回）: {json.dumps(credit_data, ensure_ascii=False)}

## 工具调用链
仅 1 轮：query_user_credit_data → 简洁回答

## 关键约束
- 只调用 query_user_credit_data 一次，不要调用 predict_risk_score 或其他工具
- 最终回答只呈现查询结果，不做风险分析或审批建议
- 回答简洁明了，用列表形式展示关键数据
- 回答长度 100-300 字

## 参考示例（严格模仿此格式）
{few_shot_json}

## 输出
只输出 JSON 对象，不要输出其他任何内容。"""

    metadata = {
        "type": "B",
        "user_id": user_id,
        "risk_profile": profile_key,
        "focus": focus,
    }
    return prompt, metadata
