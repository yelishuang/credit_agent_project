"""
类型 A：完整审批请求 — prompt 模板与变量池

生成目标：~960 条（遴选 ~600 条）
工具调用链：query_user_credit_data → predict_risk_score → search_knowledge_base → 最终审批报告
特点：3-4 轮工具调用，每轮有 Thought，最终输出结构化审批报告
"""
import random
import json

from .type_d_explanation import SYSTEM_PROMPT, RISK_PROFILES

# ── 问法模板 ─────────────────────────────────────────────────────────
QUESTION_TEMPLATES = [
    "帮我审批客户ID={user_id}的贷款申请",
    "评估用户{user_id}的信用风险",
    "客户{user_id}申请了{loan_amount}万的{loan_type}，帮我做个风险评估",
    "对客户{user_id}进行完整的信贷审批",
    "帮我看看客户{user_id}能不能通过贷款审批",
    "客户{user_id}的贷款申请需要审批，请评估",
    "请对客户{user_id}做一个全面的信用风险评估",
    "客户{user_id}想申请{loan_type}，金额{loan_amount}万，帮我审核",
    "新的贷款申请：客户{user_id}，请进行风险评估",
    "帮我处理客户{user_id}的信贷审批请求",
    "客户{user_id}提交了贷款申请，请做风险评估并给出审批建议",
    "评估一下客户{user_id}是否符合贷款条件",
    "客户{user_id}申请贷款，帮我跑一下风控流程",
    "请对客户{user_id}的贷款资质进行审查",
    "帮我完成客户{user_id}的信贷审批全流程",
]

# ── 贷款类型 ─────────────────────────────────────────────────────────
LOAN_TYPES = ["消费贷", "现金贷", "信用贷", "装修贷", "教育贷", "医疗贷", "旅游贷"]

# ── RAG 检索结果素材（审批政策类）─────────────────────────────────────
RAG_RESPONSES = [
    "根据《商业银行信用风险管理指引》第23条，对于风险评分在0.6-0.8区间的中高风险客户，应当：1）要求追加担保或抵押；2）降低授信额度至原申请的60%-80%；3）缩短贷款期限；4）提高贷款利率以覆盖风险溢价。",
    "根据《个人贷款管理暂行办法》第16条，贷款人应建立借款人合理的收入偿债比例控制机制。月还款额不应超过借款人月收入的50%，对于高风险客户应进一步收紧至40%。",
    "根据消费金融行业风控实践，低风险客户（PD<0.10）可直接通过，给予标准额度；中风险客户（PD 0.10-0.30）建议适当降额并加强贷后监控；高风险客户（PD>0.30）建议拒绝或要求追加强担保。",
    "根据《商业银行互联网贷款管理暂行办法》第20条，单户用于消费的个人信用贷款授信额度应当不超过人民币20万元。贷款资金不得用于购房、股票、期货等投资。",
    "根据巴塞尔协议III零售信贷风险权重规定，违约概率PD在0.10-0.20区间的客户属于中低风险，对应内部评级BBB级。建议通过但适当降额，预期损失率参考1%-2%。",
    "根据《征信业管理条例》，贷款审批应综合考虑借款人征信记录、还款能力、担保情况等因素。对于征信查询次数过多（近6个月超6次）的客户，应触发人工复核流程。",
    "根据信用风险管理最佳实践，对于PD在0.30-0.50区间的中高风险客户，建议：1）授信额度不超过申请金额的50%；2）贷款期限不超过24个月；3）利率上浮20%-40%；4）要求提供第三方担保或资产抵押。",
    "根据《商业银行风险管理指引》，对于首次申请且无征信记录的客户（白户），应采取审慎策略：初始额度不超过5万元，期限不超过12个月，并在3-6个月后根据还款表现调整额度。",
]

# ── 审批建议模板 ─────────────────────────────────────────────────────
APPROVAL_DECISIONS = {
    "high": ["建议拒绝", "有条件通过（需强担保）"],
    "medium": ["有条件通过", "有条件通过（建议降额）"],
    "low": ["建议通过", "建议通过，可给予标准额度"],
}

# ── Few-shot 示例（3 轮完整审批）─────────────────────────────────────
FEW_SHOT = {
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "评估用户100002的信用风险"},
        {"role": "assistant", "content": "我需要先查询该用户的信用数据，了解其基本财务状况。\n<tool_call>\n{\"name\": \"query_user_credit_data\", \"arguments\": {\"user_id\": 100002}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"mainoccupationinc_384A\": 22500, \"credamount_770A\": 129350, \"annuity_780A\": 3570, \"totaldebt_9A\": 145000, \"currdebt_22A\": 78000, \"maxdpdlast24m_143P\": 45, \"maxdpdlast3m_392P\": 18, \"numactivecreds_622L\": 5, \"applications30d_658L\": 3, \"debtoverdue_47A\": 12000, \"numrejects9m_859L\": 2}\n</tool_response>"},
        {"role": "assistant", "content": "用户信贷金额较高，且有逾期记录，需要量化评估违约风险。\n<tool_call>\n{\"name\": \"predict_risk_score\", \"arguments\": {\"features\": {\"mainoccupationinc_384A\": 22500, \"credamount_770A\": 129350, \"annuity_780A\": 3570, \"totaldebt_9A\": 145000, \"currdebt_22A\": 78000, \"maxdpdlast24m_143P\": 45, \"maxdpdlast3m_392P\": 18, \"numactivecreds_622L\": 5, \"applications30d_658L\": 3, \"debtoverdue_47A\": 12000, \"numrejects9m_859L\": 2}}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"risk_score\": 0.68, \"risk_level\": \"高风险\", \"top_factors\": [\"负债收入比过高（值: 6.44）\", \"近24个月最大逾期天数（值: 45）\", \"当前逾期金额（值: 12000）\", \"近30天申请次数（值: 3）\", \"近9个月拒绝次数（值: 2）\"]}\n</tool_response>"},
        {"role": "assistant", "content": "风险评分0.68属于高风险，需要查询相关风控政策来给出合规的审批建议。\n<tool_call>\n{\"name\": \"search_knowledge_base\", \"arguments\": {\"query\": \"高风险客户信贷审批标准\", \"top_k\": 3}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n根据《商业银行信用风险管理指引》第23条，对于风险评分在0.6-0.8区间的中高风险客户，应当：1）要求追加担保或抵押；2）降低授信额度至原申请的60%-80%；3）缩短贷款期限；4）提高贷款利率以覆盖风险溢价。\n</tool_response>"},
        {"role": "assistant", "content": "## 信用风险评估报告\n\n**用户ID:** 100002\n**综合风险等级: 高风险（评分 0.68）**\n\n### 一、关键指标分析\n- 月收入: 22,500元\n- 信贷金额: 129,350元\n- 负债收入比: 6.44倍（警戒阈值: 3.0倍）⚠️\n- 月供: 3,570元，占月收入15.9%\n- 总债务: 145,000元\n- 当前逾期金额: 12,000元\n\n### 二、主要风险因素\n1. **负债收入比过高（6.44倍）**: 远超3.0倍的健康阈值，借款人承担了与收入不匹配的债务负担\n2. **逾期记录严重**: 近24个月最大逾期45天（M1+），近3个月仍有18天逾期，当前仍有12,000元逾期未还\n3. **多头借贷迹象**: 活跃信用账户5个，近30天申请3次，近9个月被拒2次\n\n### 三、政策依据\n根据《商业银行信用风险管理指引》第23条，该客户处于0.6-0.8高风险区间，应当要求追加担保、降低授信额度、缩短期限并上浮利率。\n\n### 四、审批建议\n**有条件通过** — 需满足以下条件：\n1. 要求提供房产抵押或第三方担保\n2. 授信额度降至77,610-103,480元（原申请的60%-80%）\n3. 贷款期限不超过24个月\n4. 利率在基准基础上上浮20%-30%\n5. 要求先清偿当前12,000元逾期\n\n> 注：以上评估基于客户数据特征分析，供审批决策参考。"},
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
    """构建一条类型 A 的生成 prompt。"""
    profile_key = random.choice(["high", "medium", "low"])
    profile = RISK_PROFILES[profile_key]

    user_id = random.randint(100001, 999999)
    score = round(random.uniform(*profile["score_range"]), 2)
    loan_type = random.choice(LOAN_TYPES)
    loan_amount = random.choice([5, 8, 10, 15, 20, 30, 50])
    credit_data = _gen_credit_data(profile)
    rag_response = random.choice(RAG_RESPONSES)
    decision = random.choice(APPROVAL_DECISIONS[profile_key])

    question_tpl = random.choice(QUESTION_TEMPLATES)
    question = question_tpl.format(
        user_id=user_id,
        loan_type=loan_type,
        loan_amount=loan_amount,
    )

    few_shot_json = json.dumps(FEW_SHOT, ensure_ascii=False, indent=2)

    prompt = f"""你是一个微调数据生成器。请严格按照 Qwen2.5 tool call 格式，生成一条"完整审批请求"类型的完整多轮对话数据。

## 格式要求（极其重要，必须严格遵守）
1. 输出一个 JSON 对象，包含 "messages" 数组
2. messages[0] 的 role 为 "system"，content 为以下系统提示词（原样使用，不要修改）：
{json.dumps(SYSTEM_PROMPT, ensure_ascii=False)}

3. assistant 的思考过程直接写在 content 开头，然后跟 <tool_call> 标签
4. <tool_call> 内是单行 JSON，arguments 是 JSON 对象（不是字符串！）
5. 工具返回消息的 role 是 "user"（不是 "tool"），content 以 <tool_response> 开头
6. 最后一条 assistant 消息不包含 <tool_call> 标签，是结构化审批报告
7. role 必须严格交替：system → user → assistant → user → assistant → ...

## 场景参数
- 用户ID: {user_id}
- 用户问题: "{question}"
- 贷款类型: {loan_type}
- 申请金额: {loan_amount}万元
- 风险档位: {profile["level"]}（评分 {score}）
- 客户信用数据（用于 query_user_credit_data 的模拟返回）: {json.dumps(credit_data, ensure_ascii=False)}
- 风险评分结果（用于 predict_risk_score 的模拟返回）: 评分 {score}，等级 "{profile["level"]}"，主要因素从以下选取: {json.dumps(profile["typical_factors"], ensure_ascii=False)}
- RAG 检索结果（用于 search_knowledge_base 的模拟返回）: "{rag_response}"
- 审批建议方向: {decision}

## 工具调用链
3 轮：query_user_credit_data → predict_risk_score → search_knowledge_base → 结构化审批报告

## 最终报告要求
- 使用 markdown 格式的结构化审批报告
- 包含：用户ID、风险等级、关键指标分析、主要风险因素、政策依据、审批建议
- 审批建议要具体（通过/有条件通过/拒绝 + 具体条件）
- 末尾加注："注：以上评估基于客户数据特征分析，供审批决策参考。"
- 报告长度 400-700 字

## 参考示例（严格模仿此格式）
{few_shot_json}

## 输出
只输出 JSON 对象，不要输出其他任何内容。"""

    metadata = {
        "type": "A",
        "user_id": user_id,
        "risk_profile": profile_key,
        "loan_type": loan_type,
        "loan_amount": loan_amount,
        "score": score,
        "decision": decision,
    }
    return prompt, metadata
