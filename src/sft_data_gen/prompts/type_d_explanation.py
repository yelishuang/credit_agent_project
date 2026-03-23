"""
类型 D：模型解释 — prompt 模板与变量池

生成目标：~360 条（遴选 ~225 条）
工具调用链：query_user_credit_data → predict_risk_score → (可选 search_knowledge_base) → 解释性回答
"""
import random
import json

# ── 系统提示词（嵌入 Qwen2.5 tool call 格式）──────────────────────────
SYSTEM_PROMPT = """你是信用风险评估专家Agent。你拥有以下能力：查询用户数据、调用风险模型、检索风控知识库。
请先思考分析步骤，再调用合适的工具，最后给出完整的风险评估报告。

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "query_user_credit_data", "description": "根据用户ID查询其信用数据（收入、负债、逾期等）", "parameters": {"type": "object", "properties": {"user_id": {"type": "integer", "description": "用户ID"}}, "required": ["user_id"]}}}
{"type": "function", "function": {"name": "predict_risk_score", "description": "调用信用风险模型预测违约概率", "parameters": {"type": "object", "properties": {"features": {"type": "object", "description": "用户特征字典"}}, "required": ["features"]}}}
{"type": "function", "function": {"name": "search_knowledge_base", "description": "在风控知识库中搜索相关政策法规和业务规则", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "检索关键词"}, "top_k": {"type": "integer", "description": "返回条数，默认3"}}, "required": ["query"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "<function-name>", "arguments": <args-json-object>}
</tool_call>"""

# ── 问法模板 ─────────────────────────────────────────────────────────
QUESTION_TEMPLATES = [
    "为什么客户{user_id}被判定为{risk_level}？",
    "客户{user_id}的风险评分主要受什么因素影响？",
    "帮我分析一下客户{user_id}的风险评估结果",
    "客户{user_id}的违约概率为什么这么高？能解释一下吗？",
    "我想了解客户{user_id}被评为{risk_level}的具体原因",
    "客户{user_id}的信用评分只有{score}分，为什么？",
    "请解释客户{user_id}的风险评估报告",
    "客户{user_id}为什么会被模型标记为{risk_level}？哪些指标有问题？",
    "能不能详细说明客户{user_id}风险评分{score}背后的原因？",
    "客户{user_id}的哪些数据导致了{risk_level}的判定？",
    "帮我看看客户{user_id}的风险因素，为什么评分是{score}？",
    "客户{user_id}的信用状况怎么样？为什么被归为{risk_level}？",
    "分析一下客户{user_id}的风险评分为什么偏高",
    "客户{user_id}的风险等级是怎么得出来的？",
    "为什么系统给客户{user_id}打了{score}的风险分？",
    "客户{user_id}有哪些风险点？详细解释一下",
]

# ── 风险档位配置 ──────────────────────────────────────────────────────
RISK_PROFILES = {
    "high": {
        "score_range": (0.60, 0.95),
        "level": "高风险",
        "credit_data_ranges": {
            "mainoccupationinc_384A": (3000, 30000),
            "credamount_770A": (200000, 800000),
            "annuity_780A": (8000, 25000),
            "totaldebt_9A": (300000, 1200000),
            "currdebt_22A": (150000, 600000),
            "maxdpdlast24m_143P": (30, 180),
            "maxdpdlast3m_392P": (15, 90),
            "numactivecreds_622L": (3, 12),
            "applications30d_658L": (2, 8),
            "numrejects9m_859L": (1, 6),
            "debtoverdue_47A": (10000, 200000),
            "days30_165L": (3, 15),
            "numinstlswithdpd5_4187116L": (3, 20),
        },
        "typical_factors": [
            "负债收入比过高", "近期逾期记录严重", "多头借贷明显",
            "总债务金额远超收入", "近期频繁申请贷款", "当前逾期金额较大",
            "信用查询次数过多", "逾期分期占比高",
        ],
    },
    "medium": {
        "score_range": (0.30, 0.60),
        "level": "中风险",
        "credit_data_ranges": {
            "mainoccupationinc_384A": (25000, 80000),
            "credamount_770A": (80000, 300000),
            "annuity_780A": (3000, 10000),
            "totaldebt_9A": (50000, 300000),
            "currdebt_22A": (30000, 150000),
            "maxdpdlast24m_143P": (5, 30),
            "maxdpdlast3m_392P": (0, 15),
            "numactivecreds_622L": (2, 6),
            "applications30d_658L": (0, 3),
            "numrejects9m_859L": (0, 2),
            "debtoverdue_47A": (0, 20000),
            "days30_165L": (1, 5),
            "numinstlswithdpd5_4187116L": (0, 5),
        },
        "typical_factors": [
            "负债收入比偏高", "有轻微逾期记录", "信贷金额与收入不太匹配",
            "活跃信用账户偏多", "部分还款不够及时", "征信查询次数偏多",
        ],
    },
    "low": {
        "score_range": (0.03, 0.30),
        "level": "低风险",
        "credit_data_ranges": {
            "mainoccupationinc_384A": (60000, 200000),
            "credamount_770A": (10000, 100000),
            "annuity_780A": (500, 4000),
            "totaldebt_9A": (0, 60000),
            "currdebt_22A": (0, 30000),
            "maxdpdlast24m_143P": (0, 5),
            "maxdpdlast3m_392P": (0, 0),
            "numactivecreds_622L": (0, 3),
            "applications30d_658L": (0, 1),
            "numrejects9m_859L": (0, 0),
            "debtoverdue_47A": (0, 0),
            "days30_165L": (0, 2),
            "numinstlswithdpd5_4187116L": (0, 1),
        },
        "typical_factors": [
            "收入稳定且充足", "负债水平健康", "无逾期记录",
            "信贷使用率低", "还款记录良好", "申请行为正常",
        ],
    },
}

# ── 解释侧重点 ───────────────────────────────────────────────────────
EXPLANATION_FOCUSES = [
    "debt_income_ratio",
    "delinquency_history",
    "credit_utilization",
    "application_behavior",
    "income_stability",
]

FOCUS_DESCRIPTIONS = {
    "debt_income_ratio": "重点分析负债收入比及其对风险评分的影响",
    "delinquency_history": "重点分析逾期历史及还款行为模式",
    "credit_utilization": "重点分析信用使用率和授信额度利用情况",
    "application_behavior": "重点分析申请行为（频率、多头借贷、拒绝历史）",
    "income_stability": "重点分析收入水平与稳定性对风险的影响",
}

# ── RAG 检索结果素材（用于 3 轮变体）────────────────────────────────
RAG_RESPONSE_TEMPLATES = [
    "根据《商业银行信用风险管理指引》第23条，对于风险评分在0.6-0.8区间的中高风险客户，应当：1）要求追加担保或抵押；2）降低授信额度至原申请的60%-80%；3）缩短贷款期限；4）提高贷款利率以覆盖风险溢价。",
    "根据《个人贷款管理暂行办法》第16条，贷款人应建立借款人合理的收入偿债比例控制机制。实践中，月还款额不应超过借款人月收入的50%，对于高风险客户应进一步收紧至40%。",
    "根据巴塞尔协议III零售信贷风险权重规定，违约概率PD在0.30-0.50区间的客户属于中高风险，对应内部评级B级，风险权重约75%-150%。银行应计提更高的资本准备金。",
    "根据《征信业管理条例》和央行征信中心规定，近6个月内征信查询超过6次属于查询过于频繁，可能反映借款人资金紧张或多头借贷行为，是重要的风险预警信号。",
    "根据消费金融行业风控实践，负债收入比（DTI）超过50%即进入警戒区间，超过70%属于高风险。DTI的计算应包含所有已知负债的月还款额与月收入的比值。",
    "根据《商业银行互联网贷款管理暂行办法》第20条，商业银行应当建立有效的风险评估机制，对借款人的还款能力、还款意愿、多头借贷等进行综合评估。近30天内申请次数超过3次应触发人工复核。",
    "根据信用评分卡原理，PD（违约概率）本质上是群体统计概率。PD=0.15意味着100个类似客户中约有15个会在未来12个月内违约（逾期90天以上）。预期损失EL=PD×LGD×EAD，无抵押消费贷LGD通常在60%-80%。",
    "根据《商业银行风险管理指引》，对于存在逾期记录的客户，应重点关注逾期频率、逾期金额和逾期持续时间三个维度。近3个月内出现逾期超过15天的记录，应列为重点关注客户。",
]

# ── Few-shot 示例（2 轮变体，无 RAG）─────────────────────────────────
FEW_SHOT_2ROUND = {
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "为什么客户100058被判定为高风险？"},
        {"role": "assistant", "content": "我需要先查询该客户的信用数据，了解其基本财务状况，然后调用风险模型获取评分详情。\n<tool_call>\n{\"name\": \"query_user_credit_data\", \"arguments\": {\"user_id\": 100058}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"mainoccupationinc_384A\": 18000, \"credamount_770A\": 450000, \"annuity_780A\": 15200, \"totaldebt_9A\": 520000, \"currdebt_22A\": 280000, \"maxdpdlast24m_143P\": 67, \"maxdpdlast3m_392P\": 32, \"numactivecreds_622L\": 7, \"applications30d_658L\": 4, \"debtoverdue_47A\": 45000, \"numrejects9m_859L\": 3}\n</tool_response>"},
        {"role": "assistant", "content": "该客户负债水平很高，收入仅1.8万但总债务达52万，且有较严重的逾期记录。我来调用风险模型量化评估。\n<tool_call>\n{\"name\": \"predict_risk_score\", \"arguments\": {\"features\": {\"mainoccupationinc_384A\": 18000, \"credamount_770A\": 450000, \"annuity_780A\": 15200, \"totaldebt_9A\": 520000, \"currdebt_22A\": 280000, \"maxdpdlast24m_143P\": 67, \"maxdpdlast3m_392P\": 32, \"numactivecreds_622L\": 7, \"applications30d_658L\": 4, \"debtoverdue_47A\": 45000, \"numrejects9m_859L\": 3}}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"risk_score\": 0.78, \"risk_level\": \"高风险\", \"top_factors\": [\"负债收入比过高（值: 28.89）\", \"近24个月最大逾期天数（值: 67）\", \"当前逾期金额（值: 45000）\", \"近9个月拒绝次数（值: 3）\", \"活跃信用账户数（值: 7）\"]}\n</tool_response>"},
        {"role": "assistant", "content": "## 客户100058风险评分解释\n\n**风险评分: 0.78（高风险）**\n\n基于该客户的数据特征，以下是影响其被判定为高风险的主要因素分析：\n\n### 1. 负债收入比严重超标\n该客户月收入仅18,000元，但总债务高达520,000元，负债收入比约28.89倍，远超行业警戒线（通常为3-5倍）。月供15,200元已占月收入的84.4%，远超50%的安全阈值，还款压力极大。\n\n### 2. 逾期记录严重\n近24个月最大逾期天数达67天，已超过M2（逾期60天）标准；近3个月仍有32天逾期，说明逾期问题持续存在且未改善。当前仍有45,000元逾期未还。\n\n### 3. 多头借贷明显\n活跃信用账户多达7个，近30天内申请了4次贷款，近9个月被拒绝3次。频繁申请和多头借贷是资金链紧张的典型信号。\n\n### 4. 综合判断\n该客户的核心问题是收入与负债严重不匹配，叠加持续逾期和多头借贷行为，违约风险显著高于正常客群。风险评分0.78意味着类似特征的客户群体中约78%会在未来12个月内发生违约。\n\n> 注：以上分析基于客户数据特征的风险因素推断，供审批参考。"},
    ]
}

# ── Few-shot 示例（3 轮变体，含 RAG）─────────────────────────────────
FEW_SHOT_3ROUND = {
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "客户200015的风险评分主要受什么因素影响？"},
        {"role": "assistant", "content": "我先查询该客户的信用数据。\n<tool_call>\n{\"name\": \"query_user_credit_data\", \"arguments\": {\"user_id\": 200015}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"mainoccupationinc_384A\": 45000, \"credamount_770A\": 180000, \"annuity_780A\": 6500, \"totaldebt_9A\": 160000, \"currdebt_22A\": 85000, \"maxdpdlast24m_143P\": 12, \"maxdpdlast3m_392P\": 5, \"numactivecreds_622L\": 4, \"applications30d_658L\": 1, \"debtoverdue_47A\": 3500, \"numrejects9m_859L\": 1}\n</tool_response>"},
        {"role": "assistant", "content": "客户有一定负债且存在轻微逾期，我来调用风险模型进行量化评估。\n<tool_call>\n{\"name\": \"predict_risk_score\", \"arguments\": {\"features\": {\"mainoccupationinc_384A\": 45000, \"credamount_770A\": 180000, \"annuity_780A\": 6500, \"totaldebt_9A\": 160000, \"currdebt_22A\": 85000, \"maxdpdlast24m_143P\": 12, \"maxdpdlast3m_392P\": 5, \"numactivecreds_622L\": 4, \"applications30d_658L\": 1, \"debtoverdue_47A\": 3500, \"numrejects9m_859L\": 1}}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n{\"risk_score\": 0.42, \"risk_level\": \"中风险\", \"top_factors\": [\"负债收入比偏高（值: 3.56）\", \"近24个月最大逾期天数（值: 12）\", \"活跃信用账户数（值: 4）\", \"当前逾期金额（值: 3500）\", \"近9个月拒绝次数（值: 1）\"]}\n</tool_response>"},
        {"role": "assistant", "content": "该客户属于中风险，我检索一下相关风控政策来补充说明。\n<tool_call>\n{\"name\": \"search_knowledge_base\", \"arguments\": {\"query\": \"中风险客户审批标准与负债收入比要求\", \"top_k\": 3}}\n</tool_call>"},
        {"role": "user", "content": "<tool_response>\n根据《个人贷款管理暂行办法》第16条，贷款人应建立借款人合理的收入偿债比例控制机制。实践中，月还款额不应超过借款人月收入的50%，对于高风险客户应进一步收紧至40%。\n</tool_response>"},
        {"role": "assistant", "content": "## 客户200015风险评分因素分析\n\n**风险评分: 0.42（中风险）**\n\n### 主要影响因素\n\n**1. 负债收入比偏高（DTI ≈ 3.56）**\n客户月收入45,000元，总债务160,000元，负债收入比约3.56倍。虽未达到高风险阈值（5倍以上），但已高于健康水平（2倍以内）。月供6,500元占月收入14.4%，尚在可控范围，但需关注总负债压力。\n\n**2. 存在轻微逾期记录**\n近24个月最大逾期12天，近3个月有5天逾期，当前仍有3,500元逾期未还。虽然逾期天数不长，但持续存在的逾期行为反映还款习惯有待改善。\n\n**3. 多账户管理压力**\n活跃信用账户4个，近9个月有1次申请被拒。多账户本身不构成高风险，但结合逾期记录，说明客户可能在多个平台间周转。\n\n### 政策参考\n根据《个人贷款管理暂行办法》第16条，月还款额不应超过月收入的50%。该客户月供占比14.4%符合要求，但考虑到其他负债的月还款压力，实际偿债比可能更高。\n\n### 改善建议\n- 优先清偿当前3,500元逾期，改善征信记录\n- 控制新增借贷，减少活跃账户数\n- 如能将总负债降至10万以内，风险评分有望改善至低风险区间\n\n> 注：以上分析基于客户数据特征的风险因素推断，供审批参考。"},
    ]
}


# ── 构建 prompt 的核心函数 ────────────────────────────────────────────
def _gen_credit_data(profile: dict) -> dict:
    """根据风险档位随机生成一组信用数据。"""
    data = {}
    for field, (lo, hi) in profile["credit_data_ranges"].items():
        if isinstance(lo, int) and isinstance(hi, int):
            data[field] = random.randint(lo, hi)
        else:
            data[field] = round(random.uniform(lo, hi), 2)
    return data


def _gen_risk_result(score: float, level: str, factors: list) -> dict:
    """生成 predict_risk_score 的模拟返回。"""
    selected = random.sample(factors, min(5, len(factors)))
    return {"risk_score": round(score, 2), "risk_level": level, "top_factors": selected}


def build_prompt(variant_index: int = None) -> tuple[str, dict]:
    """
    构建一条发给 Claude 的生成 prompt。

    返回: (prompt_text, metadata)
        metadata 包含 user_id, risk_profile, variant, focus 等信息
    """
    # 随机选择风险档位（各 1/3）
    profile_key = random.choice(["high", "medium", "low"])
    profile = RISK_PROFILES[profile_key]

    # 随机选择变体：60% 2轮，40% 3轮
    variant = "2round" if random.random() < 0.6 else "3round"

    # 随机参数
    user_id = random.randint(100001, 999999)
    score = round(random.uniform(*profile["score_range"]), 2)
    focus = random.choice(EXPLANATION_FOCUSES)
    credit_data = _gen_credit_data(profile)

    # 选择问法模板并填充
    question_tpl = random.choice(QUESTION_TEMPLATES)
    question = question_tpl.format(
        user_id=user_id,
        risk_level=profile["level"],
        score=score,
    )

    # 选择 few-shot
    few_shot = FEW_SHOT_2ROUND if variant == "2round" else FEW_SHOT_3ROUND
    few_shot_json = json.dumps(few_shot, ensure_ascii=False, indent=2)

    # RAG 相关指令
    rag_instruction = ""
    rag_response = ""
    if variant == "3round":
        rag_response = random.choice(RAG_RESPONSE_TEMPLATES)
        rag_instruction = f"""
在 predict_risk_score 返回后，还需要调用 search_knowledge_base 检索相关政策法规，并在最终解释中引用政策依据。
模拟的 search_knowledge_base 返回内容参考："{rag_response}"（你可以基于此适当改写，使其与具体场景匹配）。"""

    prompt = f"""你是一个微调数据生成器。请严格按照 Qwen2.5 tool call 格式，生成一条"模型解释"类型的完整多轮对话数据。

## 格式要求（极其重要，必须严格遵守）
1. 输出一个 JSON 对象，包含 "messages" 数组
2. messages[0] 的 role 为 "system"，content 为以下系统提示词（原样使用，不要修改）：
{json.dumps(SYSTEM_PROMPT, ensure_ascii=False)}

3. assistant 的思考过程直接写在 content 开头（自由文本），然后跟 <tool_call> 标签
4. <tool_call> 内是单行 JSON，arguments 是 JSON 对象（不是字符串！）
5. 工具返回消息的 role 是 "user"（不是 "tool"），content 以 <tool_response> 开头
6. 最后一条 assistant 消息不包含 <tool_call> 标签，是最终的解释性回答
7. role 必须严格交替：system → user → assistant → user → assistant → ...

## 场景参数
- 用户ID: {user_id}
- 用户问题: "{question}"
- 风险档位: {profile["level"]}（评分 {score}）
- 解释侧重: {FOCUS_DESCRIPTIONS[focus]}
- 客户信用数据（用于 query_user_credit_data 的模拟返回）: {json.dumps(credit_data, ensure_ascii=False)}
- 风险评分结果（用于 predict_risk_score 的模拟返回）: 评分 {score}，等级 "{profile["level"]}"，主要因素从以下选取: {json.dumps(profile["typical_factors"], ensure_ascii=False)}

## 工具调用链
{"2轮：query_user_credit_data → predict_risk_score → 最终解释" if variant == "2round" else "3轮：query_user_credit_data → predict_risk_score → search_knowledge_base → 最终解释"}
{rag_instruction}

## 最终回答要求
- 必须解释为什么该客户被判定为该风险等级
- 包含具体数值对比（如"负债收入比X倍，超过阈值Y"）
- 分析至少3个主要风险因素
- {"包含政策法规引用" if variant == "3round" else "不需要引用政策"}
- 末尾加注："注：以上分析基于客户数据特征的风险因素推断，供审批参考。"
- 回答长度 300-600 字

## 参考示例（严格模仿此格式）
{few_shot_json}

## 输出
只输出 JSON 对象，不要输出其他任何内容（不要 markdown 代码块包裹）。"""

    metadata = {
        "type": "D",
        "user_id": user_id,
        "risk_profile": profile_key,
        "variant": variant,
        "focus": focus,
        "score": score,
    }
    return prompt, metadata
