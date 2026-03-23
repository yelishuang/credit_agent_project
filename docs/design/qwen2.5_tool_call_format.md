# Qwen2.5-14B-Instruct Tool Call 格式规范

> 本文档为项目核心约束，所有微调数据构造、Agent 编排、Prompt 模板设计必须严格遵循此格式。

---

## 一、特殊 Token

| Token | 作用 |
|-------|------|
| `<\|im_start\|>` | 消息开始标记 |
| `<\|im_end\|>` | 消息结束标记 |
| `<tools>` / `</tools>` | 工具定义区域（system prompt 内） |
| `<tool_call>` / `</tool_call>` | assistant 发起工具调用 |
| `<tool_response>` / `</tool_response>` | 工具执行结果返回 |

---

## 二、System Prompt 中的工具定义格式

工具定义放在 system prompt 末尾，用 `<tools></tools>` 包裹，每个工具一行 JSON：

```
<|im_start|>system
你是信用风险评估专家Agent。你拥有以下能力：查询用户数据、调用风险模型、检索风控知识库。
请先思考分析步骤，再调用合适的工具，最后给出完整的风险评估报告。

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "query_user_credit_data", "description": "根据用户ID查询其信用数据（收入、负债、逾期等）", "parameters": {"type": "object", "properties": {"user_id": {"type": "integer", "description": "用户ID"}}, "required": ["user_id"]}}}
{"type": "function", "function": {"name": "predict_risk_score", "description": "调用信用风险模型预测违约概率", "parameters": {"type": "object", "properties": {"features": {"type": "object", "description": "用户特征字典"}}, "required": ["features"]}}}
{"type": "function", "function": {"name": "search_knowledge_base", "description": "在风控知识库中搜索相关政策法规和业务规则", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "检索关键词"}, "top_k": {"type": "integer", "description": "返回条数，默认3"}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "generate_risk_report", "description": "根据分析结果生成结构化风险评估报告", "parameters": {"type": "object", "properties": {"user_id": {"type": "integer"}, "risk_score": {"type": "number"}, "risk_factors": {"type": "string"}, "policy_reference": {"type": "string"}}, "required": ["user_id", "risk_score", "risk_factors", "policy_reference"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "<function-name>", "arguments": <args-json-object>}
</tool_call>
<|im_end|>
```

### 工具定义 JSON Schema 结构

```json
{
  "type": "function",
  "function": {
    "name": "函数名（英文snake_case）",
    "description": "函数功能描述",
    "parameters": {
      "type": "object",
      "properties": {
        "参数名": {
          "type": "参数类型（string/integer/number/object/array/boolean）",
          "description": "参数说明"
        }
      },
      "required": ["必填参数列表"]
    }
  }
}
```

---

## 三、Assistant 工具调用输出格式

assistant 的 content 中，Thought（思考过程）在前，`<tool_call>` 标签在后：

```
<|im_start|>assistant
我需要先查询该用户的信用数据，了解其基本财务状况。
<tool_call>
{"name": "query_user_credit_data", "arguments": {"user_id": 100002}}
</tool_call>
<|im_end|>
```

### 关键约束

1. **Thought 是自由文本**，直接写在 content 开头，不需要特殊标签
2. **`<tool_call>` 内是纯 JSON**，不允许有注释、多余空格或换行（单行 JSON）
3. **`arguments` 的值是 JSON 对象**，不是字符串（区别于 OpenAI 格式的 `"arguments": "{...}"`）
4. **一次可以调用多个工具**，每个用独立的 `<tool_call></tool_call>` 包裹
5. **没有 `<tool_call>` 标签 = 最终回答**，Agent 循环终止条件

### 单次多工具调用示例

```
<|im_start|>assistant
我需要同时查询用户数据和相关政策。
<tool_call>
{"name": "query_user_credit_data", "arguments": {"user_id": 100002}}
</tool_call>
<tool_call>
{"name": "search_knowledge_base", "arguments": {"query": "信贷审批标准"}}
</tool_call>
<|im_end|>
```

---

## 四、工具结果返回格式

工具执行结果以 `role: "user"` 角色返回，用 `<tool_response>` 标签包裹：

```
<|im_start|>user
<tool_response>
{"income": 270000, "credit_amount": 1293502, "annuity": 35698}
</tool_response>
<|im_end|>
```

### 关键约束

1. **role 是 "user"，不是 "tool"**（区别于 OpenAI 格式）
2. **`<tool_response>` 内容是工具返回的原始结果**，通常为 JSON 字符串或纯文本
3. **多个工具的结果可以放在同一个 user 消息中**，每个用独立的 `<tool_response>` 包裹

---

## 五、完整多轮对话示例（微调数据标准模板）

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是信用风险评估专家Agent。你拥有以下能力：查询用户数据、调用风险模型、检索风控知识库。\n请先思考分析步骤，再调用合适的工具，最后给出完整的风险评估报告。\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"query_user_credit_data\", \"description\": \"根据用户ID查询其信用数据（收入、负债、逾期等）\", \"parameters\": {\"type\": \"object\", \"properties\": {\"user_id\": {\"type\": \"integer\", \"description\": \"用户ID\"}}, \"required\": [\"user_id\"]}}}\n{\"type\": \"function\", \"function\": {\"name\": \"predict_risk_score\", \"description\": \"调用信用风险模型预测违约概率\", \"parameters\": {\"type\": \"object\", \"properties\": {\"features\": {\"type\": \"object\", \"description\": \"用户特征字典\"}}, \"required\": [\"features\"]}}}\n{\"type\": \"function\", \"function\": {\"name\": \"search_knowledge_base\", \"description\": \"在风控知识库中搜索相关政策法规和业务规则\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"检索关键词\"}, \"top_k\": {\"type\": \"integer\", \"description\": \"返回条数，默认3\"}}, \"required\": [\"query\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": \"<function-name>\", \"arguments\": <args-json-object>}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "评估用户ID=100002的信用风险"
    },
    {
      "role": "assistant",
      "content": "我需要先查询该用户的信用数据，了解其基本财务状况。\n<tool_call>\n{\"name\": \"query_user_credit_data\", \"arguments\": {\"user_id\": 100002}}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "<tool_response>\n{\"income\": 270000, \"credit_amount\": 1293502, \"annuity\": 35698, \"days_employed\": 3650, \"ext_source_mean\": 0.52}\n</tool_response>"
    },
    {
      "role": "assistant",
      "content": "用户信贷金额较高，信贷收入比约4.79，需要量化评估违约风险。\n<tool_call>\n{\"name\": \"predict_risk_score\", \"arguments\": {\"features\": {\"income\": 270000, \"credit_amount\": 1293502, \"annuity\": 35698, \"days_employed\": 3650, \"ext_source_mean\": 0.52}}}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "<tool_response>\n{\"risk_score\": 0.68, \"risk_level\": \"中高风险\", \"top_factors\": [\"信贷收入比过高\", \"外部评分偏低\"]}\n</tool_response>"
    },
    {
      "role": "assistant",
      "content": "风险评分0.68属于中高风险，需要查询相关风控政策来给出合规建议。\n<tool_call>\n{\"name\": \"search_knowledge_base\", \"arguments\": {\"query\": \"中高风险客户信贷审批标准\"}}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "<tool_response>\n根据《商业银行信用风险管理指引》第23条，对于风险评分在0.6-0.8区间的中高风险客户，应当：1）要求追加担保或抵押；2）降低授信额度至原申请的60%-80%；3）缩短贷款期限；4）提高贷款利率以覆盖风险溢价。\n</tool_response>"
    },
    {
      "role": "assistant",
      "content": "## 信用风险评估报告\n\n**用户ID:** 100002\n**综合风险等级: 中高风险 (评分 0.68)**\n\n### 一、关键指标分析\n- 年收入: 27万元\n- 信贷金额: 129.4万元\n- 信贷收入比: 4.79 (警戒阈值: 3.0) ⚠️\n- 年还款额: 3.57万元，占收入13.2%\n- 外部评分均值: 0.52 (偏低)\n\n### 二、主要风险因素\n1. **信贷收入比过高 (4.79)**: 远超3.0的健康阈值，表明借款人承担了与收入不匹配的债务负担\n2. **外部评分偏低 (0.52)**: 第三方征信评分低于平均水平，历史信用表现存在隐患\n\n### 三、政策依据与处置建议\n根据《商业银行信用风险管理指引》第23条：\n1. **要求追加担保**: 建议要求提供房产或第三方担保\n2. **降低授信额度**: 建议将授信额度调整至77.6-103.5万元（原申请的60%-80%）\n3. **缩短贷款期限**: 建议贷款期限不超过3年\n4. **上浮利率**: 建议在基准利率基础上上浮15%-30%以覆盖风险溢价\n\n### 四、审批建议\n**有条件通过** — 在落实上述风险缓释措施后可予以审批。"
    }
  ]
}
```

---

## 六、与 OpenAI 格式的关键差异（易错点）

| 对比项 | Qwen2.5 原生格式 | OpenAI 格式 |
|--------|------------------|-------------|
| 工具调用位置 | `content` 内的 `<tool_call>` 标签 | 独立的 `tool_calls` 字段 |
| arguments 类型 | **JSON 对象** `{"user_id": 100002}` | **JSON 字符串** `"{\"user_id\": 100002}"` |
| 工具结果 role | **`"user"`** + `<tool_response>` 标签 | **`"tool"`** |
| 工具定义位置 | system prompt 内 `<tools>` 标签 | 独立的 `tools` 参数 |
| Thought 位置 | content 开头自由文本 | 无原生支持 |

> **警告**: 微调数据中混用 OpenAI 格式会导致模型输出格式混乱，tool call 遵循率从 ~90% 降至 ~50%。

---

## 七、微调数据校验清单

每条训练数据生成后，必须通过以下校验：

- [ ] system prompt 包含 `<tools></tools>` 且内部每行是合法 JSON
- [ ] 每个 `<tool_call>` 内的 JSON 可被 `json.loads()` 解析
- [ ] `arguments` 值是 dict 对象，不是字符串
- [ ] `name` 字段值与 `<tools>` 中定义的函数名完全匹配
- [ ] 参数名和类型与 `<tools>` 中的 schema 一致
- [ ] 工具结果消息的 role 是 `"user"`，content 以 `<tool_response>` 开头
- [ ] 最后一条 assistant 消息不包含 `<tool_call>` 标签（表示最终回答）
- [ ] 整个 messages 数组的 role 交替合理（不出现连续两个相同 role）

---

## 参考来源

- [Qwen2.5 Official Chat Template (Ollama)](https://ollama.com/library/qwen2.5/blobs/eb4402837c78)
- [Qwen Function Call Documentation](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Qwen2.5-14B-Instruct Model Card](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- [Qwen2.5 GitHub Repository](https://github.com/QwenLM/Qwen2.5)
