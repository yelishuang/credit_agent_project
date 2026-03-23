# 代码审查待修复问题

## W1. search_knowledge_base 的 top_k 参数被静默丢弃

**严重程度：** Warning

**问题描述：**

system prompt（`src/agent/orchestrator.py:30`）告诉模型 `search_knowledge_base` 有两个参数：
```json
{"query": {"type": "string"}, "top_k": {"type": "integer", "description": "返回条数，默认3"}}
```

模型学会了这个 schema，推理时可能输出：
```json
{"name": "search_knowledge_base", "arguments": {"query": "逾期处理政策", "top_k": 5}}
```

然后 `src/agent/tool_executor.py:39-40` 在调用函数前会过滤参数——只保留函数签名里存在的参数：
```python
sig = inspect.signature(fn)
valid_args = {k: v for k, v in arguments.items() if k in sig.parameters}
```

但 `src/mcp_server/server.py:113` 的函数签名只有 `query`：
```python
def search_knowledge_base(query: str) -> str:
```

所以 `top_k` 被过滤掉了，函数内部硬编码 `top_k=3`。模型传 `top_k=5` 想多拿几条结果，实际永远只返回 3 条，模型不知道自己的请求被忽略了。

**影响程度：** 不大。当前 SFT 训练数据里 few-shot 示例用的也是 `top_k=3`，模型大概率不会传别的值。但 schema 声明了这个参数却不生效，属于接口承诺和实现不一致。

**修复方案（二选一）：**
1. 给 `search_knowledge_base` 函数加上 `top_k: int = 3` 参数，让它真正生效
2. 从 system prompt 的 schema 里删掉 `top_k`，不再暴露这个参数

---

## W5. SFT few-shot 示例缺少 2 个字段（days30_165L、numinstlswithdpd5_4187116L）

**严重程度：** Info

**问题描述：**

`src/sft_data_gen/prompts/` 下多个 prompt 模板的 few-shot 示例中，`query_user_credit_data` 的 tool_response 只包含 11 个字段，缺少 `days30_165L` 和 `numinstlswithdpd5_4187116L`。而实际 MCP server 返回 13 个字段。

**实际影响：** 经检查已生成的训练数据，Claude 在生成时自行补全了这两个字段。1682 条含 `query_user_credit_data` 响应的样本中，仅 2 条缺失（type_d_explanation），占比 < 0.1%。训练数据质量未受实质影响。

**修复方案：** 在 few-shot 示例中补全 13 个字段，确保下次重新生成训练数据时不依赖 Claude 的自行补全。
