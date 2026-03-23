# 代码审查修复指南

## 背景

本文档记录 CreditAgent 项目全面代码审查的进度，供下一个对话继续修复。

审查 prompt 在 `temp/prompt_F_code_review.md`。

## 已完成的修复

### C1. predict_risk_score 类型签名不匹配（已修复）
- `src/mcp_server/server.py:88` — `features: str` 改为 `features: dict`，删除 `isinstance` 兜底分支
- 现在 server、orchestrator system prompt、SFT 训练数据三方统一为 dict 类型

### C2. 评测与生产使用不同 checkpoint + 目录结构整理（已修复）
- 将 `outputs/qwen2.5_14b_lora/` 下的文件移入 `20260323_150000/` 子目录，符合 `train_lora.py` 的 run_name 规范
- `llm_judge.py:43` 默认值改为 `None`，run 模式不带 `--lora-adapter` 时报错提示
- `orchestrator.py:19` 路径更新为 `20260323_150000/`（顶层最佳模型）
- `discover_adapters` 改为只扫描时间戳子目录的顶层 adapter（每次训练的最佳模型），不再递归进 checkpoint 子目录
- `train_lora.py` 的 `save_total_limit` 从 3 改为 2
- 删除了 `checkpoint-300` 和 `checkpoint-450`，只保留 `checkpoint-492`

### C3. FIELD_MAPPING 导致特征名不匹配（已修复）
- `src/mcp_server/server.py` 的 `predict_risk_score` 函数中，调用 `predict_credit_risk` 前将 SFT 字段名映射回模型实际列名
- 修复前：`mainoccupationinc_384A` 和 `debtoverdue_47A` 两个关键特征在预测时永远是 NaN
- 此 bug 只影响推理路径，不影响已完成的训练

### C4. 评测 agent loop 只处理第一个 tool_call（已修复）
- `llm_judge.py` 的评测循环改为遍历所有 tool_calls，逐个生成 mock 响应，与 orchestrator 逻辑一致

### C5. generate_risk_report 幽灵工具（已修复）
- 从 `src/sft_data_gen/generate_sft_data.py` 的 `VALID_TOOL_NAMES` 和 `TOOL_SCHEMAS` 中删除了 `generate_risk_report`

### W3. get_eval_paths 类型注解错误（已修复）
- `tuple[Path, Path, Path]` 改为 `tuple[Path, Path, Path, Path]`

### W4. parse_tool_calls 静默吞掉 JSON 解析错误（已修复）
- `src/agent/orchestrator.py:46-47` 的 `except json.JSONDecodeError: pass` 改为 `logger.warning` 输出原文

### C6. 缺少 .gitignore（暂缓）
- 等项目准备上 GitHub 时再创建

## 待修复问题

详细描述见 `docs/code_review/pending_issues.md`。

### W1. search_knowledge_base 的 top_k 参数被静默丢弃（待定）
- 文件：`src/mcp_server/server.py:113`、`src/agent/orchestrator.py:30`
- 二选一：给函数加 `top_k` 参数，或从 schema 里删掉 `top_k`

### W6. predict.py np.mean 对多样本场景有隐患（待修复）
- 文件：`src/credit_risk_model/predict.py:68`
- `float(np.mean(y_preds))` 对单样本碰巧正确，多样本时会坍缩为单个标量
- 修复：`float(np.mean(y_preds, axis=0)[0])`

### W7. predict.py 用位置索引应改为列名索引（待修复）
- 文件：`src/credit_risk_model/predict.py:109`，`predict_batch` 约 190 行同样问题
- `sample_row.iloc[i]` 改为 `sample_row[cols[i]]`

### W8. 评测与生产的 parse_tool_calls 行为不一致（待修复）
- orchestrator 静默丢弃解析失败的 tool_call（已加 warning 但仍然丢弃）
- evaluator 记录为 `__PARSE_ERROR__` 并计入评分扣分
- 评测会系统性地多扣分，需要统一行为

### W9. generate_all_responses 依赖全局变量而非参数传递（待修复）
- 文件：`src/evaluation/llm_judge.py` 约 303 行
- 函数直接引用模块级 `LORA_ADAPTER_PATH`，应改为参数传递

### W10. 缺少多个 __init__.py（待修复）
- 缺失：`src/`、`src/evaluation/`、`src/rag/`、`src/training/`、`src/sft_data_gen/`（顶层）
- 当前通过 sys.path hack 能跑，但包结构不完整

### W11. tests/ 目录不在标准位置（记录）
- MCP 测试文件在 `src/mcp_server/test_tools.py` 和 `test_e2e.py`

### W12. requirements.txt 检查（待确认）
- 需确认是否缺少依赖（如 jsonlines）

### W13. 环境变量未文档化（待修复）
- `ANTHROPIC_API_KEY`、`HF_HUB_OFFLINE` 等未在 README 或 .env.example 中列出

### W14. BAAI/bge-m3 模型名硬编码无覆盖机制（记录）
- 文件：`src/mcp_server/server.py:107`

### W15. Agent 循环耗尽 MAX_TURNS 时的返回逻辑有 bug（待修复）
- 文件：`src/agent/orchestrator.py:133`
- 循环最后一步总是追加 `role: "user"` 的 tool_response，所以永远走 fallback
- 应在追加 tool_response 前保存最后一次 assistant 回复

### Info 级别
- I1. 延迟导入导致启动时无法发现依赖问题（server.py）
- I2. feature_engineering.py:46 — MONTH 列可能不存在导致 ColumnNotFoundError
- I3. feature_config.py:26 — score == 1.0 时落入 fallback，碰巧正确但脆弱
- I4. rag_retrieval_eval.py:127 — `'raw_text' in dir()` 应为 `'raw_text' in locals()`

## W5. SFT few-shot 示例缺少 2 个字段（Info，已验证无影响）
- 经检查训练数据，Claude 自行补全了字段，1682 条中仅 2 条缺失，不影响训练质量
- 下次重新生成时建议补全 few-shot 示例
