# SFT数据遴选与格式化日志

## 时间：2026-03-23

## 做了什么

### 1. 数据格式校验（scripts/curate_sft_data.py）

对 outputs/sft_data/ 下 5 个 JSONL 文件（2377 条）执行四项校验：
- JSON 合法性
- 角色交替正确性（system 开头 → user/assistant 交替 → tool_response 跟在 tool_call 后）
- tool_call 合法性（函数名、参数完整性）
- 最终 assistant 回复完整性（非空、非截断）

结果：2377 条全部通过格式校验，发现 157 条 hash 重复，去重后 2220 条。

### 2. Tokenizer 兼容性验证

确认全部 2220 条样本可被 Qwen2.5-14B-Instruct tokenizer 正确消费：
- apply_chat_template 无报错，输出 ChatML 格式（`<|im_start|>/<|im_end|>`）
- 无 UNK token
- token 长度分布：min=486, max=2248, mean=1410

### 3. 数据遴选（scripts/select_sft_data.py）

三个维度筛选：

**逻辑一致性检查**：从 tool_response 提取 risk_score，从最终回复提取审批建议，检查矛盾。剔除 1 条（risk_score=0.74 却建议无条件通过）。

**Token 长度过滤**：以训练 max_length=2048 为上限，剔除 164 条超长样本。

**内容多样性采样**：使用 BGE-M3 embedding + MMR（最大边际相关性）贪心采样，按目标数量选取：
- type_a（审批）：795 → 600
- type_b（查询）：360 → 225
- type_c（知识）：358 → 300
- type_d（解释）：360 → 225
- type_e（拒绝）：182 → 150（三个子类各 50：tamper/illegal/out_of_scope）

### 4. Train/Val 语义泄漏检查与修复

初次随机划分后发现严重泄漏：
- 29 条 val 与 train 问题完全相同（sim=1.0）
- 86% val 样本与 train 相似度 ≥ 0.9

原因：hash 去重基于完整 messages（含不同 tool_response），同一问题配不同用户数据被视为不同样本。

修复方案：用 BGE-M3 embedding 对全部 1500 条样本做 Union-Find 语义聚类（阈值 0.85），同 cluster 样本只能全部进 train 或全部进 val。

修复后：
- ≥ 0.85 相似度：0 条（修复前 147 条）
- 最大相似度：0.848（修复前 1.0）
- 最终 train=1308, val=192

## 产出文件

```
outputs/sft_data_curated/          # 格式校验+去重后（中间产物）
├── train.jsonl (1998)
└── val.jsonl (222)

outputs/sft_data_selected/         # 最终遴选结果（用这个训练）
├── train.jsonl (1308)
└── val.jsonl (192)

scripts/curate_sft_data.py         # 格式校验脚本
scripts/select_sft_data.py         # 遴选脚本
```

## 已知局限

- 数据规模 1500 条属于 LoRA 微调可用下限，泛化能力可能不足，后续可按需扩充
- 未设计独立评估基准（test set + 评估指标），需在训练前补充
- system prompt 单一，模型可能过拟合特定 prompt 措辞
- 逻辑一致性仅检查极端矛盾（≥0.7通过/≤0.1拒绝），中间地带语义矛盾未覆盖
