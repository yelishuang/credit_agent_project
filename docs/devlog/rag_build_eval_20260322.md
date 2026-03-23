# RAG 向量检索知识库构建 — 2026-03-22

## 完成事项

### 1. RAG 文档切分与索引构建
- 编写 `RAG/build_rag_index.py`，实现 Markdown-aware 切分（按 `##` 标题边界切块，超长 section 按段落二次切分，每个 chunk 保留标题前缀）
- 4 篇文档切分为 134 个 chunk，平均 341 字，分布均匀
- 使用 `BAAI/bge-m3` 模型（通过 SentenceTransformer）编码为 1024 维向量，存入 FAISS IndexFlatIP
- 输出：`rag_index/credit_risk.index` + `rag_index/chunks.json`
- 检索函数 `load_retriever()` 封装完毕，供 MCP Remote Server 直接导入使用

### 2. 自动化检索质量评测体系
- 编写 `spec/rag_retrieval_eval.py`，实现完整评测流程：
  - 用 Claude API 为每个 chunk 自动生成 3 条检索 query（精确/口语化/换说法）
  - 反向检索 FAISS 索引，计算 Hit@1、Hit@3、MRR 三个标准 IR 指标
  - eval_pairs 缓存机制：生成的 query 保存为 JSON，后续调参重跑不花 API 费用
- 评测结果（基线，最终采用）：
  - Hit@1: 68.91% [PASS]
  - Hit@3: 85.57% [PASS]
  - MRR: 76.12% [PASS]

### 3. 尝试过但未采用的优化方向
| 方案 | 结果 | 原因 |
|------|------|------|
| chunk 加文档来源前缀 `[文档名]` | 全指标下降 | 前缀稀释了正文语义信号 |
| 去掉 BGE query 前缀 | 下降 | bge-m3 配合该前缀效果更好 |
| FlagEmbedding dense+sparse 混合检索 | 未能正确运行 | FlagEmbedding 与 transformers 版本不兼容，sparse_linear 权重未随 AutoModel 加载 |
| 手动实现 sparse（L2 norm 近似） | 下降 | 近似质量不足 |

### 4. 失败 case 分析结论
- 58/402 条 query 未命中（14.4%），86% 的失败来自口语化/换说法 query
- 失败 case 的 top-1 分数平均 0.634（vs 命中 0.678），属于"近似命中错误 chunk"而非"完全找不到"
- 根因：文档内多个 chunk 涉及相似概念（DPD、DTI、风险等级），口语化 query 丢失术语锚点后区分度不足

## 关键文件
- `RAG/build_rag_index.py` — 索引构建 + 检索函数
- `spec/rag_retrieval_eval.py` — 自动化评测脚本
- `rag_index/` — 索引产物（需在服务器上生成）
- `rag_index/eval_report.json` — 详细评测结果
- `rag_index/eval_report_pairs.json` — eval query 缓存（复用免 API 费）

## 环境备注
- 服务器：A6000，conda 环境 `credit_agent`，Python 3.10
- HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- 依赖：`sentence-transformers`, `faiss-gpu`（或 `faiss-cpu`），`anthropic`
- torch 已升级至 2.10+（解决 CVE-2025-32434 安全限制）
