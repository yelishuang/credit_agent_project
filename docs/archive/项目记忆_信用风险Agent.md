# 信用风险评估 AI Agent 项目 — 关键记忆

## 项目定位
- **项目名**：CreditAgent — 基于大模型微调的消费贷款智能审批系统
- **目标**：放简历，面数据挖掘岗位
- **时间**：7-10天完成

## 业务场景
- **公司背景**：Home Credit，国际消费金融公司，专门为信用记录薄弱/无的人群提供消费贷款
- **场景**：个人消费贷款审批（不是信用卡，不是银行）
- **核心决策**：批不批 + 额度 + 利率
- **客群特点**：征信白户是主流客群，普惠金融场景
- **但是**：风控规则手册保持银行风格即可，两者差别不大

## 技术架构
- **ML模型**：基于Kaggle "Home Credit - Credit Risk Model Stability" 开源方案，已跑通 ✅
  - 模型只输出一个浮点数：违约概率（如0.68）
  - 不输出风险等级、关键因子等
- **LLM微调**：Qwen2.5-7B-Instruct，LoRA SFT
  - 用Qwen原生tool calling格式 + ReAct思考链（兼顾稳定性和可解释性）
  - 训练数据500-600条，4类：风险分析40%、工具调用30%、金融知识20%、拒绝边界10%
- **RAG**：FAISS + BGE-large-zh-v1.5
- **MCP**：Local(stdio) 2个工具 + Remote(SSE/Streamable HTTP) 2个工具
- **评测**：LLM-as-a-Judge，Claude API，微调前后对比
- **展示**：命令行优先

## Agent工具调用链路
```
① 查数据库（MCP-Local）→ 拿到客户原始特征（收入、负债、逾期等）
② 调ML模型（MCP-Local）→ 只拿到违约概率 0.68
③ 搜RAG知识库（MCP-Remote）→ 找相关风控政策
④ Agent自己结合①②③ → 分析为什么高风险 + 给出建议
⑤ 输出完整审批报告
```

## 设备
- A6000 48GB，每晚可用8小时训练
- 服务器完全自由（装包、开端口都可以）
- 有Claude/OpenAI API key，API预算¥100-200

## Kaggle数据位置
- 数据目录：`E:\home-credit-credit-risk-model-stability\`
- 特征定义：`feature_definitions.csv`
- 训练数据：`csv_files/train/` 下有 train_base.csv, train_applprev, train_credit_bureau 等表
- target在train_base.csv中，case_id为主键

## 已完成的RAG文档（位于 C:\Users\ye\Desktop\RAG文档\）
1. ✅ 征信业管理条例_全文.txt — 核心法规
2. ✅ 巴塞尔协议III_信用风险_中文摘要.md — 中文摘要
3. ✅ 银行零售信贷审批风控规则手册.md — 模拟审批规则（核心文档）
4. ✅ 个人征信基础知识手册.md — 征信科普
5. ✅ 常见问题解答.pdf — 央行征信中心官方FAQ
6. ✅ bcbs189.pdf — 巴塞尔III英文原版（备用）
7. ⚪ 商业银行市场风险管理办法.pdf — 是"市场风险"非"信用风险"，相关性一般
8. ❌ 征信业管理条例.pdf — 空白PDF，已被txt替代

## 还需要补充的RAG文档
1. Home Credit比赛说明（业务背景、数据表结构、评估指标、stability概念）
2. 消费金融相关法规（消费金融公司管理办法、互联网贷款管理办法等）
3. 信用评分专业知识（PD模型原理、KS/AUC/PSI评估指标、特征工程经验）
4. 场景QA文档（30-50个典型问答对，提高RAG检索命中率）

## 时间线
- Day 0（今晚）：✅ Kaggle模型跑通、✅ 已有RAG文档生成、⬜ 补充RAG文档、⬜ 微调数据Prompt模板设计、⬜ 生成类型3+4训练数据150条
- Day 1-2：构造全部微调数据
- Day 2晚：LoRA微调训练
- Day 3：验证模型 + 完成RAG
- Day 4-5：MCP工具
- Day 5-6：Agent编排
- Day 7：评测
- Day 8：整理代码
