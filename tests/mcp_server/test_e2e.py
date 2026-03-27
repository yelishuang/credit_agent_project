"""
MCP 端到端测试 — 通过 stdio 协议连接 MCP Server，模拟 Agent 真实调用流程
测试组：
  1. 多个 case_id 的 query_user_credit_data
  2. 不同特征输入的 predict_risk_score
  3. 多种 query 的 search_knowledge_base
  4. 完整流水线：查数据 -> 预测 -> 检索
  5. 边界/异常情况
"""

import asyncio
import json
import sys
import os
import time

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

# ── 配置 ──────────────────────────────────────────────────────
SERVER_CMD = sys.executable  # 当前 python 解释器
SERVER_ARGS = ["-m", "src.mcp_server.server"]
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SFT_FIELDS = [
    "mainoccupationinc_384A", "credamount_770A", "annuity_780A",
    "totaldebt_9A", "currdebt_22A", "maxdpdlast24m_143P",
    "maxdpdlast3m_392P", "numactivecreds_622L", "applications30d_658L",
    "numrejects9m_859L", "debtoverdue_47A", "days30_165L",
    "numinstlswithdpd5_4187116L",
]

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{tag}] {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")
async def run_tests():
    server_params = StdioServerParameters(
        command=SERVER_CMD,
        args=SERVER_ARGS,
        cwd=PROJECT_DIR,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 验证工具列表
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print("=" * 70)
            print(f"已注册工具: {tool_names}")
            print("=" * 70)

            # ── 测试组 1: query_user_credit_data (多 case_id) ──
            print("\n[测试组 1] query_user_credit_data — 多客户查询")
            print("-" * 50)
            test_ids = [57543, 57549, 57569, 57632, 57634]
            queried_features = {}
            for uid in test_ids:
                t0 = time.time()
                res = await session.call_tool("query_user_credit_data", {"user_id": uid})
                elapsed = time.time() - t0
                text = res.content[0].text
                data = json.loads(text)
                has_all = all(f in data for f in SFT_FIELDS)
                no_error = "error" not in data
                ok = has_all and no_error
                report(
                    f"case_id={uid} ({elapsed:.2f}s)",
                    ok,
                    f"字段数={len(data)}, 收入={data.get('mainoccupationinc_384A')}, "
                    f"贷款={data.get('credamount_770A')}, 逾期={data.get('maxdpdlast24m_143P')}",
                )
                if ok:
                    queried_features[uid] = data

            # ── 测试组 2: predict_risk_score (多组特征) ──
            print("\n[测试组 2] predict_risk_score — 多组特征预测")
            print("-" * 50)
            for uid, feat in queried_features.items():
                t0 = time.time()
                res = await session.call_tool("predict_risk_score", {"features": json.dumps(feat)})
                elapsed = time.time() - t0
                text = res.content[0].text
                pred = json.loads(text)
                ok = (
                    "risk_score" in pred
                    and "risk_level" in pred
                    and "top_factors" in pred
                    and 0 <= pred["risk_score"] <= 1
                    and pred["risk_level"] in ("低风险", "中风险", "高风险")
                )
                report(
                    f"case_id={uid} ({elapsed:.2f}s)",
                    ok,
                    f"score={pred.get('risk_score')}, level={pred.get('risk_level')}, "
                    f"top1={pred.get('top_factors', [''])[0][:40]}",
                )

            # ── 测试组 3: search_knowledge_base (多种 query) ──
            print("\n[测试组 3] search_knowledge_base — 多种检索查询")
            print("-" * 50)
            queries = [
                "逾期风险评估标准",
                "债务收入比如何计算",
                "信用评分模型的主要特征",
                "贷款审批流程",
            ]
            for q in queries:
                t0 = time.time()
                res = await session.call_tool("search_knowledge_base", {"query": q})
                elapsed = time.time() - t0
                text = res.content[0].text
                segments = text.split("---")
                ok = len(text) > 50 and len(segments) >= 2
                report(
                    f"query=\"{q}\" ({elapsed:.2f}s)",
                    ok,
                    f"返回长度={len(text)}, 片段数={len(segments)}, 前80字={text[:80].replace(chr(10), ' ')}",
                )

            # ── 测试组 4: 完整流水线 (查数据 -> 预测 -> 检索) ──
            print("\n[测试组 4] 完整流水线 — 模拟 Agent 审批流程")
            print("-" * 50)
            pipeline_id = 57551
            t0 = time.time()

            # Step 1: 查数据
            r1 = await session.call_tool("query_user_credit_data", {"user_id": pipeline_id})
            feat = json.loads(r1.content[0].text)
            step1_ok = all(f in feat for f in SFT_FIELDS)
            report("Step1 查询客户数据", step1_ok, f"收入={feat.get('mainoccupationinc_384A')}")

            # Step 2: 预测
            r2 = await session.call_tool("predict_risk_score", {"features": json.dumps(feat)})
            pred = json.loads(r2.content[0].text)
            step2_ok = "risk_score" in pred and "risk_level" in pred
            report("Step2 风险预测", step2_ok, f"score={pred.get('risk_score')}, level={pred.get('risk_level')}")

            # Step 3: 根据风险等级检索知识
            level = pred.get("risk_level", "")
            r3 = await session.call_tool("search_knowledge_base", {"query": f"{level}客户的审批策略"})
            kb_text = r3.content[0].text
            step3_ok = len(kb_text) > 50
            report("Step3 知识库检索", step3_ok, f"返回长度={len(kb_text)}")

            total_time = time.time() - t0
            report(f"流水线总耗时", True, f"{total_time:.2f}s")

            # ── 测试组 5: 边界/异常情况 ──
            print("\n[测试组 5] 边界与异常情况")
            print("-" * 50)

            # 不存在的 case_id
            r = await session.call_tool("query_user_credit_data", {"user_id": 99999999})
            text = r.content[0].text
            data = json.loads(text)
            report("不存在的 case_id", "error" in data, f"返回: {text[:80]}")

            # 空 query 检索
            r = await session.call_tool("search_knowledge_base", {"query": ""})
            text = r.content[0].text
            report("空 query 检索", len(text) > 0, f"返回长度={len(text)}")

            # 手工构造特征预测
            manual_feat = {f: 0 for f in SFT_FIELDS}
            manual_feat["credamount_770A"] = 500000
            manual_feat["mainoccupationinc_384A"] = 10000
            r = await session.call_tool("predict_risk_score", {"features": json.dumps(manual_feat)})
            pred = json.loads(r.content[0].text)
            report(
                "手工高风险特征",
                "risk_score" in pred,
                f"score={pred.get('risk_score')}, level={pred.get('risk_level')}",
            )

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print(f"测试完成: {passed} passed, {failed} failed, 共 {passed + failed} 项")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(run_tests())
    sys.exit(0 if ok else 1)
