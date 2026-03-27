"""
微调数据生成主脚本

用法:
    python generate_sft_data.py --type d --count 360                    # 单类型
    python generate_sft_data.py --type all                              # 全部类型（使用预设数量）
    python generate_sft_data.py --type a --count 5 --concurrency 3     # 小批量测试
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic

# 确保 prompts 包可导入
sys.path.insert(0, str(Path(__file__).parent))
from prompts import type_a_approval
from prompts import type_b_query
from prompts import type_c_knowledge
from prompts import type_d_explanation
from prompts import type_e_rejection

# ── 类型配置 ─────────────────────────────────────────────────────────
TYPE_CONFIG = {
    "a": {"build_fn": type_a_approval.build_prompt, "file": "type_a_approval.jsonl", "default_count": 960},
    "b": {"build_fn": type_b_query.build_prompt, "file": "type_b_query.jsonl", "default_count": 360},
    "c": {"build_fn": type_c_knowledge.build_prompt, "file": "type_c_knowledge.jsonl", "default_count": 480},
    "d": {"build_fn": type_d_explanation.build_prompt, "file": "type_d_explanation.jsonl", "default_count": 360},
    "e": {"build_fn": type_e_rejection.build_prompt, "file": "type_e_rejection.jsonl", "default_count": 240},
}

# ── 工具名白名单 ─────────────────────────────────────────────────────
VALID_TOOL_NAMES = {
    "query_user_credit_data",
    "predict_risk_score",
    "search_knowledge_base",
}

# ── 工具参数 schema（用于校验）────────────────────────────────────────
TOOL_SCHEMAS = {
    "query_user_credit_data": {"required": ["user_id"], "types": {"user_id": int}},
    "predict_risk_score": {"required": ["features"], "types": {"features": dict}},
    "search_knowledge_base": {"required": ["query"], "types": {"query": str}},
}


# ── JSON 提取 ────────────────────────────────────────────────────────
def parse_json_from_response(text: str) -> dict | None:
    """从 Claude 响应中提取 JSON 对象。处理可能的 markdown 代码块包裹。"""
    text = text.strip()
    # 去掉 markdown 代码块
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "messages" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # 尝试找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ── 格式校验 ─────────────────────────────────────────────────────────
def validate_sample(sample: dict, data_type: str = "D") -> tuple[bool, str]:
    """
    校验一条微调数据的格式合规性。
    返回 (is_valid, error_message)。
    """
    msgs = sample.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return False, "messages 不是列表或长度不足"

    # 1. system prompt 校验
    if msgs[0].get("role") != "system":
        return False, "首条消息 role 不是 system"
    sys_content = msgs[0].get("content", "")
    if "<tools>" not in sys_content or "</tools>" not in sys_content:
        return False, "system prompt 缺少 <tools> 标签"
    # 校验 tools 内每行 JSON
    tools_match = re.search(r"<tools>\n?(.*?)\n?</tools>", sys_content, re.DOTALL)
    if tools_match:
        for line in tools_match.group(1).strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                return False, f"tools 中非法 JSON: {line[:80]}"

    # 2. role 交替校验
    for i in range(1, len(msgs)):
        if msgs[i]["role"] == msgs[i - 1]["role"]:
            # system 后面跟 user 是正常的，但 user-user 或 assistant-assistant 不行
            if not (i == 1 and msgs[0]["role"] == "system"):
                return False, f"消息 {i} role 连续重复: {msgs[i]['role']}"

    # 3. tool_call 校验
    tool_call_count = 0
    for i, msg in enumerate(msgs):
        content = msg.get("content", "")
        if msg["role"] == "assistant" and "<tool_call>" in content:
            tool_call_count += 1
            # 提取所有 tool_call
            calls = re.findall(r"<tool_call>\n?(.*?)\n?</tool_call>", content, re.DOTALL)
            for call_str in calls:
                try:
                    call = json.loads(call_str.strip())
                except json.JSONDecodeError:
                    return False, f"tool_call JSON 解析失败: {call_str[:80]}"
                if call.get("name") not in VALID_TOOL_NAMES:
                    return False, f"未知工具名: {call.get('name')}"
                if not isinstance(call.get("arguments"), dict):
                    return False, f"arguments 不是 dict: {type(call.get('arguments'))}"
                # 参数校验
                schema = TOOL_SCHEMAS.get(call["name"])
                if schema:
                    for req in schema["required"]:
                        if req not in call["arguments"]:
                            return False, f"缺少必填参数 {req} in {call['name']}"

    # 4. tool_response 校验
    for msg in msgs:
        if msg["role"] == "user" and "<tool_response>" in msg.get("content", ""):
            if not msg["content"].strip().startswith("<tool_response>"):
                return False, "tool_response 消息 content 未以 <tool_response> 开头"

    # 5. 最后一条 assistant 不含 tool_call
    last_assistant = None
    for msg in reversed(msgs):
        if msg["role"] == "assistant":
            last_assistant = msg
            break
    if last_assistant and "<tool_call>" in last_assistant.get("content", ""):
        return False, "最后一条 assistant 消息仍包含 <tool_call>"

    # 6. 类型特有校验
    if data_type == "D":
        if tool_call_count < 2:
            return False, f"类型D至少需要2轮tool call，实际{tool_call_count}轮"
        if last_assistant and len(last_assistant.get("content", "")) < 200:
            return False, "最终解释回答过短（<200字符）"
    elif data_type == "A":
        if tool_call_count < 3:
            return False, f"类型A至少需要3轮tool call，实际{tool_call_count}轮"
        if last_assistant and len(last_assistant.get("content", "")) < 300:
            return False, "最终审批报告过短（<300字符）"
    elif data_type == "B":
        if tool_call_count != 1:
            return False, f"类型B应恰好1轮tool call，实际{tool_call_count}轮"
    elif data_type == "C":
        if tool_call_count != 1:
            return False, f"类型C应恰好1轮tool call，实际{tool_call_count}轮"
    elif data_type == "E":
        if tool_call_count != 0:
            return False, f"类型E不应有tool call，实际{tool_call_count}轮"

    return True, "OK"


# ── 单条生成 ─────────────────────────────────────────────────────────
async def generate_one(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    model: str,
    data_type: str = "D",
    max_retries: int = 2,
) -> dict | None:
    """生成一条数据，失败最多重试 max_retries 次。"""
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            sample = parse_json_from_response(text)
            if sample is None:
                last_err = "JSON解析失败"
                continue
            valid, err = validate_sample(sample, data_type)
            if valid:
                return sample
            last_err = err
        except anthropic.APIError as e:
            last_err = f"API错误: {e}"
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
            continue
    return last_err


# ── 批量生成 ─────────────────────────────────────────────────────────
async def generate_batch(
    data_type: str,
    count: int,
    model: str,
    output_dir: str,
    concurrency: int = 10,
):
    """批量生成微调数据。"""
    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN", os.environ.get("ANTHROPIC_API_KEY")),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )
    semaphore = asyncio.Semaphore(concurrency)

    # 选择 prompt 构建器
    config = TYPE_CONFIG.get(data_type)
    if not config:
        print(f"未知类型: {data_type}")
        return
    build_fn = config["build_fn"]
    output_file = Path(output_dir) / config["file"]
    validate_type = data_type.upper()

    os.makedirs(output_dir, exist_ok=True)

    results = []
    failed = 0
    error_reasons = {}
    start_time = time.time()

    async def _task(idx: int):
        nonlocal failed
        async with semaphore:
            prompt, meta = build_fn()
            result = await generate_one(client, prompt, model, validate_type)
            if isinstance(result, dict):
                result["_metadata"] = meta
                return result
            else:
                failed += 1
                # result 是错误原因字符串
                reason = result if isinstance(result, str) else "未知错误"
                error_reasons[reason] = error_reasons.get(reason, 0) + 1
                return None

    # 多生成 15% 余量应对失败
    total_attempts = int(count * 1.15)
    print(f"开始生成类型 {data_type.upper()} 数据")
    print(f"  目标: {count} 条, 计划发起: {total_attempts} 次, 并发: {concurrency}")
    print(f"  模型: {model}")
    print(f"  输出: {output_file}")
    print()

    tasks = [_task(i) for i in range(total_attempts)]
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        if result:
            results.append(result)
        if completed % 10 == 0 or completed == total_attempts:
            elapsed = time.time() - start_time
            print(f"  进度: {completed}/{total_attempts} | 成功: {len(results)} | "
                  f"失败: {failed} | 耗时: {elapsed:.1f}s")
        if len(results) >= count:
            break

    results = results[:count]

    # 写入 JSONL（追加模式）
    with open(output_file, "a", encoding="utf-8") as f:
        for sample in results:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"\n生成完成:")
    print(f"  总条数: {len(results)}")
    print(f"  通过率: {len(results)/(completed)*100:.1f}%")
    print(f"  总耗时: {elapsed:.1f}s")

    if results:
        profile_dist = {}
        variant_dist = {}
        for r in results:
            meta = r.get("_metadata", {})
            p = meta.get("risk_profile", "unknown")
            v = meta.get("variant", "unknown")
            profile_dist[p] = profile_dist.get(p, 0) + 1
            variant_dist[v] = variant_dist.get(v, 0) + 1
        print(f"  风险分布: {profile_dist}")
        print(f"  变体分布: {variant_dist}")

    if error_reasons:
        print(f"  失败原因分布:")
        for reason, cnt in sorted(error_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {cnt}次")

    print(f"\n输出文件: {output_file}")


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="微调数据生成器")
    parser.add_argument("--type", required=True, choices=["a", "b", "c", "d", "e", "all"],
                        help="数据类型 (a-e 或 all)")
    parser.add_argument("--count", type=int, default=None,
                        help="目标生成条数 (默认使用各类型预设值)")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Claude 模型 ID")
    parser.add_argument("--output_dir", default=str(Path(__file__).parent.parent.parent / "outputs" / "sft_data" / "raw"),
                        help="输出目录")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="并发数 (默认 10)")
    args = parser.parse_args()

    types_to_run = list(TYPE_CONFIG.keys()) if args.type == "all" else [args.type]

    for t in types_to_run:
        count = args.count or TYPE_CONFIG[t]["default_count"]
        asyncio.run(generate_batch(
            data_type=t,
            count=count,
            model=args.model,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
        ))
        if len(types_to_run) > 1:
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
