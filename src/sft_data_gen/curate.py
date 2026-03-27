"""
SFT 数据遴选与格式化脚本
- 校验 JSON 合法性、角色交替、tool_call 格式、内容完整性
- 过滤不合格样本，统计通过率
- 转为 Qwen2.5 chat template 兼容格式
- 划分 train/val (90%/10%)
"""

import json
import re
import os
import random
from pathlib import Path
from collections import defaultdict

# ============ 配置 ============
INPUT_DIR = "outputs/sft_data/raw"
OUTPUT_DIR = "outputs/sft_data/curated"
VALID_TOOLS = {"query_user_credit_data", "predict_risk_score", "search_knowledge_base"}
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

# ============ 校验函数 ============

def validate_json_line(line: str, line_num: int) -> tuple[dict | None, str | None]:
    """校验单行 JSON 是否合法"""
    try:
        data = json.loads(line.strip())
        if "messages" not in data:
            return None, "missing 'messages' key"
        if not isinstance(data["messages"], list):
            return None, "'messages' is not a list"
        return data, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


def validate_roles(messages: list[dict]) -> str | None:
    """校验角色交替是否正确"""
    if not messages:
        return "empty messages"
    if messages[0].get("role") != "system":
        return "first message is not system"
    if len(messages) < 3:
        return f"too few messages ({len(messages)})"

    # system 之后应该是 user/assistant 交替
    # tool_response (在 user 消息中) 应跟在含 tool_call 的 assistant 之后
    prev_role = "system"
    prev_had_tool_call = False

    for i, msg in enumerate(messages[1:], 1):
        role = msg.get("role")
        content = msg.get("content", "")

        if role not in ("user", "assistant"):
            return f"msg[{i}] invalid role: {role}"

        is_tool_response = "<tool_response>" in content if isinstance(content, str) else False

        if role == "user":
            if prev_role == "user":
                return f"msg[{i}] consecutive user messages"
            if is_tool_response and not prev_had_tool_call:
                return f"msg[{i}] tool_response without preceding tool_call"
        elif role == "assistant":
            if prev_role == "assistant":
                return f"msg[{i}] consecutive assistant messages"

        prev_had_tool_call = ("<tool_call>" in content if isinstance(content, str) else False) and role == "assistant"
        prev_role = role

    # 最后一条应该是 assistant
    if messages[-1].get("role") != "assistant":
        return "last message is not assistant"

    return None


def extract_tool_calls(content: str) -> list[dict]:
    """从 assistant 消息中提取 tool_call"""
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    calls = []
    for match in re.finditer(pattern, content, re.DOTALL):
        try:
            call = json.loads(match.group(1))
            calls.append(call)
        except json.JSONDecodeError:
            calls.append({"_parse_error": True, "raw": match.group(1)})
    return calls

def validate_tool_calls(messages: list[dict]) -> str | None:
    """校验 tool_call 中的函数名和参数"""
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str) or "<tool_call>" not in content:
            continue

        calls = extract_tool_calls(content)
        if not calls:
            return f"msg[{i}] has <tool_call> tag but no parseable call"

        for call in calls:
            if call.get("_parse_error"):
                return f"msg[{i}] tool_call JSON parse error: {call['raw'][:80]}"
            name = call.get("name")
            if name not in VALID_TOOLS:
                return f"msg[{i}] invalid tool name: {name}"
            args = call.get("arguments")
            if not isinstance(args, dict):
                return f"msg[{i}] tool '{name}' arguments is not a dict"
            # 参数合理性检查
            if name == "query_user_credit_data":
                if "user_id" not in args:
                    return f"msg[{i}] query_user_credit_data missing user_id"
            elif name == "predict_risk_score":
                if "features" not in args:
                    return f"msg[{i}] predict_risk_score missing features"
            elif name == "search_knowledge_base":
                if "query" not in args:
                    return f"msg[{i}] search_knowledge_base missing query"
    return None


def validate_final_response(messages: list[dict]) -> str | None:
    """校验最终 assistant 回复是否有实质内容"""
    last_msg = messages[-1]
    if last_msg.get("role") != "assistant":
        return "last message is not assistant"
    content = last_msg.get("content", "")
    if not isinstance(content, str):
        return "last assistant content is not string"
    # 去掉 tool_call 标签后检查剩余内容
    clean = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
    if len(clean) < 10:
        return f"final response too short ({len(clean)} chars)"
    # 检查是否截断（以不完整的句子结尾）
    if clean.endswith("...") and len(clean) < 50:
        return "final response appears truncated"
    return None


def validate_sample(data: dict) -> tuple[bool, str | None]:
    """综合校验单条样本"""
    messages = data.get("messages", [])

    err = validate_roles(messages)
    if err:
        return False, f"role_error: {err}"

    err = validate_tool_calls(messages)
    if err:
        return False, f"tool_error: {err}"

    err = validate_final_response(messages)
    if err:
        return False, f"response_error: {err}"

    return True, None


# ============ 主流程 ============

def main():
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(Path(INPUT_DIR).glob("*.jsonl"))
    print(f"Found {len(files)} JSONL files\n")

    all_valid = []
    stats = defaultdict(lambda: {"total": 0, "valid": 0, "errors": defaultdict(int)})

    for fpath in files:
        fname = fpath.name
        data_type = fname.split(".")[0]  # e.g. type_a_approval

        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                stats[data_type]["total"] += 1

                data, err = validate_json_line(line, line_num)
                if err:
                    stats[data_type]["errors"][f"json: {err}"] += 1
                    continue

                valid, err = validate_sample(data)
                if not valid:
                    stats[data_type]["errors"][err] += 1
                    continue

                stats[data_type]["valid"] += 1
                all_valid.append(data)

    # 打印统计
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)
    total_all, valid_all = 0, 0
    for dtype in sorted(stats.keys()):
        s = stats[dtype]
        total_all += s["total"]
        valid_all += s["valid"]
        rate = s["valid"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"\n{dtype}: {s['valid']}/{s['total']} passed ({rate:.1f}%)")
        if s["errors"]:
            for err, cnt in sorted(s["errors"].items(), key=lambda x: -x[1]):
                print(f"  - {err}: {cnt}")

    print(f"\nTotal: {valid_all}/{total_all} passed ({valid_all/total_all*100:.1f}%)")

    # 去重（基于 messages 内容的 hash）
    seen = set()
    deduped = []
    for item in all_valid:
        key = json.dumps(item["messages"], ensure_ascii=False, sort_keys=True)
        h = hash(key)
        if h not in seen:
            seen.add(h)
            deduped.append(item)
    dup_count = len(all_valid) - len(deduped)
    if dup_count > 0:
        print(f"\nRemoved {dup_count} duplicates, {len(deduped)} unique samples remain")
    all_valid = deduped

    # 划分 train/val
    random.shuffle(all_valid)
    split_idx = int(len(all_valid) * TRAIN_RATIO)
    train_data = all_valid[:split_idx]
    val_data = all_valid[split_idx:]

    # 写出（保留原始 messages 格式，Qwen2.5 chat template 兼容）
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in split_data:
                # 只保留 messages，去掉 _metadata
                out = {"messages": item["messages"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(split_data)} samples to {out_path}")

    print(f"\nDone! train={len(train_data)}, val={len(val_data)}")


if __name__ == "__main__":
    main()
