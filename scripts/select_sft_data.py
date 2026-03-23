"""
SFT 数据遴选脚本
- 逻辑一致性检查（type_a: risk_score vs 审批建议）
- 内容多样性（BGE-M3 embedding + 最大边际相关性采样）
- token 长度均衡（剔除极端长度）
- type_e 子类均衡采样
目标：~600 A, ~225 B, ~300 C, ~225 D, ~150 E
"""

import json
import re
import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CURATED_DIR = "outputs/sft_data_curated"
RAW_DIR = "outputs/sft_data"
OUTPUT_DIR = "outputs/sft_data_selected"
MODEL_PATH = "/home/dell/credit_agent_project/data/base_models/Qwen2.5-14B-Instruct"
EMBED_MODEL = "/home/dell/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
RANDOM_SEED = 42
MAX_LENGTH = 2048  # 训练时的 max_length

TARGETS = {"type_a": 600, "type_b": 225, "type_c": 300, "type_d": 225, "type_e": 150}
E_SUB_TARGET = 50  # 每个 E 子类目标数

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============ 工具函数 ============

def get_user_question(messages):
    """提取第一个非 tool_response 的 user 消息"""
    for m in messages:
        if m["role"] == "user" and "<tool_response>" not in m.get("content", ""):
            return m["content"]
    return ""

def get_last_assistant(messages):
    """提取最后一条 assistant 消息（去掉 tool_call 标签）"""
    for m in reversed(messages):
        if m["role"] == "assistant":
            content = m.get("content", "")
            clean = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
            return clean
    return ""

def extract_risk_score(messages):
    """从 tool_response 中提取 risk_score"""
    for m in messages:
        if m["role"] == "user" and "<tool_response>" in m.get("content", ""):
            content = m["content"]
            # 尝试从 tool_response 中解析 JSON
            match = re.search(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", content, re.DOTALL)
            if match:
                try:
                    resp = json.loads(match.group(1))
                    if "risk_score" in resp:
                        return resp["risk_score"], resp.get("risk_level", "")
                except json.JSONDecodeError:
                    pass
            # 也可能 tool_response 内容不是 JSON 而是纯文本
    return None, None

def extract_decision(last_response):
    """从最终 assistant 回复中提取审批建议"""
    text = last_response
    # 明确通过
    if re.search(r"建议通过|建议批准|建议审批通过|可以批准|可以通过|建议放款", text):
        if re.search(r"有条件|附加条件|条件通过|降低额度|缩短期限|上浮利率|需要担保|需要抵押", text):
            return "conditional_approve"
        return "approve"
    # 明确拒绝
    if re.search(r"建议拒绝|不建议通过|不建议批准|建议不予|拒绝贷款|不予批准|不予通过", text):
        return "reject"
    # 谨慎/观望
    if re.search(r"建议谨慎|需要进一步|暂缓|待.*核实|补充材料", text):
        return "pending"
    return "unknown"

def check_logical_consistency(messages):
    """检查 risk_score 与审批建议的逻辑一致性"""
    risk_score, risk_level = extract_risk_score(messages)
    if risk_score is None:
        return True, "no_risk_score"  # 没有 risk_score 的类型不检查

    last_resp = get_last_assistant(messages)
    decision = extract_decision(last_resp)

    # 逻辑矛盾检查
    if risk_score >= 0.7 and decision == "approve":
        return False, f"high_risk({risk_score})+approve"
    if risk_score <= 0.1 and decision == "reject":
        return False, f"low_risk({risk_score})+reject"

    return True, f"score={risk_score},decision={decision}"

def get_type_from_metadata(data):
    """从原始数据的 _metadata 获取类型"""
    meta = data.get("_metadata", {})
    t = meta.get("type", "")
    return t

def load_raw_with_metadata():
    """加载原始数据（带 _metadata），用于获取类型和子类信息"""
    raw_map = {}  # key: messages hash -> metadata
    for fpath in sorted(Path(RAW_DIR).glob("*.jsonl")):
        fname = fpath.stem  # e.g. type_a_approval
        with open(fpath) as f:
            for line in f:
                d = json.loads(line.strip())
                key = json.dumps(d["messages"], ensure_ascii=False, sort_keys=True)
                h = hash(key)
                raw_map[h] = {
                    "file_type": fname,
                    "metadata": d.get("_metadata", {}),
                }
    return raw_map

def compute_token_lengths(samples, tokenizer):
    """计算每条样本的 token 长度"""
    lengths = []
    for s in samples:
        text = tokenizer.apply_chat_template(s["messages"], tokenize=False)
        token_ids = tokenizer(text)["input_ids"]
        lengths.append(len(token_ids))
    return lengths

def diversity_select(embeddings, n_select, indices=None):
    """最大边际相关性 (MMR) 采样，确保多样性"""
    if indices is None:
        indices = list(range(len(embeddings)))
    if len(indices) <= n_select:
        return indices

    emb = embeddings[indices]
    # 归一化
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_norm = emb / norms

    # 贪心 MMR
    sim_matrix = emb_norm @ emb_norm.T
    selected = [0]  # 从第一个开始
    remaining = set(range(1, len(indices)))

    while len(selected) < n_select and remaining:
        # 对每个候选，计算与已选集合的最大相似度
        max_sim_to_selected = np.max(sim_matrix[list(remaining)][:, selected], axis=1)
        remaining_list = list(remaining)
        # 选最大相似度最小的（最不像已选的）
        best_idx = remaining_list[np.argmin(max_sim_to_selected)]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [indices[i] for i in selected]

def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading raw metadata...")
    raw_map = load_raw_with_metadata()

    # 加载去重后的数据
    all_samples = []
    for split in ["train", "val"]:
        path = f"{CURATED_DIR}/{split}.jsonl"
        with open(path) as f:
            for line in f:
                all_samples.append(json.loads(line.strip()))
    print(f"Loaded {len(all_samples)} curated samples")

    # 关联 metadata
    for s in all_samples:
        key = json.dumps(s["messages"], ensure_ascii=False, sort_keys=True)
        h = hash(key)
        meta_info = raw_map.get(h, {})
        s["_file_type"] = meta_info.get("file_type", "unknown")
        s["_metadata"] = meta_info.get("metadata", {})

    # 按类型分组
    by_type = defaultdict(list)
    for s in all_samples:
        by_type[s["_file_type"]].append(s)

    print("\n=== Current distribution ===")
    for t in sorted(by_type.keys()):
        print(f"  {t}: {len(by_type[t])}")

    # ========== Step 1: 逻辑一致性检查 (主要针对 type_a) ==========
    print("\n=== Step 1: Logical consistency check ===")
    inconsistent_count = 0
    for t in by_type:
        filtered = []
        for s in by_type[t]:
            ok, reason = check_logical_consistency(s["messages"])
            if ok:
                filtered.append(s)
            else:
                inconsistent_count += 1
                print(f"  REMOVED [{t}]: {reason}")
        by_type[t] = filtered
    print(f"  Removed {inconsistent_count} logically inconsistent samples")

    # ========== Step 2: Token 长度过滤 ==========
    print("\n=== Step 2: Token length filtering ===")
    length_removed = 0
    for t in by_type:
        lengths = compute_token_lengths(by_type[t], tokenizer)
        filtered = []
        for s, l in zip(by_type[t], lengths):
            s["_token_len"] = l
            if l > MAX_LENGTH:
                length_removed += 1
            else:
                filtered.append(s)
        by_type[t] = filtered
    print(f"  Removed {length_removed} samples exceeding {MAX_LENGTH} tokens")

    for t in sorted(by_type.keys()):
        lens = [s["_token_len"] for s in by_type[t]]
        print(f"  {t}: {len(by_type[t])} remaining, "
              f"len range [{min(lens)}-{max(lens)}], mean={np.mean(lens):.0f}")

    # ========== Step 3: Embedding + 多样性采样 ==========
    print("\n=== Step 3: Diversity sampling with embeddings ===")
    print("Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL)

    final_selected = []

    for t in sorted(TARGETS.keys()):
        file_type = {
            "type_a": "type_a_approval", "type_b": "type_b_query",
            "type_c": "type_c_knowledge", "type_d": "type_d_explanation",
            "type_e": "type_e_rejection",
        }[t]
        candidates = by_type.get(file_type, [])
        target_n = TARGETS[t]

        if t == "type_e":
            # E 类按子类均衡采样
            by_sub = defaultdict(list)
            for s in candidates:
                sub = s["_metadata"].get("subcategory", "unknown")
                by_sub[sub] = by_sub.get(sub, []) or []
                by_sub[sub].append(s)
            # 修正: defaultdict 已经处理了
            by_sub_fixed = defaultdict(list)
            for s in candidates:
                sub = s["_metadata"].get("subcategory", "unknown")
                by_sub_fixed[sub].append(s)

            print(f"\n  {t} subcategories: {dict((k, len(v)) for k, v in by_sub_fixed.items())}")

            e_selected = []
            for sub in sorted(by_sub_fixed.keys()):
                sub_candidates = by_sub_fixed[sub]
                sub_target = E_SUB_TARGET
                questions = [get_user_question(s["messages"]) for s in sub_candidates]
                embs = embed_model.encode(questions, normalize_embeddings=True)
                sel_indices = diversity_select(embs, min(sub_target, len(sub_candidates)))
                for i in sel_indices:
                    e_selected.append(sub_candidates[i])
                print(f"    {sub}: {len(sub_candidates)} -> {len(sel_indices)}")
            final_selected.extend(e_selected)
            print(f"  {t}: {len(candidates)} -> {len(e_selected)}")
        else:
            questions = [get_user_question(s["messages"]) for s in candidates]
            embs = embed_model.encode(questions, normalize_embeddings=True)
            sel_indices = diversity_select(embs, min(target_n, len(candidates)))
            selected = [candidates[i] for i in sel_indices]
            final_selected.extend(selected)
            print(f"  {t}: {len(candidates)} -> {len(selected)}")

    # ========== Step 4: 输出 ==========
    print(f"\n=== Final selection: {len(final_selected)} samples ===")

    # 统计最终分布
    final_by_type = defaultdict(int)
    for s in final_selected:
        final_by_type[s["_file_type"]] += 1
    for t in sorted(final_by_type.keys()):
        print(f"  {t}: {final_by_type[t]}")

    # 划分 train/val (90/10)
    random.shuffle(final_selected)
    split_idx = int(len(final_selected) * 0.9)
    train_data = final_selected[:split_idx]
    val_data = final_selected[split_idx:]

    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        out_path = f"{OUTPUT_DIR}/{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in split_data:
                out = {"messages": item["messages"]}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(split_data)} to {out_path}")

    # token 长度统计
    all_lens = [s["_token_len"] for s in final_selected]
    print(f"\n  Token length: min={min(all_lens)}, max={max(all_lens)}, "
          f"mean={np.mean(all_lens):.0f}, median={np.median(all_lens):.0f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
