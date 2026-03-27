"""
RAG 向量检索质量自动评测脚本

原理：
  1. 对每个文档 chunk，用 Claude API 自动生成 3 个检索 query（精确/模糊/换说法）
  2. 用生成的 query 反查 FAISS 索引，检查原 chunk 是否出现在 top-k 结果中
  3. 计算 Hit@1、Hit@3、MRR 三个标准 IR 指标，量化检索质量

验收阈值：
  - Hit@1 > 60%（目标 > 75%）
  - Hit@3 > 80%（目标 > 90%）
  - MRR   > 0.65（目标 > 0.80）

未达标排查优先级：
  1. chunk_size 不合适 → 对比 256 / 512 / 768
  2. BGE query 前缀未添加
  3. 文档内容质量（太泛、重复度高）

使用方式：
  python spec/rag_retrieval_eval.py \
      --chunks_json ./rag_index/chunks.json \
      --faiss_index ./rag_index/credit_risk.index \
      --eval_output ./rag_index/eval_report.json \
      --model claude-sonnet-4-20250514 \
      --top_k 3 \
      --sample_ratio 1.0
"""

import argparse
import json
import os
import re
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"

import anthropic
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# 1. 加载已有索引和 chunks
# ──────────────────────────────────────────────

def load_index_and_chunks(faiss_path: str, chunks_json_path: str):
    index = faiss.read_index(faiss_path)
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    metadata = data["metadata"]
    assert index.ntotal == len(chunks), (
        f"索引向量数 ({index.ntotal}) 与 chunk 数 ({len(chunks)}) 不一致"
    )
    return index, chunks, metadata


# ──────────────────────────────────────────────
# 2. 用 Claude API 为每个 chunk 生成评测 query
# ──────────────────────────────────────────────

QUERY_GEN_PROMPT = """根据以下文档片段，生成3个用户可能会用来检索这段内容的自然语言问题。
要求多样化：
1. 一个精确查询（直接使用片段中的关键术语）
2. 一个模糊/口语化查询（普通用户的提问方式）
3. 一个语义相关但换了说法的查询（同义改写）

文档片段：
{chunk}

只输出JSON数组，不要用markdown代码块包裹，不要输出任何其他文字：
["问题1", "问题2", "问题3"]"""


def extract_json_array(text: str) -> list:
    """从模型返回文本中提取 JSON 数组，兼容 markdown 代码块包裹。"""
    # 去掉 markdown 代码块
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return json.loads(text.strip())


def generate_eval_queries(
    client: anthropic.Anthropic,
    chunks: list[str],
    model: str,
    sample_ratio: float = 1.0,
) -> dict[int, list[str]]:
    """
    返回 {chunk_index: [query1, query2, query3], ...}
    sample_ratio < 1.0 时随机采样部分 chunk 以节省 API 费用。
    """
    indices = list(range(len(chunks)))
    if sample_ratio < 1.0:
        import random
        k = max(1, int(len(indices) * sample_ratio))
        indices = sorted(random.sample(indices, k))

    eval_pairs: dict[int, list[str]] = {}
    total = len(indices)

    for i, idx in enumerate(indices):
        chunk = chunks[idx]
        if len(chunk.strip()) < 20:
            continue

        max_retries = 8
        for attempt in range(max_retries):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=500,
                    messages=[{
                        "role": "user",
                        "content": QUERY_GEN_PROMPT.format(chunk=chunk),
                    }],
                )
                raw_text = resp.content[0].text
                queries = extract_json_array(raw_text)
                assert isinstance(queries, list) and len(queries) == 3
                assert all(isinstance(q, str) and len(q) > 0 for q in queries)
                eval_pairs[idx] = queries
                print(f"  [OK] chunk {idx} ({i+1}/{total})")
                break
            except (json.JSONDecodeError, AssertionError) as e:
                raw_preview = raw_text[:200] if 'raw_text' in dir() else 'N/A'
                if attempt == max_retries - 1:
                    print(f"  [FAIL] chunk {idx} 经 {max_retries} 次重试仍失败")
                    print(f"         错误: {e}")
                    print(f"         模型原始返回: {raw_preview}")
                    print(f"         chunk 内容前100字: {chunk[:100]}")
                    raise RuntimeError(f"chunk {idx} 生成失败，终止评测") from e
                else:
                    wait = min(2 ** attempt, 10)
                    print(f"  [RETRY] chunk {idx} 第{attempt+1}次失败 ({e}), {wait}s 后重试...")
                    time.sleep(wait)
            except anthropic.APIError as e:
                if attempt == max_retries - 1:
                    print(f"  [FAIL] chunk {idx} API 错误: {e}")
                    raise RuntimeError(f"chunk {idx} API 调用失败，终止评测") from e
                else:
                    wait = min(2 ** attempt, 30)
                    print(f"  [RETRY] chunk {idx} API错误 ({e}), {wait}s 后重试...")
                    time.sleep(wait)

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  query 生成进度: {i + 1}/{total}")

    return eval_pairs


# ──────────────────────────────────────────────
# 3. 检索 + 计算指标
# ──────────────────────────────────────────────

BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关段落："


def evaluate_retrieval(
    eval_pairs: dict[int, list[str]],
    embed_model: SentenceTransformer,
    index: faiss.Index,
    top_k: int = 3,
) -> dict:
    hit_at_1 = 0
    hit_at_3 = 0
    mrr_sum = 0.0
    total = 0
    per_query_results = []

    for chunk_idx, queries in eval_pairs.items():
        for query in queries:
            query_vec = embed_model.encode(
                [BGE_QUERY_PREFIX + query],
                normalize_embeddings=True,
            ).astype(np.float32)

            scores, indices = index.search(query_vec, top_k)
            retrieved = indices[0].tolist()

            total += 1
            hit = chunk_idx in retrieved
            rank = (retrieved.index(chunk_idx) + 1) if hit else None

            if hit:
                hit_at_3 += 1
                mrr_sum += 1.0 / rank
                if rank == 1:
                    hit_at_1 += 1

            per_query_results.append({
                "chunk_idx": chunk_idx,
                "query": query,
                "hit": hit,
                "rank": rank,
                "top_scores": [float(s) for s in scores[0]],
            })

    n = max(total, 1)
    metrics = {
        "total_queries": total,
        "hit_at_1": hit_at_1 / n,
        "hit_at_3": hit_at_3 / n,
        "mrr": mrr_sum / n,
    }
    return {"metrics": metrics, "details": per_query_results}


# ──────────────────────────────────────────────
# 4. 报告输出
# ──────────────────────────────────────────────

THRESHOLDS = {
    "hit_at_1": {"pass": 0.60, "good": 0.75},
    "hit_at_3": {"pass": 0.80, "good": 0.90},
    "mrr":      {"pass": 0.65, "good": 0.80},
}


def print_report(metrics: dict):
    print("\n" + "=" * 50)
    print("  RAG 检索质量评测报告")
    print("=" * 50)
    print(f"  总评测 query 数: {metrics['total_queries']}")
    print()
    for key in ["hit_at_1", "hit_at_3", "mrr"]:
        val = metrics[key]
        t = THRESHOLDS[key]
        if val >= t["good"]:
            status = "GOOD"
        elif val >= t["pass"]:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  {key:8s}: {val:.2%}  [{status}]  (及格>{t['pass']:.0%}, 目标>{t['good']:.0%})")
    print("=" * 50)


# ──────────────────────────────────────────────
# 5. 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG 检索质量自动评测")
    parser.add_argument("--chunks_json", required=True, help="chunks.json 路径")
    parser.add_argument("--faiss_index", required=True, help="FAISS 索引文件路径")
    parser.add_argument("--eval_output", default="./outputs/rag_index/eval_report.json",
                        help="评测结果输出路径")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="用于生成 query 的 Claude 模型")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                        help="采样比例，<1.0 可节省 API 费用")
    parser.add_argument("--eval_pairs_cache", default=None,
                        help="已生成的 eval_pairs JSON 缓存路径，避免重复调 API")
    args = parser.parse_args()

    # 加载
    print("[1/4] 加载索引和 chunks ...")
    index, chunks, metadata = load_index_and_chunks(args.faiss_index, args.chunks_json)
    print(f"  共 {len(chunks)} 个 chunks, 索引维度 {index.d}")

    # 生成或加载 eval pairs
    if args.eval_pairs_cache and os.path.exists(args.eval_pairs_cache):
        print("[2/4] 从缓存加载 eval_pairs ...")
        with open(args.eval_pairs_cache, "r", encoding="utf-8") as f:
            raw = json.load(f)
        eval_pairs = {int(k): v for k, v in raw.items()}
    else:
        print("[2/4] 调用 Claude API 生成评测 query ...")
        client = anthropic.Anthropic()
        eval_pairs = generate_eval_queries(client, chunks, args.model, args.sample_ratio)
        # 缓存保存
        cache_path = args.eval_pairs_cache or args.eval_output.replace(".json", "_pairs.json")
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(eval_pairs, f, ensure_ascii=False, indent=2)
        print(f"  eval_pairs 已缓存至 {cache_path}")

    print(f"  共 {len(eval_pairs)} 个 chunk 参与评测, {sum(len(v) for v in eval_pairs.values())} 条 query")

    # 评测
    print("[3/4] 加载 BGE 模型并执行检索评测 ...")
    embed_model = SentenceTransformer("BAAI/bge-m3")
    result = evaluate_retrieval(eval_pairs, embed_model, index, top_k=args.top_k)

    # 输出
    print("[4/4] 生成报告 ...")
    print_report(result["metrics"])

    os.makedirs(os.path.dirname(args.eval_output) or ".", exist_ok=True)
    with open(args.eval_output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至 {args.eval_output}")


if __name__ == "__main__":
    main()
