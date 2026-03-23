"""
RAG 知识库构建脚本

功能：加载 RAG/ 目录下的 markdown 文档 → 按 ## 标题切分 → BGE 编码 → 存入 FAISS → 持久化

输出：
  - rag_index/credit_risk.index   (FAISS 索引)
  - rag_index/chunks.json         (文本块 + 元数据)

使用：
  python build_rag_index.py --doc_dir ./RAG --output_dir ./rag_index
"""

import argparse
import json
import os
import re

os.environ["HF_HUB_OFFLINE"] = "1"

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────
# 1. Markdown-aware 切分：按 ## 标题切块
# ──────────────────────────────────────────────

def split_markdown_by_heading(text: str, source: str, max_chunk_size: int = 800) -> list[dict]:
    """
    按 ## 及以上标题切分 markdown 文本。
    如果某个 section 超过 max_chunk_size 字符，再按段落二次切分。
    每个 chunk 保留所属标题作为上下文前缀。
    """
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    sections = []
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return [{"content": text.strip(), "source": source, "heading": ""}] if text.strip() else []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append({"heading": "", "body": preamble})

    for i, match in enumerate(matches):
        heading = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append({"heading": heading, "body": body})

    # 对过长 section 做二次切分
    chunks = []
    for sec in sections:
        heading = sec["heading"]
        body = sec["body"]

        if len(body) <= max_chunk_size:
            content = f"{heading}\n\n{body}" if heading else body
            chunks.append({"content": content, "source": source, "heading": heading})
        else:
            paragraphs = re.split(r"\n{2,}", body)
            current = ""
            for para in paragraphs:
                if len(current) + len(para) + 2 > max_chunk_size and current:
                    content = f"{heading}\n\n{current}" if heading else current
                    chunks.append({"content": content, "source": source, "heading": heading})
                    current = para
                else:
                    current = f"{current}\n\n{para}" if current else para
            if current.strip():
                content = f"{heading}\n\n{current}" if heading else current
                chunks.append({"content": content, "source": source, "heading": heading})

    return chunks


# ──────────────────────────────────────────────
# 2. 加载所有文档并切分
# ──────────────────────────────────────────────

def load_and_split(doc_dir: str, max_chunk_size: int = 800) -> tuple[list[str], list[dict]]:
    all_chunks = []
    all_metadata = []

    for fname in sorted(os.listdir(doc_dir)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(doc_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        doc_chunks = split_markdown_by_heading(text, source=fname, max_chunk_size=max_chunk_size)
        for i, chunk in enumerate(doc_chunks):
            all_chunks.append(chunk["content"])
            all_metadata.append({
                "source": chunk["source"],
                "heading": chunk["heading"],
                "chunk_idx": i,
            })

    return all_chunks, all_metadata


# ──────────────────────────────────────────────
# 3. 编码 + 建索引 + 持久化
# ──────────────────────────────────────────────

def build_index(
    chunks: list[str],
    metadata: list[dict],
    model_name: str,
    output_dir: str,
):
    print(f"加载 embedding 模型: {model_name} ...")
    embed_model = SentenceTransformer(model_name)
    dimension = embed_model.get_sentence_embedding_dimension()

    print(f"编码 {len(chunks)} 个 chunks (dim={dimension}) ...")
    vectors = embed_model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype(np.float32)

    print("构建 FAISS IndexFlatIP ...")
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    print(f"索引中共 {index.ntotal} 个向量")

    # 持久化
    os.makedirs(output_dir, exist_ok=True)

    index_path = os.path.join(output_dir, "credit_risk.index")
    faiss.write_index(index, index_path)
    print(f"索引已保存: {index_path}")

    chunks_path = os.path.join(output_dir, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "metadata": metadata}, f, ensure_ascii=False, indent=2)
    print(f"chunks 已保存: {chunks_path}")

    # 输出切分统计
    sources = {}
    for m in metadata:
        sources[m["source"]] = sources.get(m["source"], 0) + 1
    print("\n切分统计:")
    for src, cnt in sources.items():
        print(f"  {src}: {cnt} chunks")
    lengths = [len(c) for c in chunks]
    print(f"\nchunk 长度: min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths)/len(lengths):.0f}, median={sorted(lengths)[len(lengths)//2]}")


# ──────────────────────────────────────────────
# 4. 检索函数（供外部导入使用）
# ──────────────────────────────────────────────

BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关段落："


def load_retriever(index_path: str, chunks_json_path: str, model_name: str):
    """加载已构建的索引，返回 search 函数。"""
    index = faiss.read_index(index_path)
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    metadata = data["metadata"]
    embed_model = SentenceTransformer(model_name)

    def search(query: str, top_k: int = 3) -> list[dict]:
        query_vec = embed_model.encode(
            [BGE_QUERY_PREFIX + query],
            normalize_embeddings=True,
        ).astype(np.float32)
        scores, indices = index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "content": chunks[idx],
                "source": metadata[idx]["source"],
                "heading": metadata[idx]["heading"],
                "score": float(score),
            })
        return results

    return search


# ──────────────────────────────────────────────
# 5. 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
    parser.add_argument("--doc_dir", default="./data/knowledge_base", help="markdown 文档目录")
    parser.add_argument("--output_dir", default="./outputs/rag_index", help="索引输出目录")
    parser.add_argument("--model", default="BAAI/bge-m3", help="embedding 模型")
    parser.add_argument("--max_chunk_size", type=int, default=800,
                        help="单个 chunk 最大字符数，超过则按段落二次切分")
    args = parser.parse_args()

    chunks, metadata = load_and_split(args.doc_dir, args.max_chunk_size)
    if not chunks:
        print("未找到任何文档内容，请检查 --doc_dir")
        return

    build_index(chunks, metadata, args.model, args.output_dir)
    print("\n构建完成。")


if __name__ == "__main__":
    main()
