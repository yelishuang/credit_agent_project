"""
项目统一配置 — 集中管理所有路径常量、模型参数和阈值。

所有模块通过 `from config import ...` 引用路径，避免各模块硬编码。
"""
from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── 输入数据 ────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
BASE_MODEL_PATH = DATA_DIR / "base_models" / "Qwen2.5-14B-Instruct"
KAGGLE_DATA_DIR = DATA_DIR / "kaggle_raw" / "home-credit-credit-risk-model-stability"
KAGGLE_PARQUET_DIR = KAGGLE_DATA_DIR / "parquet_files"
FEATURE_DEFS_PATH = KAGGLE_DATA_DIR / "feature_definitions.csv"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"

# ── 产出目录 ────────────────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 模型
MODELS_DIR = OUTPUTS_DIR / "models"
LGB_MODELS_DIR = MODELS_DIR / "credit_risk_lgb"
LORA_DIR = MODELS_DIR / "lora"
LORA_ADAPTER_PATH = LORA_DIR / "20260323_150000"

# SFT 数据
SFT_DATA_DIR = OUTPUTS_DIR / "sft_data"
SFT_DATA_RAW_DIR = SFT_DATA_DIR / "raw"
SFT_DATA_CURATED_DIR = SFT_DATA_DIR / "curated"
SFT_DATA_SELECTED_DIR = SFT_DATA_DIR / "selected"

# RAG 索引
RAG_INDEX_DIR = OUTPUTS_DIR / "rag_index"
RAG_INDEX_PATH = RAG_INDEX_DIR / "credit_risk.index"
RAG_CHUNKS_PATH = RAG_INDEX_DIR / "chunks.json"

# 评测
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

# ── 测试数据 ────────────────────────────────────────────────────────────
TESTS_DIR = PROJECT_ROOT / "tests"
TEST_CASES_PATH = TESTS_DIR / "evaluation" / "test_cases.json"

# ── 模型参数 ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-m3"
