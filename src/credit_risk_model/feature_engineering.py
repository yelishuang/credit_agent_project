"""
特征工程模块。
从原始 parquet 文件构建模型所需的 610 维特征，复现 hc_rank71 的 Pipeline 和 Aggregator。
"""
import datetime
from glob import glob
from pathlib import Path

import gc
import numpy as np
import pandas as pd
import polars as pl

from config import KAGGLE_PARQUET_DIR

# 默认数据目录
DEFAULT_DATA_DIR = KAGGLE_PARQUET_DIR


# ── Pipeline: 类型转换、日期处理、列过滤 ──────────────────────────────

class Pipeline:

    @staticmethod
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    @staticmethod
    def handle_dates(df):
        base_date = datetime.datetime(1900, 1, 1)
        for col in df.columns:
            if col[-1] in ("D",):
                days_since_base = (pl.col(col) - pl.lit(base_date)).dt.total_days()
                df = df.with_columns(days_since_base.alias(col + "_days_since_1900_D"))
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
        df = df.drop("date_decision", "MONTH")
        return df

    @staticmethod
    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.98:
                    df = df.drop(col)
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) and (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) or (freq > 200):
                    df = df.drop(col)
        return df


# ── Aggregator: 对 depth_1/depth_2 表做分组聚合 ──────────────────────

class Aggregator:

    @staticmethod
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_sum = [pl.sum(col).alias(f"sum_{col}") for col in cols]
        return expr_max + expr_last + expr_mean + expr_sum

    @staticmethod
    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_last + expr_mean

    @staticmethod
    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return expr_max + expr_last

    @staticmethod
    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return expr_max + expr_last

    @staticmethod
    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return expr_max + expr_last

    @staticmethod
    def get_exprs(df):
        return (Aggregator.num_expr(df) + Aggregator.date_expr(df) +
                Aggregator.str_expr(df) + Aggregator.other_expr(df) +
                Aggregator.count_expr(df))


# ── 读取与特征构建函数 ─────────────────────────────────────────────

def read_file(path, depth=None, case_ids=None):
    """
    读取单个 parquet 文件。
    如果 case_ids 不为 None，先按 case_id 过滤再加载，大幅节省内存。
    """
    if case_ids is not None:
        # 使用 scan → filter → collect 实现谓词下推，只读需要的行
        df = (pl.scan_parquet(path)
              .filter(pl.col("case_id").is_in(case_ids))
              .collect())
    else:
        df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df


def read_files(regex_path, depth=None, case_ids=None):
    """
    读取多个 parquet 文件（glob 匹配）并合并。
    如果 case_ids 不为 None，每个文件读取时先按 case_id 过滤。
    """
    chunks = []
    for path in sorted(glob(str(regex_path))):
        if case_ids is not None:
            df = (pl.scan_parquet(path)
                  .filter(pl.col("case_id").is_in(case_ids))
                  .collect())
        else:
            df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        chunks.append(df)
    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    return df


def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = df_base.with_columns(
        month_decision=pl.col("date_decision").dt.month(),
        weekday_decision=pl.col("date_decision").dt.weekday(),
    )
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
    df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base


def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols


def build_features_from_parquet(case_ids=None, data_dir=None, split="test"):
    """
    从原始 parquet 文件构建特征 DataFrame。

    参数:
        case_ids: 可选，筛选特定 case_id 的列表。None 表示全量（仅适合 test 等小数据集）。
        data_dir: 数据根目录，默认 DEFAULT_DATA_DIR。
        split: "test" 或 "train"。

    返回:
        Polars DataFrame，包含 case_id 和所有构建的特征列。
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    prefix = split

    # 规范化 case_ids
    _filter_ids = None
    if case_ids is not None:
        if isinstance(case_ids, (int, np.integer)):
            case_ids = [int(case_ids)]
        _filter_ids = [int(c) for c in case_ids]

    # 读取所有表 —— 传入 case_ids 使每个表在读取时就过滤
    data_store = {
        "df_base": read_file(split_dir / f"{prefix}_base.parquet",
                             case_ids=_filter_ids),
        "depth_0": [
            read_file(split_dir / f"{prefix}_static_cb_0.parquet",
                      case_ids=_filter_ids),
            read_files(split_dir / f"{prefix}_static_0_*.parquet",
                       case_ids=_filter_ids),
        ],
        "depth_1": [
            read_files(split_dir / f"{prefix}_applprev_1_*.parquet", 1,
                       case_ids=_filter_ids),
            read_file(split_dir / f"{prefix}_tax_registry_a_1.parquet", 1,
                      case_ids=_filter_ids),
            read_file(split_dir / f"{prefix}_tax_registry_b_1.parquet", 1,
                      case_ids=_filter_ids),
            read_file(split_dir / f"{prefix}_tax_registry_c_1.parquet", 1,
                      case_ids=_filter_ids),
            read_files(split_dir / f"{prefix}_credit_bureau_a_1_*.parquet", 1,
                       case_ids=_filter_ids),
            read_file(split_dir / f"{prefix}_credit_bureau_b_1.parquet", 1,
                      case_ids=_filter_ids),
            read_file(split_dir / f"{prefix}_other_1.parquet", 1,
                      case_ids=_filter_ids),
            read_file(split_dir / f"{prefix}_person_1.parquet", 1,
                      case_ids=_filter_ids),
            read_file(split_dir / f"{prefix}_deposit_1.parquet", 1,
                      case_ids=_filter_ids),
            read_file(split_dir / f"{prefix}_debitcard_1.parquet", 1,
                      case_ids=_filter_ids),
        ],
        "depth_2": [
            read_file(split_dir / f"{prefix}_credit_bureau_b_2.parquet", 2,
                      case_ids=_filter_ids),
            read_files(split_dir / f"{prefix}_credit_bureau_a_2_*.parquet", 2,
                       case_ids=_filter_ids),
            read_file(split_dir / f"{prefix}_applprev_2.parquet", 2,
                      case_ids=_filter_ids),
            read_file(split_dir / f"{prefix}_person_2.parquet", 2,
                      case_ids=_filter_ids),
        ],
    }

    df = feature_eng(**data_store)
    del data_store
    gc.collect()

    return df


def prepare_for_model(df_polars, cols, cat_cols):
    """
    将 Polars DataFrame 转为模型可用的 Pandas DataFrame。

    参数:
        df_polars: feature_eng 返回的 Polars DataFrame
        cols: 模型所需的 610 个特征名列表
        cat_cols: 类别特征列表

    返回:
        pandas DataFrame，以 case_id 为 index，包含模型所需的所有列。
    """
    # 只选模型需要的列（+ case_id）
    available = [c for c in cols if c in df_polars.columns]
    df_polars = df_polars.select(["case_id"] + available)

    df_pd, _ = to_pandas(df_polars, [c for c in cat_cols if c in available])

    # 补齐缺失列
    for c in cols:
        if c not in df_pd.columns:
            if c in cat_cols:
                df_pd[c] = pd.Categorical([np.nan] * len(df_pd))
            else:
                df_pd[c] = np.nan

    df_pd = df_pd.set_index("case_id")

    # 确保列顺序
    df_pd = df_pd[cols]
    # 类别列转 category
    for c in cat_cols:
        if df_pd[c].dtype.name != "category":
            df_pd[c] = df_pd[c].astype("category")

    return df_pd


def build_features_from_dict(features_dict, cols, cat_cols):
    """
    从用户字典构建单行 DataFrame，映射到模型所需的 610 列。
    缺失列填 NaN。

    参数:
        features_dict: 用户提供的特征字典
        cols: 模型所需的 610 个特征名列表
        cat_cols: 类别特征列表

    返回:
        pandas DataFrame，单行，列顺序与模型一致。
    """
    row = {}
    for c in cols:
        val = features_dict.get(c, np.nan)
        if c in cat_cols:
            row[c] = pd.Categorical([val])
        else:
            row[c] = [val]

    df = pd.DataFrame(row)
    return df
