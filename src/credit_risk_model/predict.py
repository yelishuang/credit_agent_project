"""
信用风险预测核心模块。
封装 hc_rank71 的 LightGBM 5 折集成模型，提供 predict_credit_risk() 函数。
"""
import gc
import numpy as np
import pandas as pd

from . import model_loader
from . import feature_engineering as fe
from . import feature_config as fc


def predict_credit_risk(features: dict, top_n: int = 5, data_dir: str = None) -> dict:
    """
    信用风险预测函数。

    两种输入模式：
    1. features 包含 "case_id" → 从 parquet 数据中查找并构建特征
    2. features 包含原始特征字段 → 直接映射到模型输入

    参数:
        features: 用户特征字典。
            模式1: {"case_id": 57543}
            模式2: {"credamount_770A": 20000, "mainoccupationinc_384A": 50000, ...}
        top_n: 返回的 top 风险因素数量，默认 5
        data_dir: 数据目录（仅模式1需要），默认使用 feature_engineering 中的默认路径

    返回:
        {
            "risk_score": 0.73,
            "risk_level": "高风险",
            "top_factors": ["负债收入比过高", "近期逾期次数多", ...]
        }
    """
    # 加载模型和特征配置
    models, cols, cat_cols = model_loader.get_all()

    # ── 构建特征 ─────────────────────────────────────────────────────
    if "case_id" in features:
        case_id = features["case_id"]
        # 判断是用 test 还是 train
        split = features.get("split", "test")
        df_polars = fe.build_features_from_parquet(
            case_ids=[case_id],
            data_dir=data_dir,
            split=split,
        )
        df_input = fe.prepare_for_model(df_polars, cols, cat_cols)
        del df_polars
        gc.collect()
    else:
        # 模式2：直接从字典构建
        df_input = fe.build_features_from_dict(features, cols, cat_cols)

    # ── 模型预测 ─────────────────────────────────────────────────────
    # 提取 LGB 所需列，数值列 NaN 填 0（与原始推理代码一致）
    lgb_input = df_input[cols].copy()
    num_cols = lgb_input.select_dtypes(exclude="category").columns
    lgb_input[num_cols] = lgb_input[num_cols].fillna(0)

    # 5 折预测取平均
    y_preds = []
    for model in models:
        pred = model.predict_proba(lgb_input)[:, 1]
        y_preds.append(pred)

    risk_score = float(np.mean(y_preds, axis=0)[0])
    risk_score = np.clip(risk_score, 0.0, 1.0)

    # ── 风险等级 ─────────────────────────────────────────────────────
    risk_level = fc.get_risk_level(risk_score)

    # ── Top 影响因素 ──────────────────────────────────────────────────
    # 使用 feature_importance (gain) 加权 + 特征值偏离程度
    importances = np.zeros(len(cols))
    for model in models:
        imp = model.feature_importances_
        importances += imp
    importances /= len(models)

    # 获取该样本的特征值（处理类别列：转为数值 NaN）
    sample_row = lgb_input.iloc[0]
    sample_values = np.zeros(len(cols))
    for i, c in enumerate(cols):
        v = sample_row[c]
        if isinstance(v, (int, float, np.integer, np.floating)):
            sample_values[i] = float(v) if not pd.isna(v) else 0.0
        else:
            # 类别列：非空标记为 1，空标记为 0
            sample_values[i] = 0.0 if pd.isna(v) else 1.0

    # 综合评分：importance × |特征值|（归一化）
    abs_values = np.abs(sample_values)
    abs_max = abs_values.max()
    if abs_max > 0:
        norm_values = abs_values / abs_max
    else:
        norm_values = abs_values

    # 综合得分 = importance * (0.5 + 0.5 * norm_value)
    combined_scores = importances * (0.5 + 0.5 * norm_values)

    # 取 top_n
    top_indices = np.argsort(combined_scores)[::-1][:top_n]
    top_feature_names = [cols[i] for i in top_indices]
    top_feature_values = []
    for i in top_indices:
        v = sample_row[cols[i]]
        if isinstance(v, (int, float, np.integer, np.floating)):
            top_feature_values.append(float(v) if not pd.isna(v) else None)
        else:
            top_feature_values.append(str(v) if not pd.isna(v) else None)
    top_importances = [float(importances[i]) for i in top_indices]

    top_factors = fc.explain_top_factors(
        top_feature_names, top_feature_values, top_importances
    )

    return {
        "risk_score": round(risk_score, 6),
        "risk_level": risk_level,
        "top_factors": top_factors,
    }


def predict_batch(case_ids: list, split: str = "test",
                  data_dir: str = None, top_n: int = 5) -> list:
    """
    批量预测多个 case_id 的信用风险。

    参数:
        case_ids: case_id 列表
        split: "test" 或 "train"
        data_dir: 数据目录
        top_n: 返回的 top 风险因素数量

    返回:
        [{case_id: int, risk_score: float, risk_level: str, top_factors: list}, ...]
    """
    models, cols, cat_cols = model_loader.get_all()

    df_polars = fe.build_features_from_parquet(
        case_ids=case_ids, data_dir=data_dir, split=split
    )
    df_input = fe.prepare_for_model(df_polars, cols, cat_cols)
    del df_polars
    gc.collect()

    # 数值列 NaN 填 0
    lgb_input = df_input[cols].copy()
    num_cols_list = lgb_input.select_dtypes(exclude="category").columns
    lgb_input[num_cols_list] = lgb_input[num_cols_list].fillna(0)

    # 5 折预测取平均
    y_preds = np.zeros(len(lgb_input))
    for model in models:
        y_preds += model.predict_proba(lgb_input)[:, 1]
    y_preds /= len(models)

    # Feature importance (共享)
    importances = np.zeros(len(cols))
    for model in models:
        importances += model.feature_importances_
    importances /= len(models)

    results = []
    for idx, (cid, score) in enumerate(zip(df_input.index, y_preds)):
        score = float(np.clip(score, 0.0, 1.0))
        risk_level = fc.get_risk_level(score)

        # Per-sample top factors
        sample_row = lgb_input.iloc[idx]
        sample_values = np.zeros(len(cols))
        for i, c in enumerate(cols):
            v = sample_row[c]
            if isinstance(v, (int, float, np.integer, np.floating)):
                sample_values[i] = float(v) if not pd.isna(v) else 0.0
            else:
                sample_values[i] = 0.0 if pd.isna(v) else 1.0
        abs_values = np.abs(sample_values)
        abs_max = abs_values.max()
        norm_values = abs_values / abs_max if abs_max > 0 else abs_values
        combined_scores = importances * (0.5 + 0.5 * norm_values)

        top_indices = np.argsort(combined_scores)[::-1][:top_n]
        top_feature_names = [cols[i] for i in top_indices]
        top_feature_values = []
        for i in top_indices:
            v = sample_row[cols[i]]
            if isinstance(v, (int, float, np.integer, np.floating)):
                top_feature_values.append(float(v) if not pd.isna(v) else None)
            else:
                top_feature_values.append(str(v) if not pd.isna(v) else None)

        top_factors = fc.explain_top_factors(top_feature_names, top_feature_values)

        results.append({
            "case_id": int(cid),
            "risk_score": round(score, 6),
            "risk_level": risk_level,
            "top_factors": top_factors,
        })

    return results
