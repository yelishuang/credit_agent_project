"""
模型加载与缓存模块。
使用单例模式加载 5 个 LightGBM 模型和特征配置，避免重复加载。
"""
import joblib
from pathlib import Path

_MODEL_DIR = Path(__file__).parent.parent.parent / "outputs" / "credit_risk_lgb"

_cache = {}


def get_models():
    """返回 5 个 LGBMClassifier 模型列表（带缓存）。"""
    if "lgb_models" not in _cache:
        _cache["lgb_models"] = joblib.load(_MODEL_DIR / "lgb_models.joblib")
    return _cache["lgb_models"]


def get_feature_info():
    """返回 (cols, cat_cols) 元组。cols 为模型所需的 610 个特征名，cat_cols 为类别特征列表。"""
    if "feature_info" not in _cache:
        info = joblib.load(_MODEL_DIR / "notebook_info.joblib")
        _cache["feature_info"] = (info["cols"], info["cat_cols"])
    return _cache["feature_info"]


def get_all():
    """一次性返回 (models, cols, cat_cols)。"""
    models = get_models()
    cols, cat_cols = get_feature_info()
    return models, cols, cat_cols
