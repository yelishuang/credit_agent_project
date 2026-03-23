"""
credit_risk_model 包。
提供 predict_credit_risk() 和 predict_batch() 作为顶层 API。
"""
from .predict import predict_credit_risk, predict_batch

__all__ = ["predict_credit_risk", "predict_batch"]
