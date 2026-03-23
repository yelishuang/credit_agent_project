"""
特征配置模块。
- 将模型 610 个特征映射到中文业务含义
- 风险等级划分
- 用于输出 top_factors 时翻译为可读的中文风险因素
"""
import csv
from pathlib import Path

# ── 特征定义文件 ─────────────────────────────────────────────────────
FEATURE_DEFS_PATH = Path(__file__).parent.parent.parent / "data" / "kaggle_raw" / "home-credit-credit-risk-model-stability" / "feature_definitions.csv"

# ── 风险等级阈值 ─────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "低风险": (0.0, 0.3),
    "中风险": (0.3, 0.6),
    "高风险": (0.6, 1.0),
}


def get_risk_level(score: float) -> str:
    """根据风险评分返回风险等级。"""
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "高风险"


# ── 聚合前缀到中文 ──────────────────────────────────────────────────
AGG_PREFIX_CN = {
    "max_": "最大",
    "last_": "最近",
    "mean_": "平均",
    "sum_": "总计",
    "min_": "最小",
    "median_": "中位数",
    "var_": "方差",
    "first_": "首次",
    "count_": "计数",
}


def _load_feature_definitions() -> dict:
    """从 CSV 加载原始特征名 → 英文描述的映射。"""
    mapping = {}
    if not FEATURE_DEFS_PATH.exists():
        return mapping
    with open(FEATURE_DEFS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            var = row.get("Variable", "").strip()
            desc = row.get("Description", "").strip()
            if var and desc:
                mapping[var] = desc
    return mapping


# 加载一次
_FEATURE_DEFS = _load_feature_definitions()

# ── 关键特征中文翻译（手工维护高频/重要特征）───────────────────────
_MANUAL_CN = {
    # 基本申请信息
    "month_decision": "申请决策月份",
    "weekday_decision": "申请决策星期",
    "credamount_770A": "贷款金额/信用额度",
    "applicationcnt_361L": "同邮箱关联申请数",
    "applications30d_658L": "近30天申请次数",
    "applicationscnt_1086L": "同电话号关联申请数",
    "applicationscnt_464L": "同雇主近30天申请数",
    "applicationscnt_867L": "同手机号关联申请数",
    "disbursedcredamount_1113A": "已发放贷款金额",
    "downpmt_116A": "首付金额",
    "homephncnt_628L": "家庭电话数量",
    "mobilephncnt_593L": "手机号数量",
    "numactivecreds_622L": "活跃信用账户数",
    "numactivecredschannel_414L": "活跃信用渠道数",
    "numactiverelcontr_750L": "活跃循环贷合同数",
    "numcontrs3months_479L": "近3个月合同数",
    "numnotactivated_1143L": "未激活信用数",
    "numpmtchanneldd_318L": "直接扣款还款渠道数",
    "numrejects9m_859L": "近9个月拒绝次数",
    "sellerplacecnt_915L": "关联销售点数",
    "isbidproduct_1095L": "是否交叉销售产品",

    # 收入与债务
    "mainoccupationinc_384A": "主要职业收入",
    "maininc_215A": "主要收入金额",
    "currdebt_22A": "当前债务金额",
    "currdebt_94A": "历史申请当前债务",
    "currdebtcredtyperange_828A": "各类信用当前债务",
    "totaldebt_9A": "总债务金额",
    "totaldebtoverduevalue_178A": "活跃合同逾期总额",
    "totaldebtoverduevalue_718A": "已关闭合同逾期总额",
    "totaloutstanddebtvalue_39A": "活跃合同未偿债务总额",
    "totaloutstanddebtvalue_668A": "已关闭合同未偿债务总额",
    "debtoutstand_525A": "现有合同未偿还金额",
    "debtoverdue_47A": "当前逾期金额",

    # 逾期相关
    "actualdpd_943P": "实际逾期天数",
    "actualdpdtolerance_344P": "容忍期逾期天数",
    "dpdmax_139P": "活跃合同最大逾期天数",
    "dpdmax_757P": "已关闭合同最大逾期天数",
    "maxdpdlast12m_727P": "近12个月最大逾期天数",
    "maxdpdlast24m_143P": "近24个月最大逾期天数",
    "maxdpdlast3m_392P": "近3个月最大逾期天数",
    "maxdpdlast6m_474P": "近6个月最大逾期天数",
    "maxdpdlast9m_1059P": "近9个月最大逾期天数",
    "maxdbddpdlast1m_3658939P": "近1个月最大逾期天数",
    "maxdbddpdtollast12m_3658940P": "近12个月最大容忍期逾期天数",
    "numinstlswithdpd5_4187116L": "逾期超5天的分期数",
    "numinstlswithdpd10_728L": "逾期超10天的分期数",
    "pctinstlsallpaidlate1d_3546856L": "逾期1天以上分期占比",
    "pctinstlsallpaidlat10d_839L": "逾期10天以上分期占比",

    # 还款行为
    "avgdbddpdlast24m_3658932P": "近24个月平均逾期天数",
    "avgdbddpdlast3m_4187120P": "近3个月平均逾期天数",
    "avginstallast24m_3658937A": "近24个月平均分期金额",
    "avglnamtstart24m_4525187A": "近24个月平均贷款金额",
    "avgpmtlast12m_4525200A": "近12个月平均还款金额",
    "cntpmts24_3658933L": "近24个月有还款的月数",
    "numinstpaidlate1d_3546852L": "逾期超1天的已还期数",
    "numinstpaidearly_338L": "提前还款期数",
    "numinstregularpaid_973L": "按时全额还款期数",
    "sumoutstandtotal_3546847A": "未偿还总额",

    # 信用额度
    "credlmt_1052A": "活跃贷款信用额度",
    "credlmt_228A": "已关闭贷款信用额度",
    "credlmt_935A": "信用局活跃贷款额度",
    "overdueamountmax_950A": "活跃合同最大逾期金额",
    "overdueamountmax_155A": "活跃合同最大逾期金额",
    "outstandingamount_362A": "活跃合同未偿金额",

    # 信用查询
    "days30_165L": "近30天征信查询次数",
    "days180_256L": "近180天征信查询次数",
    "days360_512L": "近360天征信查询次数",

    # 年龄与个人信息
    "birth_259D": "出生日期",
    "dateofbirth_337D": "出生日期",
    "dateofbirth_342D": "出生日期",
    "birthdate_574D": "出生日期(征信)",

    # 合同信息
    "contractssum_5085716L": "征信合同总数",
    "annuity_780A": "月供金额",

    # 税务
    "pmtaverage_4527227A": "税务扣款平均额",
    "pmtcount_4527229L": "税务扣款次数",
    "assignmentdate_238D": "税务分配日期",

    # 拒绝历史
    "for3years_128L": "近3年被拒次数",
    "for3years_504L": "近3年信用历史",
    "formonth_118L": "近1个月被拒次数",
    "forquarter_462L": "近1季度被拒次数",
    "forweek_601L": "近1周被拒次数",
    "foryear_818L": "近1年取消次数",

    # 风险评估
    "riskassesment_940T": "信用评估分数",
    "riskassesment_302T": "违约概率估计",
}


def get_feature_description(feature_name: str) -> str:
    """
    获取特征的中文描述。

    对于带聚合前缀的特征(如 max_xxx, last_xxx, mean_xxx, sum_xxx)，
    拆解出基础特征名并拼接中文前缀。
    """
    # 1. 直接查手工中文映射
    if feature_name in _MANUAL_CN:
        return _MANUAL_CN[feature_name]

    # 2. 尝试拆解聚合前缀
    agg_cn = ""
    base_name = feature_name
    for prefix, cn in AGG_PREFIX_CN.items():
        if feature_name.startswith(prefix):
            agg_cn = cn
            base_name = feature_name[len(prefix):]
            break

    # 3. 基础特征查手工映射
    if base_name in _MANUAL_CN:
        return f"{agg_cn}{_MANUAL_CN[base_name]}"

    # 4. 查 feature_definitions.csv
    if base_name in _FEATURE_DEFS:
        desc = _FEATURE_DEFS[base_name]
        if agg_cn:
            return f"{agg_cn} - {desc}"
        return desc

    # 5. 处理 _days_since_1900_D 后缀
    if base_name.endswith("_days_since_1900_D"):
        inner = base_name.replace("_days_since_1900_D", "")
        if inner in _FEATURE_DEFS:
            return f"{agg_cn}{_FEATURE_DEFS[inner]}(距1900天数)"
        if inner in _MANUAL_CN:
            return f"{agg_cn}{_MANUAL_CN[inner]}(距1900天数)"

    # 6. 带编号后缀（如 _7, _9）的分组特征
    for suffix in ["_0", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9",
                    "_10", "_11", "_12", "_13", "_14", "_15"]:
        if base_name.endswith(suffix):
            inner = base_name[:-(len(suffix))]
            if inner in _FEATURE_DEFS:
                return f"{agg_cn}{_FEATURE_DEFS[inner]}(表{suffix[1:]})"

    # 7. 兜底：返回原始特征名
    return feature_name


def explain_top_factors(feature_names: list, feature_values: list = None,
                        importances: list = None) -> list:
    """
    将 top N 特征转换为人类可读的中文风险因素列表。

    参数:
        feature_names: 特征名列表
        feature_values: 对应特征值（可选，用于补充说明）
        importances: 对应重要性分数（可选）

    返回:
        中文风险因素描述列表
    """
    factors = []
    for i, name in enumerate(feature_names):
        desc = get_feature_description(name)
        if feature_values is not None and i < len(feature_values):
            val = feature_values[i]
            if val is not None and str(val) != "nan":
                desc = f"{desc}（值: {val:.4g}）" if isinstance(val, float) else f"{desc}（值: {val}）"
        factors.append(desc)
    return factors
