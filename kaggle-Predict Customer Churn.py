# =============================================================================
#  Customer Churn Prediction — Kaggle Playground Series S6E3
#  Score: 0.91542 (ROC-AUC) | Rank: 1097 / ~3700
#  Model: XGBoost · Feature Engineering · Interaction Features
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


# ─────────────────────────────────────────────────────────────────────────────
# 1. 载入数据
# ─────────────────────────────────────────────────────────────────────────────

def load_data(train_path="train.csv", test_path="test.csv"):
    """读取训练集和测试集 CSV 文件"""
    try:
        train = pd.read_csv(train_path)
        test  = pd.read_csv(test_path)
        print(f"✅ Loaded  train: {train.shape}  |  test: {test.shape}")
        return train, test
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e}. Make sure CSVs are in the working directory.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 特征工程
# ─────────────────────────────────────────────────────────────────────────────

# 需要聚合为 Service_Count 的增值服务列
SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# 特征工程后不再需要的原始列（已被派生特征替代）
COLS_TO_DROP = [
    "tenure", "MonthlyCharges", "TotalCharges",
    *SERVICES,
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加派生特征，训练集和测试集均可调用。
    - Service_Count: 用户订阅的增值服务数量
    - TotalCharges_Log: 总消费额的对数变换，缓解右偏分布
    - Monthly_Ratio: 月均消费占总消费比例，反映消费稳定性
    - Is_High_Risk_Newbie: 新用户 + 高月费 = 高流失风险标志
    - Tenure_Group: 按在网时长分组（新用户 / 中期 / 忠实用户）
    """
    df = df.copy()

    # TotalCharges 原始数据中可能含空字符串，强制转为数值
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # 聚合特征
    df["Service_Count"]        = (df[SERVICES] == "Yes").sum(axis=1)
    df["TotalCharges_Log"]     = np.log1p(df["TotalCharges"])
    df["Monthly_Ratio"]        = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # 行为标志特征
    df["Is_High_Risk_Newbie"]  = ((df["tenure"] < 6) & (df["MonthlyCharges"] > 70)).astype(int)
    df["Tenure_Group"]         = pd.cut(
        df["tenure"], bins=[0, 12, 48, 100], labels=["New", "Medium", "Loyal"]
    )

    return df


def drop_raw_cols(df: pd.DataFrame) -> pd.DataFrame:
    """删除已被派生特征替代的原始列"""
    return df.drop(columns=COLS_TO_DROP, errors="ignore")


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """对所有类别型列进行 One-Hot 编码，并统一转为 int 类型"""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df.astype(int)


def add_interaction_features(df: pd.DataFrame, monthly_ratio_mean: float) -> pd.DataFrame:
    """
    添加交叉特征，捕捉特征组合带来的额外流失信号。
    需在编码之后调用。monthly_ratio_mean 必须来自训练集，避免数据泄漏。
    """
    df = df.copy()

    # 安全获取编码后的列（测试集可能缺少某些列）
    e_check  = df.get("PaymentMethod_Electronic check",        pd.Series(0, index=df.index))
    two_year = df.get("Contract_Two year",                     pd.Series(0, index=df.index))
    cc_auto  = df.get("PaymentMethod_Credit card (automatic)", pd.Series(0, index=df.index))
    mail     = df.get("PaymentMethod_Mailed check",            pd.Series(0, index=df.index))
    fiber    = df.get("InternetService_Fiber optic",           pd.Series(0, index=df.index))
    tech     = df.get("TechSupport_Yes",                       pd.Series(0, index=df.index))

    # 支付方式 × 合约类型 → 流失摩擦力指标
    df["Payment_Pain_Index"]      = e_check * df["TotalCharges_Log"]   # 电子支付 × 高消费
    df["Easy_To_Leave"]           = e_check * (1 - two_year)           # 电子支付 + 非长期合约
    df["Is_Auto_Pay"]             = cc_auto + (1 - e_check - mail)     # 是否自动扣款
    df["Newbie_Electronic_Risk"]  = df["Is_High_Risk_Newbie"] * e_check

    # 服务质量 × 支付摩擦 → 不满意风险
    df["Fiber_Without_Support"]   = fiber * (1 - tech)                 # 光纤用户但无技术支持
    df["Is_Bundled_User"]         = (df["Service_Count"] >= 3).astype(int)
    df["High_Monthly_Check_Risk"] = e_check * (df["Monthly_Ratio"] > monthly_ratio_mean).astype(int)
    df["Fiber_Check_Pain"]        = fiber * e_check

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. 训练模型
# ─────────────────────────────────────────────────────────────────────────────

XGB_PARAMS = dict(
    n_estimators     = 350,
    max_depth        = 4,
    learning_rate    = 0.02,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 1.5,   # L1 正则，抑制噪声特征
    reg_lambda       = 1.5,   # L2 正则，防止过拟合
    random_state     = 42,
    eval_metric      = "logloss",
)


def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """使用 XGBoost 训练分类模型"""
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)
    print(f"🚀 Training complete on {len(X):,} samples · {X.shape[1]} features")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. 打包预测结果
# ─────────────────────────────────────────────────────────────────────────────

def predict_and_save(
    model: xgb.XGBClassifier,
    test_final: pd.DataFrame,
    test_ids: pd.Series,
    out_path: str = "submission.csv",
) -> pd.DataFrame:
    """生成流失概率预测并保存为 Kaggle 提交格式"""
    churn_prob = model.predict_proba(test_final)[:, 1]
    submission = pd.DataFrame({"id": test_ids, "Churn": churn_prob})
    submission.to_csv(out_path, index=False)
    print(f"💾 Submission saved → {out_path}  ({len(submission):,} rows)")
    return submission


# ─────────────────────────────────────────────────────────────────────────────
# 5. 特征重要性图表
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model: xgb.XGBClassifier, feature_names, top_n: int = 20):
    """绘制 Top N 特征重要性柱状图，并打印前 5 名"""
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="magma", ax=ax)
    ax.set_title(f"Top {top_n} Features — Churn Attribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\n🕵️  Top 5 churn drivers:")
    print(importance_df.head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 6. 主要流水线
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 加载数据 ──────────────────────────────────────────────────────────────
    train_raw, test_raw = load_data()

    # ── 训练集：特征工程 + 编码 ───────────────────────────────────────────────
    train_eng   = engineer_features(train_raw)
    train_clean = drop_raw_cols(train_eng)
    train_enc   = encode(train_clean)

    # 用训练集均值计算交叉特征阈值（避免测试集数据泄漏）
    monthly_mean = train_enc["Monthly_Ratio"].mean()
    train_final  = add_interaction_features(train_enc, monthly_mean)

    X = train_final.drop(columns=["Churn_Yes", "id"], errors="ignore")
    y = train_final["Churn_Yes"]

    # ── 训练模型 ──────────────────────────────────────────────────────────────
    model = train_model(X, y)

    # ── 测试集：特征工程 + 编码（流程与训练集一致）───────────────────────────
    test_eng   = engineer_features(test_raw)
    test_clean = drop_raw_cols(test_eng)
    test_enc   = encode(test_clean)
    test_inter = add_interaction_features(test_enc, monthly_mean)

    # 对齐列名，确保测试集与训练集特征完全一致
    test_final = test_inter.reindex(columns=X.columns, fill_value=0)
    assert list(test_final.columns) == list(X.columns), "Column mismatch between train and test!"
    print(f"✅ Test features aligned: {test_final.shape[1]} columns")

    # ── 预测并保存提交文件 ────────────────────────────────────────────────────
    predict_and_save(model, test_final, test_raw["id"])

    # ── 特征重要性可视化 ──────────────────────────────────────────────────────
    plot_feature_importance(model, X.columns)


if __name__ == "__main__":
    main()
