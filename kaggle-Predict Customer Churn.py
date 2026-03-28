# =============================================================================
#  Customer Churn Prediction Pipeline
#  XGBoost · Feature Engineering · Probability Output
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(train_path="train.csv", test_path="test.csv"):
    try:
        train = pd.read_csv(train_path)
        test  = pd.read_csv(test_path)
        print(f"✅ Loaded  train: {train.shape}  |  test: {test.shape}")
        return train, test
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e}. Make sure CSVs are in the working directory.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

COLS_TO_DROP = [
    "tenure", "MonthlyCharges", "TotalCharges",
    *SERVICES,                          # replaced by Service_Count
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features; safe to call on both train and test sets."""
    df = df.copy()

    # Fix TotalCharges type (can be blank string in raw data)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # --- Aggregations ---
    df["Service_Count"]        = (df[SERVICES] == "Yes").sum(axis=1)
    df["TotalCharges_Log"]     = np.log1p(df["TotalCharges"])
    df["Monthly_Ratio"]        = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # --- Behavioural flags ---
    df["Is_High_Risk_Newbie"]  = ((df["tenure"] < 6) & (df["MonthlyCharges"] > 70)).astype(int)
    df["Tenure_Group"]         = pd.cut(
        df["tenure"], bins=[0, 12, 48, 100], labels=["New", "Medium", "Loyal"]
    )

    return df


def drop_raw_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove original columns that have been replaced by engineered ones."""
    return df.drop(columns=COLS_TO_DROP, errors="ignore")


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all categorical / object columns → integer dtype."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df.astype(int)


def add_interaction_features(df: pd.DataFrame, monthly_ratio_mean: float) -> pd.DataFrame:
    """Cross-feature interactions — must be added after encoding."""
    df = df.copy()

    e_check  = df.get("PaymentMethod_Electronic check",      pd.Series(0, index=df.index))
    two_year = df.get("Contract_Two year",                   pd.Series(0, index=df.index))
    cc_auto  = df.get("PaymentMethod_Credit card (automatic)", pd.Series(0, index=df.index))
    mail     = df.get("PaymentMethod_Mailed check",          pd.Series(0, index=df.index))
    fiber    = df.get("InternetService_Fiber optic",         pd.Series(0, index=df.index))
    tech     = df.get("TechSupport_Yes",                     pd.Series(0, index=df.index))

    # Payment pain & loyalty risk
    df["Payment_Pain_Index"]      = e_check * df["TotalCharges_Log"]
    df["Easy_To_Leave"]           = e_check * (1 - two_year)
    df["Is_Auto_Pay"]             = cc_auto + (1 - e_check - mail)
    df["Newbie_Electronic_Risk"]  = df["Is_High_Risk_Newbie"] * e_check

    # Service-quality vs payment friction
    df["Fiber_Without_Support"]   = fiber * (1 - tech)
    df["Is_Bundled_User"]         = (df["Service_Count"] >= 3).astype(int)
    df["High_Monthly_Check_Risk"] = e_check * (df["Monthly_Ratio"] > monthly_ratio_mean).astype(int)
    df["Fiber_Check_Pain"]        = fiber * e_check

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

XGB_PARAMS = dict(
    n_estimators    = 350,
    max_depth       = 4,
    learning_rate   = 0.02,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_alpha       = 1.5,
    reg_lambda      = 1.5,
    random_state    = 42,
    eval_metric     = "logloss",
)


def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)
    print(f"🚀 Training complete on {len(X):,} samples · {X.shape[1]} features")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICTION & SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────

def predict_and_save(
    model: xgb.XGBClassifier,
    test_final: pd.DataFrame,
    test_ids: pd.Series,
    out_path: str = "submission.csv",
) -> pd.DataFrame:
    churn_prob = model.predict_proba(test_final)[:, 1]
    submission = pd.DataFrame({"id": test_ids, "Churn": churn_prob})
    submission.to_csv(out_path, index=False)
    print(f"💾 Submission saved → {out_path}  ({len(submission):,} rows)")
    return submission


# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE IMPORTANCE CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model: xgb.XGBClassifier, feature_names, top_n: int = 20):
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
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    train_raw, test_raw = load_data()

    # ── Engineer + encode train ───────────────────────────────────────────────
    train_eng    = engineer_features(train_raw)
    train_clean  = drop_raw_cols(train_eng)
    train_enc    = encode(train_clean)

    monthly_mean = train_enc["Monthly_Ratio"].mean()

    train_final  = add_interaction_features(train_enc, monthly_mean)

    X = train_final.drop(columns=["Churn_Yes", "id"], errors="ignore")
    y = train_final["Churn_Yes"]

    # ── Train ─────────────────────────────────────────────────────────────────
    model = train_model(X, y)

    # ── Engineer + encode test ────────────────────────────────────────────────
    test_eng   = engineer_features(test_raw)
    test_clean = drop_raw_cols(test_eng)
    test_enc   = encode(test_clean)
    test_inter = add_interaction_features(test_enc, monthly_mean)

    # Align columns exactly to training set
    test_final = test_inter.reindex(columns=X.columns, fill_value=0)

    assert list(test_final.columns) == list(X.columns), "Column mismatch between train and test!"
    print(f"✅ Test features aligned: {test_final.shape[1]} columns")

    # ── Predict & save ────────────────────────────────────────────────────────
    predict_and_save(model, test_final, test_raw["id"])

    # ── Visualise ─────────────────────────────────────────────────────────────
    plot_feature_importance(model, X.columns)


if __name__ == "__main__":
    main()
