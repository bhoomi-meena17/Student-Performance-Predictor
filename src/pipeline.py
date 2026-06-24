"""
student_performance/src/pipeline.py
Full ML pipeline: data loading → cleaning → EDA → feature engineering →
label encoding → Linear Regression → evaluation → model persistence.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ─────────────────────────── paths ───────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "students.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\n{'='*55}")
    print("  STEP 1 — DATA LOADING")
    print(f"{'='*55}")
    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS  
# ══════════════════════════════════════════════════════════════
def exploratory_analysis(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print("  STEP 2 — EXPLORATORY DATA ANALYSIS")
    print(f"{'='*55}")
    print("\n  [2a] Data types & nulls")
    info = pd.DataFrame({
        "dtype":   df.dtypes,
        "nulls":   df.isnull().sum(),
        "null_%":  (df.isnull().mean() * 100).round(2),
    })
    print(info.to_string())

    print("\n  [2b] Numeric summary")
    print(df.describe().round(2).to_string())

    print("\n  [2c] Categorical frequencies")
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        print(f"\n    {col}:")
        print(df[col].value_counts().to_string(header=False))

    _plot_eda(df)


def _plot_eda(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT   = "#00d4ff"
    ACCENT2  = "#ff6b6b"
    GREY     = "#1e2130"
    TEXT     = "#e0e6f0"

    def _ax(row, col, colspan=1):
        ax = fig.add_subplot(gs[row, col] if colspan == 1 else gs[row, col:col+colspan])
        ax.set_facecolor(GREY)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3250")
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        return ax

    # Histogram of exam scores
    ax1 = _ax(0, 0)
    ax1.hist(df["exam_score"].dropna(), bins=30, color=ACCENT, edgecolor="#0f1117", alpha=0.9)
    ax1.set_title("Exam Score Distribution")
    ax1.set_xlabel("Score")
    ax1.set_ylabel("Count")
    mean_score = df["exam_score"].mean()
    ax1.axvline(mean_score, color=ACCENT2, linestyle="--", linewidth=1.5)
    ax1.text(mean_score + 1, ax1.get_ylim()[1] * 0.88,
             f"μ={mean_score:.1f}", color=ACCENT2, fontsize=8)

    # Study hours vs score
    ax2 = _ax(0, 1)
    ax2.scatter(df["study_hours_per_day"], df["exam_score"],
                alpha=0.35, s=12, color=ACCENT, edgecolors="none")
    ax2.set_title("Study Hours vs Score")
    ax2.set_xlabel("Study Hours / Day")
    ax2.set_ylabel("Exam Score")
    # trend line
    mask = df[["study_hours_per_day","exam_score"]].notna().all(axis=1)
    z = np.polyfit(df.loc[mask,"study_hours_per_day"], df.loc[mask,"exam_score"], 1)
    xr = np.linspace(df["study_hours_per_day"].min(), df["study_hours_per_day"].max(), 200)
    ax2.plot(xr, np.polyval(z, xr), color=ACCENT2, linewidth=1.8)

    # Absences vs score
    ax3 = _ax(0, 2)
    ax3.scatter(df["absences"], df["exam_score"],
                alpha=0.3, s=12, color="#ffa94d", edgecolors="none")
    ax3.set_title("Absences vs Score")
    ax3.set_xlabel("Absences")
    ax3.set_ylabel("Exam Score")

    # Score by gender
    ax4 = _ax(1, 0)
    genders = df["gender"].dropna().unique()
    bp_data  = [df.loc[df["gender"] == g, "exam_score"].dropna() for g in genders]
    bp = ax4.boxplot(bp_data, labels=genders, patch_artist=True,
                     medianprops=dict(color=ACCENT2, linewidth=2))
    colors = [ACCENT, "#a78bfa"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_title("Score by Gender")
    ax4.set_ylabel("Exam Score")

    # Score by test prep
    ax5 = _ax(1, 1)
    preps = df["test_preparation_course"].dropna().unique()
    bp_data2 = [df.loc[df["test_preparation_course"] == p, "exam_score"].dropna() for p in preps]
    bp2 = ax5.boxplot(bp_data2, labels=preps, patch_artist=True,
                      medianprops=dict(color=ACCENT2, linewidth=2))
    for patch, color in zip(bp2["boxes"], ["#34d399","#f87171"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_title("Score by Test Prep")
    ax5.set_ylabel("Exam Score")

    # Score by lunch
    ax6 = _ax(1, 2)
    lunches = df["lunch"].dropna().unique()
    bp_data3 = [df.loc[df["lunch"] == l, "exam_score"].dropna() for l in lunches]
    bp3 = ax6.boxplot(bp_data3, labels=lunches, patch_artist=True,
                      medianprops=dict(color=ACCENT2, linewidth=2))
    for patch, color in zip(bp3["boxes"], ["#60a5fa","#fb923c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_title("Score by Lunch Type")
    ax6.set_ylabel("Exam Score")

    # Correlation heatmap (numeric)
    ax7 = _ax(2, 0, colspan=2)
    num_df = df.select_dtypes(include=np.number)
    corr   = num_df.corr()
    im = ax7.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax7.set_xticks(range(len(corr.columns)))
    ax7.set_yticks(range(len(corr.columns)))
    ax7.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=7, color=TEXT)
    ax7.set_yticklabels(corr.columns, fontsize=7, color=TEXT)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax7.text(j, i, f"{corr.iloc[i,j]:.2f}",
                     ha="center", va="center", fontsize=6.5, color="white")
    ax7.set_title("Correlation Heatmap")
    plt.colorbar(im, ax=ax7, fraction=0.03, pad=0.02)

    # Parental education vs score
    ax8 = _ax(2, 2)
    edu_order = ["some high school","high school","some college",
                 "associate's degree","bachelor's degree","master's degree"]
    means  = df.groupby("parental_level_of_education")["exam_score"].mean().reindex(edu_order)
    bars   = ax8.barh(range(len(edu_order)), means.values, color=ACCENT, alpha=0.85)
    ax8.set_yticks(range(len(edu_order)))
    ax8.set_yticklabels([e.replace("'s", "") for e in edu_order], fontsize=7, color=TEXT)
    ax8.set_title("Avg Score by Parental Edu")
    ax8.set_xlabel("Mean Score")
    for bar, val in zip(bars, means.values):
        ax8.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=7, color=TEXT)

    fig.suptitle("Student Performance — Exploratory Data Analysis",
                 fontsize=15, color=TEXT, fontweight="bold", y=0.98)

    out = os.path.join(PLOT_DIR, "eda.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  EDA plot saved → {out}")


# ══════════════════════════════════════════════════════════════
# 3. DATA CLEANING
# ══════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  STEP 3 — DATA CLEANING")
    print(f"{'='*55}")
    df = df.copy()

    # Missing values
    before = df.isnull().sum().sum()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    after = df.isnull().sum().sum()
    print(f"  Missing values: {before} → {after}")

    # Outlier capping (IQR) on exam_score
    Q1, Q3 = df["exam_score"].quantile([0.25, 0.75])
    IQR     = Q3 - Q1
    lo, hi  = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    out_cnt = ((df["exam_score"] < lo) | (df["exam_score"] > hi)).sum()
    df["exam_score"] = df["exam_score"].clip(lo, hi)
    print(f"  Outliers capped in exam_score: {out_cnt}")

    # Duplicates
    dups = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"  Duplicate rows removed: {dups}")

    print(f"  Final shape after cleaning: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  STEP 4 — FEATURE ENGINEERING")
    print(f"{'='*55}")
    df = df.copy()

    # Study efficiency: score benefit per study hour (proxy)
    df["study_sleep_ratio"] = df["study_hours_per_day"] / (df["sleep_hours"] + 1e-6)
    print("  + study_sleep_ratio (study_hours / sleep_hours)")

    # Attendance rate (0-1)
    max_days = df["absences"].max() + 1
    df["attendance_rate"] = 1 - (df["absences"] / max_days)
    print("  + attendance_rate (1 - absences/max)")

    # Parental education ordinal
    edu_map = {
        "some high school": 0, "high school": 1, "some college": 2,
        "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5,
    }
    df["parental_edu_level"] = df["parental_level_of_education"].map(edu_map)
    print("  + parental_edu_level (ordinal 0-5)")

    # Composite support score
    df["support_score"] = (
        (df["lunch"] == "standard").astype(int)
        + (df["test_preparation_course"] == "completed").astype(int)
        + (df["extracurricular_activities"] == "Yes").astype(int)
    )
    print("  + support_score (lunch + test_prep + extracurricular)")

    # Interaction: study hours × test prep
    df["study_x_prep"] = (
        df["study_hours_per_day"]
        * (df["test_preparation_course"] == "completed").astype(int)
    )
    print("  + study_x_prep (study_hours × test_prep)")

    return df


# ══════════════════════════════════════════════════════════════
# 5. LABEL ENCODING
# ══════════════════════════════════════════════════════════════
def encode_features(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print("  STEP 5 — LABEL ENCODING")
    print(f"{'='*55}")
    df     = df.copy()
    le_map = {}

    binary_map = {
        "gender":                    {"Male": 0, "Female": 1},
        "lunch":                     {"standard": 1, "free/reduced": 0},
        "test_preparation_course":   {"none": 0, "completed": 1},
        "extracurricular_activities": {"No": 0, "Yes": 1},
    }
    for col, mapping in binary_map.items():
        df[col] = df[col].map(mapping)
        print(f"  Binary encode  '{col}': {mapping}")

    le_cols = ["race_ethnicity"]
    for col in le_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_map[col] = le
        print(f"  Label encode   '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    drop_cols = ["parental_level_of_education"]
    df.drop(columns=drop_cols, inplace=True)
    print(f"  Dropped original ordinal columns: {drop_cols}")

    return df, le_map


# ══════════════════════════════════════════════════════════════
# 6. MODEL TRAINING
# ══════════════════════════════════════════════════════════════
def train_model(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print("  STEP 6 — MODEL TRAINING")
    print(f"{'='*55}")

    FEATURES = [
        "gender", "race_ethnicity", "lunch", "test_preparation_course",
        "study_hours_per_day", "absences", "sleep_hours",
        "extracurricular_activities", "parental_edu_level",
        "study_sleep_ratio", "attendance_rate", "support_score", "study_x_prep",
    ]
    TARGET = "exam_score"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"  Features used: {FEATURES}")

    # Pipeline: imputer + scaler + linear regression
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LinearRegression()),
    ])
    pipe.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
    print(f"\n  5-Fold CV R² scores: {cv_scores.round(4)}")
    print(f"  Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Also train a Ridge regression for comparison
    pipe_ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=1.0)),
    ])
    pipe_ridge.fit(X_train, y_train)

    return pipe, pipe_ridge, X_train, X_test, y_train, y_test, FEATURES


# ══════════════════════════════════════════════════════════════
# 7. EVALUATION
# ══════════════════════════════════════════════════════════════
def evaluate(pipe, X_test, y_test, pipe_ridge=None, features=None) -> dict:
    print(f"\n{'='*55}")
    print("  STEP 7 — EVALUATION")
    print(f"{'='*55}")

    y_pred = pipe.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    mse    = mean_squared_error(y_test, y_pred)
    rmse   = np.sqrt(mse)
    r2     = r2_score(y_test, y_pred)

    print(f"\n  Linear Regression:")
    print(f"    MAE  = {mae:.3f}")
    print(f"    RMSE = {rmse:.3f}")
    print(f"    R²   = {r2:.4f}")

    if pipe_ridge:
        y_pred_r = pipe_ridge.predict(X_test)
        print(f"\n  Ridge Regression:")
        print(f"    MAE  = {mean_absolute_error(y_test, y_pred_r):.3f}")
        print(f"    RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_r)):.3f}")
        print(f"    R²   = {r2_score(y_test, y_pred_r):.4f}")

    # Feature importance (coefficients)
    if features:
        coefs = pipe.named_steps["model"].coef_
        feat_imp = pd.Series(coefs, index=features).sort_values(key=abs, ascending=False)
        print(f"\n  Feature Coefficients (sorted by |coef|):")
        print(feat_imp.round(4).to_string())

    _plot_results(y_test, y_pred, features,
                  pipe.named_steps["model"].coef_)

    return {"mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}


def _plot_results(y_test, y_pred, features, coefs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0f1117")
    GREY  = "#1e2130"
    ACCENT  = "#00d4ff"
    ACCENT2 = "#ff6b6b"
    TEXT    = "#e0e6f0"

    for ax in axes:
        ax.set_facecolor(GREY)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3250")
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)

    # Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.5, s=15, color=ACCENT, edgecolors="none")
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color=ACCENT2, linewidth=1.8, linestyle="--")
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Actual vs Predicted")
    r2 = r2_score(y_test, y_pred)
    ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes,
            color=ACCENT2, fontsize=10, fontweight="bold")

    # Residuals distribution
    ax = axes[1]
    residuals = y_test.values - y_pred
    ax.hist(residuals, bins=30, color=ACCENT, edgecolor="#0f1117", alpha=0.9)
    ax.axvline(0, color=ACCENT2, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residuals Distribution")

    # Feature coefficients
    ax = axes[2]
    if features and coefs is not None:
        feat_series = pd.Series(coefs, index=features).sort_values()
        colors_bar  = [ACCENT if v >= 0 else ACCENT2 for v in feat_series.values]
        ax.barh(feat_series.index, feat_series.values,
                color=colors_bar, alpha=0.85, edgecolor="#0f1117")
        ax.axvline(0, color=TEXT, linewidth=0.8, linestyle="--")
        ax.set_title("Feature Coefficients")
        ax.set_xlabel("Coefficient Value")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Model Evaluation Results",
                 fontsize=14, color=TEXT, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, "results.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Results plot saved → {out}")


# ══════════════════════════════════════════════════════════════
# 8. PERSIST MODEL
# ══════════════════════════════════════════════════════════════
def save_model(pipe, le_map: dict, features: list) -> str:
    payload = {"pipeline": pipe, "label_encoders": le_map, "features": features}
    path = os.path.join(MODEL_DIR, "student_model.pkl")
    joblib.dump(payload, path)
    print(f"\n  Model saved → {path}")
    return path


def load_model(path: str = None):
    if path is None:
        path = os.path.join(MODEL_DIR, "student_model.pkl")
    return joblib.load(path)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run_pipeline():
    df               = load_data()
    exploratory_analysis(df)
    df               = clean_data(df)
    df               = engineer_features(df)
    df, le_map       = encode_features(df)
    pipe, pipe_ridge, X_train, X_test, y_train, y_test, features = train_model(df)
    metrics          = evaluate(pipe, X_test, y_test, pipe_ridge, features)
    model_path       = save_model(pipe, le_map, features)

    print(f"\n{'='*55}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*55}")
    print(f"  R²   : {metrics['r2']:.4f}")
    print(f"  MAE  : {metrics['mae']:.3f}")
    print(f"  RMSE : {metrics['rmse']:.3f}")
    print(f"  Model: {model_path}")
    print(f"{'='*55}\n")
    return pipe, le_map, features


if __name__ == "__main__":
    run_pipeline()
