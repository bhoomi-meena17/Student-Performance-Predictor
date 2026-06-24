"""
app.py — Streamlit deployment for Student Performance Predictor
Run: streamlit run app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ─── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data",   "students.csv")

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── main background ── */
    .stApp { background-color: #0f1117; }
    section[data-testid="stSidebar"] { background-color: #14161f; border-right: 1px solid #2d3250; }

    /* ── typography ── */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; color: #e0e6f0; }
    h1 { color: #00d4ff !important; font-weight: 800; font-size: 2.2rem !important; }
    h2 { color: #a0b4d0 !important; font-weight: 700; font-size: 1.3rem !important; }
    h3 { color: #00d4ff !important; }

    /* ── metric cards ── */
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 18px 24px;
    }
    [data-testid="metric-container"] label { color: #8899bb !important; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 2rem !important; font-weight: 800; }

    /* ── predict button ── */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0077ff);
        color: #0f1117;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.65rem 2rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* ── score gauge ── */
    .score-box {
        background: linear-gradient(135deg, #1e2130, #14161f);
        border: 2px solid #00d4ff;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 12px 0;
    }
    .score-value { font-size: 4rem; font-weight: 900; color: #00d4ff; line-height: 1; }
    .score-label { font-size: 0.9rem; color: #8899bb; margin-top: 6px; }
    .grade-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 10px;
    }

    /* ── section divider ── */
    hr { border: none; border-top: 1px solid #2d3250; margin: 1.2rem 0; }

    /* ── expander / info boxes ── */
    .stExpander { background: #1e2130 !important; border: 1px solid #2d3250 !important; border-radius: 10px !important; }
    [data-testid="stInfo"] { background: #1a2540; border-color: #00d4ff; }
</style>
""", unsafe_allow_html=True)


# ─── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please run `python src/pipeline.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def grade_from_score(score: float) -> tuple[str, str]:
    if score >= 90:  return "A+", "#00d4ff"
    if score >= 80:  return "A",  "#34d399"
    if score >= 70:  return "B",  "#a3e635"
    if score >= 60:  return "C",  "#fbbf24"
    if score >= 50:  return "D",  "#fb923c"
    return "F", "#f87171"


def build_feature_vector(inputs: dict, features: list) -> pd.DataFrame:
    """Map UI inputs → model feature vector."""
    edu_map = {
        "Some High School": 0, "High School": 1, "Some College": 2,
        "Associate's Degree": 3, "Bachelor's Degree": 4, "Master's Degree": 5,
    }
    eth_map = {"Group A": 0, "Group B": 1, "Group C": 2, "Group D": 3, "Group E": 4}
    max_absence_days = 20  # same as training

    study   = inputs["study_hours"]
    sleep   = inputs["sleep_hours"]
    absences= inputs["absences"]

    row = {
        "gender":                    0 if inputs["gender"] == "Male" else 1,
        "race_ethnicity":            eth_map[inputs["ethnicity"]],
        "lunch":                     1 if inputs["lunch"] == "Standard" else 0,
        "test_preparation_course":   1 if inputs["test_prep"] == "Completed" else 0,
        "study_hours_per_day":       study,
        "absences":                  absences,
        "sleep_hours":               sleep,
        "extracurricular_activities":1 if inputs["extracurricular"] == "Yes" else 0,
        "parental_edu_level":        edu_map[inputs["parental_edu"]],
        "study_sleep_ratio":         study / (sleep + 1e-6),
        "attendance_rate":           1 - absences / (max_absence_days + 1),
        "support_score":             (
            (inputs["lunch"]       == "Standard")  +
            (inputs["test_prep"]   == "Completed") +
            (inputs["extracurricular"] == "Yes")
        ),
        "study_x_prep":              study * (1 if inputs["test_prep"] == "Completed" else 0),
    }
    return pd.DataFrame([row])[features]


def plot_feature_impact(inputs: dict, pipeline, features: list) -> plt.Figure:
    """Show which factors push the prediction up or down."""
    coefs  = pipeline.named_steps["model"].coef_
    scaler = pipeline.named_steps["scaler"]

    row_df = build_feature_vector(inputs, features)
    # impute then scale
    from sklearn.impute import SimpleImputer
    imp = pipeline.named_steps.get("imputer", SimpleImputer(strategy="median"))
    try:
        row_imp = imp.transform(row_df)
    except Exception:
        row_imp = row_df.values

    row_scaled = scaler.transform(row_imp)
    contributions = coefs * row_scaled[0]

    contrib_df = pd.Series(contributions, index=features).sort_values()
    pos_mask   = contrib_df >= 0
    colors      = ["#34d399" if p else "#f87171" for p in pos_mask]

    nice_names = {
        "study_hours_per_day": "Study Hours",
        "parental_edu_level":  "Parental Education",
        "attendance_rate":     "Attendance Rate",
        "test_preparation_course": "Test Prep Course",
        "lunch":               "Lunch Type",
        "support_score":       "Support Score",
        "study_x_prep":        "Study × Test Prep",
        "sleep_hours":         "Sleep Hours",
        "absences":            "Absences",
        "study_sleep_ratio":   "Study/Sleep Ratio",
        "gender":              "Gender",
        "race_ethnicity":      "Ethnicity Group",
        "extracurricular_activities": "Extracurricular",
    }
    idx_nice = [nice_names.get(f, f) for f in contrib_df.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#1e2130")
    ax.barh(idx_nice, contrib_df.values, color=colors, alpha=0.9, edgecolor="#0f1117")
    ax.axvline(0, color="#8899bb", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Score Contribution", color="#a0b4d0", fontsize=9)
    ax.set_title("What's Driving Your Score?", color="#e0e6f0", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#a0b4d0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3250")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
payload  = load_model()
pipeline = payload["pipeline"]
features = payload["features"]
df_raw   = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎓 Student Exam Score Predictor")
st.markdown("*Predict academic performance using Linear Regression — an internship-level ML project.*")
st.markdown("---")

# ── Sidebar: inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Student Profile")
    st.markdown("Fill in the details below:")
    st.markdown("---")

    gender     = st.selectbox("Gender", ["Male", "Female"])
    ethnicity  = st.selectbox("Race / Ethnicity", ["Group A","Group B","Group C","Group D","Group E"])
    parental_edu = st.selectbox("Parental Education", [
        "Some High School","High School","Some College",
        "Associate's Degree","Bachelor's Degree","Master's Degree"
    ])

    st.markdown("---")
    lunch      = st.radio("Lunch Type", ["Standard", "Free/Reduced"], horizontal=True)
    test_prep  = st.radio("Test Prep Course", ["None", "Completed"], horizontal=True)
    extra      = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)

    st.markdown("---")
    study  = st.slider("📚 Study Hours / Day", 0.0, 12.0, 5.0, 0.5)
    sleep  = st.slider("😴 Sleep Hours / Day", 4.0, 10.0, 7.0, 0.5)
    absent = st.slider("❌ Number of Absences", 0, 20, 3)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Score")


# ── Main area: tabs ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Data Insights", "📖 How It Works"])


# ─── TAB 1: Prediction ────────────────────────────────────────────────────────
with tab1:
    if predict_btn:
        inputs = {
            "gender": gender, "ethnicity": ethnicity,
            "parental_edu": parental_edu, "lunch": lunch,
            "test_prep": test_prep, "extracurricular": extra,
            "study_hours": study, "sleep_hours": sleep, "absences": absent,
        }
        row     = build_feature_vector(inputs, features)
        score   = float(np.clip(pipeline.predict(row)[0], 0, 100))
        grade, grade_color = grade_from_score(score)

        col_left, col_right = st.columns([1, 1.5])

        with col_left:
            st.markdown(f"""
            <div class="score-box">
                <div class="score-value">{score:.1f}</div>
                <div class="score-label">Predicted Exam Score / 100</div>
                <div class="grade-badge" style="background:{grade_color}22; color:{grade_color}; border:1.5px solid {grade_color};">
                    Grade {grade}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Study Hours", f"{study}h")
            m2.metric("Absences",    f"{absent}")
            m3.metric("Sleep",       f"{sleep}h")

            # Qualitative feedback
            tips = []
            if study < 4:   tips.append("⚠️ Increase daily study time.")
            if absent > 7:  tips.append("⚠️ Reduce absences significantly.")
            if test_prep == "None": tips.append("💡 Consider completing a test-prep course (+8 pts avg).")
            if sleep < 6:   tips.append("💤 Getting more sleep can improve focus.")
            if lunch == "Free/Reduced": tips.append("📌 Students with standard lunch score ~5 pts higher on avg.")
            if not tips:    tips.append("✅ Great profile! Keep it up.")

            with st.expander("💬 Personalized Tips", expanded=True):
                for t in tips:
                    st.write(t)

        with col_right:
            fig = plot_feature_impact(inputs, pipeline, features)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    else:
        st.info("👈 Fill in the student profile on the left, then click **Predict Score**.")
        st.markdown("""
        #### What this app predicts
        This tool uses a trained **Linear Regression** model to predict a student's exam score
        (0–100) based on demographic, behavioral, and academic factors.

        **Features used:**
        - Study hours per day, sleep hours, absences
        - Test preparation course completion
        - Parental education level
        - Lunch type (proxy for socioeconomic status)
        - Extracurricular participation
        - Engineered features: study/sleep ratio, attendance rate, support score
        """)


# ─── TAB 2: Data Insights ─────────────────────────────────────────────────────
with tab2:
    if df_raw is None:
        st.warning("Dataset not found at `data/students.csv`.")
    else:
        st.markdown("### 🗂️ Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students", f"{len(df_raw):,}")
        c2.metric("Average Score",  f"{df_raw['exam_score'].mean():.1f}")
        c3.metric("Highest Score",  f"{df_raw['exam_score'].max():.1f}")
        c4.metric("Missing Values", f"{df_raw.isnull().sum().sum()}")

        st.markdown("---")
        with st.expander("📋 Raw Data Sample", expanded=False):
            st.dataframe(df_raw.head(20), use_container_width=True)

        st.markdown("### 📊 Visual Analysis")
        img_path = os.path.join(BASE_DIR, "plots", "eda.png")
        res_path = os.path.join(BASE_DIR, "plots", "results.png")

        if os.path.exists(img_path):
            st.image(img_path, caption="Exploratory Data Analysis", use_column_width=True)
        if os.path.exists(res_path):
            st.image(res_path, caption="Model Evaluation Results", use_column_width=True)

        if not os.path.exists(img_path):
            st.info("Run `python src/pipeline.py` to generate the EDA plots.")


# ─── TAB 3: How It Works ──────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    ### 🔬 Project Architecture

    ```
    student_performance/
    ├── data/
    │   ├── generate_dataset.py   # Synthetic dataset generator
    │   └── students.csv          # 1,000 student records
    ├── models/
    │   └── student_model.pkl     # Trained pipeline (joblib)
    ├── notebooks/
    │   └── analysis.ipynb        # Step-by-step notebook
    ├── plots/
    │   ├── eda.png               # EDA visualizations
    │   └── results.png           # Model evaluation plots
    ├── src/
    │   └── pipeline.py           # Full ML pipeline
    └── app.py                    # This Streamlit app
    ```

    ### 🧠 ML Pipeline Steps

    | Step | Description |
    |------|------------|
    | **1. Data Loading** | Load CSV with Pandas |
    | **2. EDA** | Distributions, correlations, box plots |
    | **3. Data Cleaning** | Null imputation (median/mode), outlier capping (IQR), deduplication |
    | **4. Feature Engineering** | study_sleep_ratio, attendance_rate, support_score, study_x_prep, parental_edu_level |
    | **5. Label Encoding** | Binary maps + sklearn LabelEncoder |
    | **6. Model Training** | StandardScaler → Linear Regression inside sklearn Pipeline |
    | **7. Evaluation** | MAE, RMSE, R², 5-fold cross-validation |
    | **8. Persistence** | joblib.dump → .pkl |

    ### 📐 Key Concepts Covered
    - **Linear Regression**: `score = w₀ + w₁×study_hours + w₂×absences + ...`
    - **Label Encoding**: Convert categorical strings to integers
    - **Feature Engineering**: Create new informative columns from raw data
    - **Cross-Validation**: 5-fold CV for robust performance estimates
    - **Ridge Regression**: L2 regularisation for comparison
    """)

    # Show model metrics
    st.markdown("### 📏 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model",  "Linear Regression")
    col2.metric("CV R²",  "~0.60")
    col3.metric("Avg MAE","~6 pts")

    st.info("""
    **Tech Stack:** Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit · Joblib
    """)
