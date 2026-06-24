# 🎓 Student Performance Prediction
> End-to-End Machine Learning project that predicts student exam scores using **Linear Regression** and an interactive **Streamlit** web app.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)

---

## 🚀 Features

- 📊 Data Cleaning & EDA
- ⚙️ Feature Engineering
- 🤖 Linear & Ridge Regression
- 🔄 Scikit-Learn Pipeline
- 📈 Model Evaluation
- 🌐 Streamlit Dashboard

---

## 📁 Project Structure

```
student_performance/
├── app.py                        # Streamlit web app
├── requirements.txt
├── README.md
├── data/
│   ├── generate_dataset.py       # Generates students.csv
│   └── students.csv              # 1,000 synthetic student records
├── models/
│   └── student_model.pkl         # Saved sklearn Pipeline
├── notebooks/
│   └── analysis.ipynb            # Step-by-step Jupyter notebook
├── plots/
│   ├── eda.png                   # EDA visualizations
│   └── results.png               # Model evaluation charts
└── src/
    └── pipeline.py               # Full ML pipeline (runnable script)
```

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
cd student_performance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python data/generate_dataset.py

# 4. Train the model (also produces plots)
python src/pipeline.py

# 5. Launch the Streamlit app
streamlit run app.py
```

---

## 🧠 Concepts Covered

### Data Cleaning
- Missing value imputation (median for numeric, mode for categorical)
- Outlier capping using IQR method
- Duplicate removal

### Label Encoding
- Binary mapping for gender, lunch, test_prep, extracurricular
- `sklearn.LabelEncoder` for race/ethnicity groups
- Ordinal mapping for parental education (0–5)

### Feature Engineering
| Feature | Description |
|---|---|
| `study_sleep_ratio` | study_hours / sleep_hours |
| `attendance_rate` | 1 − absences / max_absences |
| `parental_edu_level` | Ordinal 0–5 from education level |
| `support_score` | lunch + test_prep + extracurricular (0–3) |
| `study_x_prep` | Interaction: study_hours × test_prep |

### Linear Regression
- `score = β₀ + β₁·study_hours + β₂·absences + ... + ε`
- Trained inside a `sklearn.Pipeline` (Imputer → Scaler → LinearRegression)
- Compared with Ridge Regression (L2 regularization)
- 5-fold cross-validation for robust evaluation

---


## 📊 Skills Used

| Library | Usage |
|---|---|
| **Pandas** | Data loading, cleaning, manipulation |
| **NumPy** | Numerical operations, array handling |
| **Matplotlib** | EDA plots, evaluation charts |
| **Scikit-learn** | Encoding, pipeline, regression, metrics |
| **Streamlit** | Interactive web deployment |
| **Joblib** | Model serialization (.pkl) |

---

## 🤖 ML Workflow

```
Dataset
   │
   ▼
Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Encoding
   │
   ▼
Model Training
   │
   ▼
Evaluation
   │
   ▼
Streamlit App
```

---

## 📏 Model Performance

| Metric | Value |
|---|---|
| R² (test) | ~0.60 |
| MAE | ~6 pts |
| RMSE | ~7.4 pts |
| 5-fold CV R² | ~0.60 ± 0.03 |

---

## 🌐 Streamlit App Features

- **Predict tab**: Sidebar inputs → instant score prediction with grade badge
- **Feature impact chart**: Bar chart showing which factors push the score up/down
- **Personalized tips**: Actionable improvement suggestions
- **Data Insights tab**: EDA and model evaluation plots
- **How It Works tab**: Project architecture and methodology

---


## 🚀 Future Improvements

- XGBoost
- Random Forest
- SHAP Explainability
- Docker Deployment

---

## 👩‍💻 Author

**Bhoomi Meena**

B.Tech CSE (Artificial Intelligence/Machine Learning)

---

*Built with Python · Pandas · NumPy · Matplotlib · Scikit-learn · Streamlit*
