"""
Script to generate a realistic synthetic student performance dataset.
Run this once to create students.csv before training the model.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

genders = np.random.choice(["Male", "Female"], N)
ethnicities = np.random.choice(["Group A", "Group B", "Group C", "Group D", "Group E"], N)
parental_edu = np.random.choice(
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"],
    N,
    p=[0.08, 0.20, 0.22, 0.20, 0.20, 0.10],
)
lunch = np.random.choice(["standard", "free/reduced"], N, p=[0.65, 0.35])
test_prep = np.random.choice(["none", "completed"], N, p=[0.60, 0.40])
study_hours = np.clip(np.random.normal(5, 2.5, N), 0, 12).round(1)
absences = np.clip(np.random.poisson(3, N), 0, 20)
sleep_hours = np.clip(np.random.normal(7, 1.2, N), 4, 10).round(1)
extracurricular = np.random.choice(["Yes", "No"], N, p=[0.45, 0.55])

# Base score with realistic correlations
edu_map = {
    "some high school": 0, "high school": 1, "some college": 2,
    "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5,
}
edu_score = np.array([edu_map[e] for e in parental_edu])

base = (
    50
    + edu_score * 2.5
    + (lunch == "standard") * 5
    + (test_prep == "completed") * 8
    + study_hours * 2.8
    - absences * 1.5
    + (sleep_hours - 7) * 1.2
    + (extracurricular == "Yes") * 2
    + np.random.normal(0, 7, N)
)
exam_score = np.clip(base, 0, 100).round(1)

df = pd.DataFrame({
    "gender": genders,
    "race_ethnicity": ethnicities,
    "parental_level_of_education": parental_edu,
    "lunch": lunch,
    "test_preparation_course": test_prep,
    "study_hours_per_day": study_hours,
    "absences": absences,
    "sleep_hours": sleep_hours,
    "extracurricular_activities": extracurricular,
    "exam_score": exam_score,
})

# Inject ~5% missing values for data cleaning practice
for col in ["study_hours_per_day", "absences", "sleep_hours", "parental_level_of_education"]:
    mask = np.random.rand(N) < 0.05
    df.loc[mask, col] = np.nan

df.to_csv("students.csv", index=False)
print(f"Dataset saved: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head())
