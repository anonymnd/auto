"""Same generalization question, third domain: does a model predicting
whether a STUDENT answers a question correctly generalize to a new
student, or is it partly recognizing who the student is?

Dataset: ASSISTments 2009-2010 skill-builder (Zhijie Liang's figshare
mirror of the classic knowledge-tracing dataset), 525,534 interactions.
Features are purely behavioral (no skill content): how long the first
response took, how many hints were used, how many attempts, and how many
times this student has practiced this skill before (opportunity count).
Target: whether the response was correct.
Run:  .\\.venv\\Scripts\\python.exe scripts\\16_assistments_leakage.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "assistments2009" / "skill_builder_data.csv"
OUT = ROOT / "outputs"

df = pd.read_csv(DATA, encoding="latin1", low_memory=False)
print(f"Raw interactions: {len(df)}")

cols = ["user_id", "correct", "attempt_count", "hint_count", "hint_total",
        "ms_first_response", "overlap_time", "opportunity", "bottom_hint"]
df = df[cols].dropna(subset=["correct", "attempt_count", "hint_count",
                              "ms_first_response", "opportunity"])
df = df[(df["ms_first_response"] > 0) & (df["ms_first_response"] < 300_000)]  # drop bad/placeholder timestamps
print(f"Usable interactions: {len(df)}, {df['user_id'].nunique()} students")

X = np.column_stack([
    np.log1p(df["ms_first_response"]),
    df["attempt_count"],
    df["hint_count"],
    df["hint_total"].fillna(0),
    np.log1p(df["opportunity"]),
    df["bottom_hint"].fillna(0),
])
y = df["correct"].astype(int).values
groups = df["user_id"].values

print(f"Feature matrix: {X.shape}, positive rate: {y.mean():.3f}")

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

random_scores = cross_val_score(clf, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="balanced_accuracy")
group_scores = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=groups, scoring="balanced_accuracy")

rng = np.random.RandomState(0)
y_shuffled = rng.permutation(y)
null_scores = cross_val_score(clf, X, y_shuffled, cv=GroupKFold(5), groups=groups, scoring="balanced_accuracy")

print("\n=== ASSISTments: predict correctness, random vs student-wise split ===")
print(f"Random split (leaky):   {random_scores.mean():.3f} +/- {random_scores.std():.3f}  {random_scores.round(3)}")
print(f"Student-wise split:     {group_scores.mean():.3f} +/- {group_scores.std():.3f}  {group_scores.round(3)}")
print(f"Null (shuffled, grp):   {null_scores.mean():.3f} +/- {null_scores.std():.3f}  {null_scores.round(3)}")
print(f"Gap (random - student-wise): {random_scores.mean() - group_scores.mean():.3f}")

pd.DataFrame({
    "setting": ["random_kfold"] * 5 + ["group_kfold"] * 5 + ["null_shuffled"] * 5,
    "fold": list(range(5)) * 3,
    "balanced_accuracy": list(random_scores) + list(group_scores) + list(null_scores),
}).to_csv(OUT / "16_assistments_leakage.csv", index=False)
print(f"Saved {OUT / '16_assistments_leakage.csv'}")
