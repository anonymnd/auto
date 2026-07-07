"""Level 4: the core leakage experiment, on the primary dataset.
Random split vs subject-wise split, CSP+LDA, left vs right fist imagery,
pooled across N_SUBJECTS subjects (runs 4/8/12 each).
Run:  .\\.venv\\Scripts\\python.exe scripts\\08_motor_imagery_leakage.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold, GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

N_SUBJECTS = 10
RUNS = [4, 8, 12]

def load_subject_epochs(subject):
    paths = mne.datasets.eegbci.load_data(subject, RUNS, path=str(DATA), update_path=True, verbose="ERROR")
    raws = [mne.io.read_raw_edf(p, preload=True, verbose="ERROR") for p in paths]
    raw = mne.concatenate_raws(raws, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)
    raw.filter(1.0, 40.0, verbose="ERROR")
    raw.notch_filter([60.0], verbose="ERROR")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    epochs = mne.Epochs(raw, events, event_id={"left": event_id["T1"], "right": event_id["T2"]},
                         tmin=-0.5, tmax=2.5, picks=picks, baseline=None, preload=True, verbose="ERROR")
    return epochs.get_data(copy=True), epochs.events[:, -1]

all_X, all_y, all_groups = [], [], []
skipped = []
for s in range(1, N_SUBJECTS + 1):
    try:
        X, y = load_subject_epochs(s)
    except Exception as e:
        skipped.append((s, str(e)))
        continue
    if X.shape[0] < 2 or len(set(y)) < 2:
        skipped.append((s, "too few epochs or single class"))
        continue
    all_X.append(X)
    all_y.append(y)
    all_groups.append(np.full(X.shape[0], s))
    print(f"...subject {s}: {X.shape[0]} epochs")

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
groups = np.concatenate(all_groups, axis=0)
print(f"\nTotal: {X.shape[0]} epochs from {len(all_X)} subjects (skipped: {skipped})")

clf = Pipeline([
    ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("lda", LinearDiscriminantAnalysis()),
])

random_scores = cross_val_score(clf, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="balanced_accuracy")
group_scores = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=groups, scoring="balanced_accuracy")

rng = np.random.RandomState(0)
y_shuffled = rng.permutation(y)
null_scores = cross_val_score(clf, X, y_shuffled, cv=GroupKFold(5), groups=groups, scoring="balanced_accuracy")

print("\n=== Motor imagery: random vs subject-wise split (balanced accuracy) ===")
print(f"Random split (leaky):   {random_scores.mean():.3f} +/- {random_scores.std():.3f}  {random_scores.round(3)}")
print(f"Subject-wise split:     {group_scores.mean():.3f} +/- {group_scores.std():.3f}  {group_scores.round(3)}")
print(f"Null (shuffled, grp):   {null_scores.mean():.3f} +/- {null_scores.std():.3f}  {null_scores.round(3)}")
print(f"Gap (random - subject-wise): {random_scores.mean() - group_scores.mean():.3f}")

pd.DataFrame({
    "setting": ["random_kfold"] * 5 + ["group_kfold"] * 5 + ["null_shuffled"] * 5,
    "fold": list(range(5)) * 3,
    "balanced_accuracy": list(random_scores) + list(group_scores) + list(null_scores),
}).to_csv(OUT / "08_motor_imagery_leakage.csv", index=False)
print(f"Saved {OUT / '08_motor_imagery_leakage.csv'}")
