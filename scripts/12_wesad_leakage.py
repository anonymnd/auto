"""Level 6: second modality. WESAD wrist sensors (Empatica E4), baseline vs
stress, same random-vs-subject-wise leakage question as the EEG posts.
Labels per WESAD readme: 1=baseline, 2=stress, 3=amusement, 4=meditation,
0/5/6/7=transient/ignore.
Run:  .\\.venv\\Scripts\\python.exe scripts\\12_wesad_leakage.py
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "WESAD"
OUT = ROOT / "outputs"

WRIST_FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}  # Hz, per Empatica E4
WINDOW_S = 30.0  # feature window length

def load_subject(subj_dir):
    pkl_path = subj_dir / f"{subj_dir.name}.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    wrist = data["signal"]["wrist"]
    labels = data["label"]  # sampled at chest rate, 700 Hz
    label_fs = 700
    return wrist, labels, label_fs

def window_features(wrist, labels, label_fs):
    """Cut into WINDOW_S windows, keep only windows with a single uniform
    label in {1 (baseline), 2 (stress)}, return (features, binary_label)."""
    n_label_samples = len(labels)
    window_label_samples = int(WINDOW_S * label_fs)
    X, y = [], []
    for start in range(0, n_label_samples - window_label_samples, window_label_samples):
        seg_labels = labels[start:start + window_label_samples]
        uniq = np.unique(seg_labels)
        if len(uniq) != 1 or uniq[0] not in (1, 2):
            continue
        label = 0 if uniq[0] == 1 else 1  # 0=baseline, 1=stress
        t0, t1 = start / label_fs, (start + window_label_samples) / label_fs
        feats = []
        for ch, fs in WRIST_FS.items():
            sig = wrist[ch]
            i0, i1 = int(t0 * fs), int(t1 * fs)
            seg = sig[i0:i1]
            if seg.ndim > 1:  # ACC has 3 axes
                seg = np.linalg.norm(seg, axis=1)
            seg = seg.flatten()
            if len(seg) == 0:
                feats.extend([0.0, 0.0])
                continue
            feats.append(float(np.mean(seg)))
            feats.append(float(np.std(seg)))
        X.append(feats)
        y.append(label)
    return np.array(X), np.array(y)

subject_dirs = sorted([p for p in DATA.glob("S*") if p.is_dir() and (p / f"{p.name}.pkl").exists()])
print(f"Found {len(subject_dirs)} subjects: {[p.name for p in subject_dirs]}")

all_X, all_y, all_groups = [], [], []
for i, sdir in enumerate(subject_dirs):
    wrist, labels, label_fs = load_subject(sdir)
    X, y = window_features(wrist, labels, label_fs)
    if len(y) == 0 or len(set(y)) < 2:
        print(f"...{sdir.name}: skip (no usable windows)")
        continue
    all_X.append(X)
    all_y.append(y)
    all_groups.append(np.full(len(y), i))
    print(f"...{sdir.name}: {len(y)} windows (baseline={np.sum(y==0)}, stress={np.sum(y==1)})")

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
groups = np.concatenate(all_groups, axis=0)
print(f"\nTotal: {X.shape[0]} windows, {X.shape[1]} features, {len(set(groups))} subjects")

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

random_scores = cross_val_score(clf, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="balanced_accuracy")
group_scores = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=groups, scoring="balanced_accuracy")

rng = np.random.RandomState(0)
y_shuffled = rng.permutation(y)
null_scores = cross_val_score(clf, X, y_shuffled, cv=GroupKFold(5), groups=groups, scoring="balanced_accuracy")

print("\n=== WESAD wrist (baseline vs stress): random vs subject-wise ===")
print(f"Random split (leaky):   {random_scores.mean():.3f} +/- {random_scores.std():.3f}  {random_scores.round(3)}")
print(f"Subject-wise split:     {group_scores.mean():.3f} +/- {group_scores.std():.3f}  {group_scores.round(3)}")
print(f"Null (shuffled, grp):   {null_scores.mean():.3f} +/- {null_scores.std():.3f}  {null_scores.round(3)}")
print(f"Gap (random - subject-wise): {random_scores.mean() - group_scores.mean():.3f}")

pd.DataFrame({
    "setting": ["random_kfold"] * 5 + ["group_kfold"] * 5 + ["null_shuffled"] * 5,
    "fold": list(range(5)) * 3,
    "balanced_accuracy": list(random_scores) + list(group_scores) + list(null_scores),
}).to_csv(OUT / "12_wesad_leakage.csv", index=False)
print(f"Saved {OUT / '12_wesad_leakage.csv'}")
