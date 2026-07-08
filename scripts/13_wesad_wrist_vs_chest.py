"""Wrist vs chest: how much is lost with only a wrist device? Same baseline
vs stress task, same subject-wise evaluation, chest (RespiBAN, 700 Hz,
richer sensor set) vs wrist (Empatica E4, consumer-grade) features.
Run:  .\\.venv\\Scripts\\python.exe scripts\\13_wesad_wrist_vs_chest.py
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "WESAD"
OUT = ROOT / "outputs"

WRIST_FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
CHEST_FS = {"ECG": 700, "EDA": 700, "EMG": 700, "Resp": 700, "Temp": 700, "ACC": 700}
WINDOW_S = 30.0
LABEL_FS = 700

def load_subject(subj_dir):
    with open(subj_dir / f"{subj_dir.name}.pkl", "rb") as f:
        return pickle.load(f, encoding="latin1")

def window_features(sensors, fs_map, labels, label_fs):
    n = len(labels)
    win = int(WINDOW_S * label_fs)
    X, y = [], []
    for start in range(0, n - win, win):
        seg_labels = labels[start:start + win]
        uniq = np.unique(seg_labels)
        if len(uniq) != 1 or uniq[0] not in (1, 2):
            continue
        label = 0 if uniq[0] == 1 else 1
        t0, t1 = start / label_fs, (start + win) / label_fs
        feats = []
        for ch, fs in fs_map.items():
            sig = sensors[ch]
            i0, i1 = int(t0 * fs), int(t1 * fs)
            seg = sig[i0:i1]
            if seg.ndim > 1:
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
print(f"Found {len(subject_dirs)} subjects")

results = {}
for device, fs_map in (("wrist", WRIST_FS), ("chest", CHEST_FS)):
    all_X, all_y, all_groups = [], [], []
    for i, sdir in enumerate(subject_dirs):
        d = load_subject(sdir)
        sensors = d["signal"][device]
        labels = d["label"]
        X, y = window_features(sensors, fs_map, labels, LABEL_FS)
        if len(y) == 0 or len(set(y)) < 2:
            continue
        all_X.append(X)
        all_y.append(y)
        all_groups.append(np.full(len(y), i))
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    group_scores = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=groups, scoring="balanced_accuracy")
    rng = np.random.RandomState(0)
    null_scores = cross_val_score(clf, X, rng.permutation(y), cv=GroupKFold(5), groups=groups, scoring="balanced_accuracy")

    results[device] = {"subject_wise_mean": group_scores.mean(), "subject_wise_std": group_scores.std(),
                        "null_mean": null_scores.mean(), "n_features": X.shape[1], "n_windows": X.shape[0]}
    print(f"\n=== {device} (n_features={X.shape[1]}, n_windows={X.shape[0]}) ===")
    print(f"Subject-wise: {group_scores.mean():.3f} +/- {group_scores.std():.3f}  {group_scores.round(3)}")
    print(f"Null:         {null_scores.mean():.3f} +/- {null_scores.std():.3f}")

print(f"\n=== Wrist vs chest gap (subject-wise balanced accuracy) ===")
print(f"Chest: {results['chest']['subject_wise_mean']:.3f}  Wrist: {results['wrist']['subject_wise_mean']:.3f}")
print(f"Cost of wrist-only: {results['chest']['subject_wise_mean'] - results['wrist']['subject_wise_mean']:.3f}")

pd.DataFrame(results).T.to_csv(OUT / "13_wesad_wrist_vs_chest.csv")
print(f"\nSaved {OUT / '13_wesad_wrist_vs_chest.csv'}")
