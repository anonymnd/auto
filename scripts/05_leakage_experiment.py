"""Core warm-up result: does accuracy change between a random split and a
subject-wise split, on the same EEGMAT rest-vs-arithmetic data?

Pipeline:
  1. Load all 36 subjects (rest + arithmetic EDFs, already downloaded).
  2. Cut each trial into non-overlapping 10 s epochs (shortest trial is
     35 s, so every subject contributes >=3 epochs per condition).
  3. Feature = log band power (delta/theta/alpha/beta/gamma) per EEG
     channel (20 ch, ECG dropped) -> 100 features per epoch.
  4. Classifier = LogisticRegression, scaler fit INSIDE each fold only.
  5. Compare 3 settings: random KFold (leaky), GroupKFold by subject
     (correct), and a label-shuffled null model under the correct split.

Run:  .\\.venv\\Scripts\\python.exe scripts\\05_leakage_experiment.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eegmat"
OUT = ROOT / "outputs"

N_SUBJECTS = 36
EPOCH_S = 10.0
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}

def band_power_features(epoch_data, sfreq):
    """epoch_data: (n_channels, n_samples) -> (n_channels * n_bands,) log power."""
    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=min(256, epoch_data.shape[-1]), axis=-1)
    feats = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        band_power = psd[:, mask].mean(axis=1)
        feats.append(np.log(band_power + 1e-12))
    return np.concatenate(feats)

X, y, groups = [], [], []
for i in range(N_SUBJECTS):
    subj = f"Subject{i:02d}"
    for cond, suffix, label in (("rest", "_1.edf", 0), ("arithmetic", "_2.edf", 1)):
        path = DATA / f"{subj}{suffix}"
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        raw.pick([ch for ch in raw.ch_names if ch.startswith("EEG")])  # drop ECG
        sfreq = raw.info["sfreq"]
        data = raw.get_data()
        n_samples_per_epoch = int(EPOCH_S * sfreq)
        n_epochs = data.shape[1] // n_samples_per_epoch
        for e in range(n_epochs):
            seg = data[:, e * n_samples_per_epoch:(e + 1) * n_samples_per_epoch]
            X.append(band_power_features(seg, sfreq))
            y.append(label)
            groups.append(i)
    if i % 6 == 0:
        print(f"...featurized {i}/{N_SUBJECTS}")

X = np.array(X)
y = np.array(y)
groups = np.array(groups)
print(f"\nTotal epochs: {len(y)}  (rest={np.sum(y==0)}, arithmetic={np.sum(y==1)})")
print(f"Feature matrix: {X.shape}")

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

SCORING = "balanced_accuracy"
majority_baseline = max(np.mean(y == 0), np.mean(y == 1))

random_scores = cross_val_score(clf, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring=SCORING)
group_scores = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=groups, scoring=SCORING)

rng = np.random.RandomState(0)
y_shuffled = rng.permutation(y)
null_scores = cross_val_score(clf, X, y_shuffled, cv=GroupKFold(5), groups=groups, scoring=SCORING)

print(f"\nClass balance: rest={np.mean(y==0):.3f}, arithmetic={np.mean(y==1):.3f}  (majority baseline, balanced_accuracy=0.500 by definition)")
print("\n=== Results (5-fold balanced accuracy) ===")
print(f"Random split (leaky):       {random_scores.mean():.3f} +/- {random_scores.std():.3f}  {random_scores.round(3)}")
print(f"Subject-wise split:         {group_scores.mean():.3f} +/- {group_scores.std():.3f}  {group_scores.round(3)}")
print(f"Null (shuffled labels, grp): {null_scores.mean():.3f} +/- {null_scores.std():.3f}  {null_scores.round(3)}")
print(f"\nGap (random - subject-wise): {random_scores.mean() - group_scores.mean():.3f}")

pd.DataFrame({
    "setting": ["random_kfold"] * 5 + ["group_kfold"] * 5 + ["null_shuffled"] * 5,
    "fold": list(range(5)) * 3,
    "accuracy": list(random_scores) + list(group_scores) + list(null_scores),
}).to_csv(OUT / "05_leakage_results.csv", index=False)
print(f"\nSaved {OUT / '05_leakage_results.csv'}")
