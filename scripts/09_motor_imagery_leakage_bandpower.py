"""Fix for 08: CSP pooled across subjects gave chance-level accuracy because
CSP spatial filters are subject-specific and washed out when fit on mixed
subjects. Replace CSP+LDA with log band-power features (same approach that
worked on EEGMAT) — no per-subject spatial filter needed, should survive
pooling and give an interpretable leakage gap.
Run:  .\\.venv\\Scripts\\python.exe scripts\\09_motor_imagery_leakage_bandpower.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

N_SUBJECTS = 10
RUNS = [4, 8, 12]
BANDS = {"theta": (4, 8), "alpha_mu": (8, 13), "beta": (13, 30), "low_gamma": (30, 40)}

def band_power_features(epoch_data, sfreq):
    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=min(256, epoch_data.shape[-1]), axis=-1)
    feats = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(np.log(psd[:, mask].mean(axis=1) + 1e-12))
    return np.concatenate(feats)

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
                         tmin=0.0, tmax=2.0, picks=picks, baseline=None, preload=True, verbose="ERROR")
    sfreq = raw.info["sfreq"]
    X = np.array([band_power_features(ep, sfreq) for ep in epochs.get_data(copy=True)])
    y = (epochs.events[:, -1] == event_id["T2"]).astype(int)  # 0=left, 1=right
    return X, y

all_X, all_y, all_groups = [], [], []
for s in range(1, N_SUBJECTS + 1):
    X, y = load_subject_epochs(s)
    all_X.append(X)
    all_y.append(y)
    all_groups.append(np.full(len(y), s))
    print(f"...subject {s}: {len(y)} epochs")

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
groups = np.concatenate(all_groups, axis=0)
print(f"\nTotal: {X.shape[0]} epochs, {X.shape[1]} features, {len(set(groups))} subjects")

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

random_scores = cross_val_score(clf, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="balanced_accuracy")
group_scores = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=groups, scoring="balanced_accuracy")

rng = np.random.RandomState(0)
y_shuffled = rng.permutation(y)
null_scores = cross_val_score(clf, X, y_shuffled, cv=GroupKFold(5), groups=groups, scoring="balanced_accuracy")

print("\n=== Motor imagery (band-power features): random vs subject-wise ===")
print(f"Random split (leaky):   {random_scores.mean():.3f} +/- {random_scores.std():.3f}  {random_scores.round(3)}")
print(f"Subject-wise split:     {group_scores.mean():.3f} +/- {group_scores.std():.3f}  {group_scores.round(3)}")
print(f"Null (shuffled, grp):   {null_scores.mean():.3f} +/- {null_scores.std():.3f}  {null_scores.round(3)}")
print(f"Gap (random - subject-wise): {random_scores.mean() - group_scores.mean():.3f}")

pd.DataFrame({
    "setting": ["random_kfold"] * 5 + ["group_kfold"] * 5 + ["null_shuffled"] * 5,
    "fold": list(range(5)) * 3,
    "balanced_accuracy": list(random_scores) + list(group_scores) + list(null_scores),
}).to_csv(OUT / "09_motor_imagery_leakage_bandpower.csv", index=False)
print(f"Saved {OUT / '09_motor_imagery_leakage_bandpower.csv'}")
