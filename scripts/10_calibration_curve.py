"""Level 5: calibration curve. Leave-subject-out baseline, then add a few
of the held-out subject's own trials to training and see how much of the
cross-subject gap closes. Reuses the 10 subjects already downloaded.
Run:  .\\.venv\\Scripts\\python.exe scripts\\10_calibration_curve.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

N_SUBJECTS = 10
RUNS = [4, 8, 12]
BANDS = {"theta": (4, 8), "alpha_mu": (8, 13), "beta": (13, 30), "low_gamma": (30, 40)}
CALIBRATION_SIZES = [0, 2, 5, 10, 20]
N_EVAL = 20  # fixed held-out-subject epochs never used for calibration, per subject

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
    y = (epochs.events[:, -1] == event_id["T2"]).astype(int)
    return X, y

print("Loading all subjects (cached locally, should be fast)...")
per_subject = {}
for s in range(1, N_SUBJECTS + 1):
    per_subject[s] = load_subject_epochs(s)
    print(f"...subject {s}: {len(per_subject[s][1])} epochs")

results = []
rng = np.random.RandomState(0)
for test_s in range(1, N_SUBJECTS + 1):
    X_test_all, y_test_all = per_subject[test_s]
    n = len(y_test_all)
    if n < N_EVAL + max(CALIBRATION_SIZES):
        print(f"skip subject {test_s}: only {n} epochs, need {N_EVAL + max(CALIBRATION_SIZES)}")
        continue
    order = rng.permutation(n)
    eval_idx = order[:N_EVAL]
    pool_idx = order[N_EVAL:]
    X_eval, y_eval = X_test_all[eval_idx], y_test_all[eval_idx]

    X_train_base = np.concatenate([per_subject[s][0] for s in per_subject if s != test_s], axis=0)
    y_train_base = np.concatenate([per_subject[s][1] for s in per_subject if s != test_s], axis=0)

    for k in CALIBRATION_SIZES:
        cal_idx = pool_idx[:k]
        if k > 0:
            X_train = np.concatenate([X_train_base, X_test_all[cal_idx]], axis=0)
            y_train = np.concatenate([y_train_base, y_test_all[cal_idx]], axis=0)
        else:
            X_train, y_train = X_train_base, y_train_base
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(X_train, y_train)
        acc = balanced_accuracy_score(y_eval, clf.predict(X_eval))
        results.append({"test_subject": test_s, "calibration_trials": k, "balanced_accuracy": acc})
    print(f"...subject {test_s} done")

df = pd.DataFrame(results)
df.to_csv(OUT / "10_calibration_curve.csv", index=False)

print("\n=== Calibration curve (mean balanced accuracy across held-out subjects) ===")
summary = df.groupby("calibration_trials")["balanced_accuracy"].agg(["mean", "std", "count"])
print(summary)
print(f"\nSaved {OUT / '10_calibration_curve.csv'}")
