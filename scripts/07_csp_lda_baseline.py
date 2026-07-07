"""Level 3: boring baseline. CSP + LDA on motor imagery, WITHIN one subject
(runs 4, 8, 12 = imagined left/right fist). No cross-subject claims yet.
Run:  .\\.venv\\Scripts\\python.exe scripts\\07_csp_lda_baseline.py
"""
from pathlib import Path
import numpy as np
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SUBJECT = 1
RUNS = [4, 8, 12]  # imagined fist movement runs

paths = mne.datasets.eegbci.load_data(SUBJECT, RUNS, path=str(DATA), update_path=True)
raws = [mne.io.read_raw_edf(p, preload=True, verbose="ERROR") for p in paths]
raw = mne.concatenate_raws(raws, verbose="ERROR")
mne.datasets.eegbci.standardize(raw)
raw.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")

raw.filter(1.0, 40.0, verbose="ERROR")
raw.notch_filter([60.0], verbose="ERROR")

events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

# T1=left fist, T2=right fist; drop T0 (rest) for this binary baseline
epochs = mne.Epochs(raw, events, event_id={"left": event_id["T1"], "right": event_id["T2"]},
                     tmin=-0.5, tmax=2.5, picks=picks, baseline=None, preload=True, verbose="ERROR")
X = epochs.get_data(copy=True)
y = epochs.events[:, -1]
print(f"Epochs: {len(y)}  (left={np.sum(y==event_id['T1'])}, right={np.sum(y==event_id['T2'])})")

clf = Pipeline([
    ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("lda", LinearDiscriminantAnalysis()),
])

cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
print(f"\n=== Within-subject CSP+LDA (subject {SUBJECT}, 5-fold) ===")
print(f"Balanced accuracy: {scores.mean():.3f} +/- {scores.std():.3f}  {scores.round(3)}")

import pandas as pd
pd.DataFrame({"fold": range(5), "balanced_accuracy": scores}).to_csv(
    OUT / "07_csp_lda_baseline.csv", index=False)
print(f"Saved {OUT / '07_csp_lda_baseline.csv'}")
