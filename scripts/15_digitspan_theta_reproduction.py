"""Level 7: reproduction attempt. Pavlov et al. 2022 report frontal
midline theta (~6 Hz) increasing in the memory condition vs. the
listen-only control condition (their Fig. 3d) -- no numbers given in the
paper text, only the qualitative claim and a figure. This script computes
frontal theta (4-8 Hz) power at channel Fz during the 3s window right
after the last digit of each trial (the retention interval before recall),
memory vs. control, per subject.

Reads the Fz channel directly via h5py (EEGLAB v7.3/HDF5 .set files) rather
than loading the full 63-channel matrix through MNE/pymatreader -- the full
load needs ~3.5 GB per subject and this machine doesn't reliably have that
much free RAM; reading one channel lazily needs tens of MB.
Run:  .\\.venv\\Scripts\\python.exe scripts\\15_digitspan_theta_reproduction.py
"""
from pathlib import Path
import re
import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ds003838"
OUT = ROOT / "outputs"

SUBJECTS = ["sub-032", "sub-033", "sub-034"]
THETA_BAND = (4, 8)
WINDOW_S = 3.0
DIGIT_DURATION = 1.0

def read_str(f, ref):
    arr = f[ref][()]
    return "".join(chr(int(c[0])) for c in arr)

def get_fz_index(f):
    labels_refs = f["chanlocs"]["labels"][()]
    n = labels_refs.shape[0]
    for i in range(n):
        ref = labels_refs[i][0] if labels_refs.ndim == 2 else labels_refs[i]
        if read_str(f, ref).lower() == "fz":
            return i
    raise ValueError("Fz channel not found")

def last_digit_events(events_df):
    trials = []
    for _, row in events_df.iterrows():
        tt = row["trial_type"]
        if not isinstance(tt, str) or "digit" not in tt:
            continue
        cond = "memory" if tt.startswith("memory") else ("control" if tt.startswith("control") else None)
        if cond is None:
            continue
        m = re.search(r"(\d+)/(\d+)", tt)
        if not m:
            continue
        idx, total = int(m.group(1)), int(m.group(2))
        onset = float(row["onset"])
        if idx == total:
            trials.append({"condition": cond, "last_onset": onset, "list_len": total})
    return trials

def theta_power(fz_data, sfreq, t0, t1):
    i0, i1 = int(t0 * sfreq), int(t1 * sfreq)
    if i1 > len(fz_data):
        return np.nan
    seg = np.asarray(fz_data[i0:i1], dtype=np.float64)
    if len(seg) < 100:
        return np.nan
    freqs, psd = welch(seg, fs=sfreq, nperseg=min(512, len(seg)))
    mask = (freqs >= THETA_BAND[0]) & (freqs < THETA_BAND[1])
    return float(np.log(psd[mask].mean() + 1e-12))

results = []
for s in SUBJECTS:
    set_path = DATA / s / "eeg" / f"{s}_task-memory_eeg.set"
    events_path = DATA / s / "eeg" / f"{s}_task-memory_events.tsv"
    if not set_path.exists():
        print(f"{s}: missing .set file, skipping")
        continue
    print(f"Loading {s} ...")

    with h5py.File(set_path, "r") as f:
        sfreq = float(f["srate"][()].flatten()[0])
        fz_idx = get_fz_index(f)
        data = f["data"]  # (n_times, n_channels), on-disk, not loaded
        n_times = data.shape[0]
        print(f"  srate={sfreq}, Fz index={fz_idx}, n_times={n_times}")

        events_df = pd.read_csv(events_path, sep="\t")
        trials = last_digit_events(events_df)
        print(f"  {s}: {len(trials)} trials found")

        # Pull only the Fz column, as a plain numpy array (small: ~30 MB)
        fz = data[:, fz_idx]

    for tr in trials:
        t0 = tr["last_onset"] + DIGIT_DURATION
        t1 = t0 + WINDOW_S
        power = theta_power(fz, sfreq, t0, t1)
        if not np.isnan(power):
            results.append({"subject": s, "condition": tr["condition"], "list_len": tr["list_len"], "log_theta_power": power})
    del fz

df = pd.DataFrame(results)
df.to_csv(OUT / "15_digitspan_theta.csv", index=False)
print(f"\nTotal usable trials: {len(df)}")

print("\n=== Frontal (Fz) theta power, memory vs control ===")
summary = df.groupby(["subject", "condition"])["log_theta_power"].mean().unstack()
print(summary)

overall = df.groupby("condition")["log_theta_power"].agg(["mean", "std", "count"])
print("\nOverall:")
print(overall)

paired = summary.dropna()
if len(paired) >= 2:
    t, p = ttest_rel(paired["memory"], paired["control"])
    print(f"\nPaired t-test (memory vs control, n={len(paired)} subjects): t={t:.3f}, p={p:.3f}")
    diff = (paired["memory"] - paired["control"]).mean()
    print(f"Mean memory-control difference (log power): {diff:+.3f}")
    print("Direction matches paper's claim (memory > control)" if diff > 0 else "Direction does NOT match paper's claim")
else:
    print("\nNot enough subjects for a paired comparison.")

print(f"\nSaved {OUT / '15_digitspan_theta.csv'}")
