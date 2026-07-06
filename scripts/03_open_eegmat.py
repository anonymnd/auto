"""Week 1 warm-up: open one subject of the EEGMAT (mental arithmetic) set.

PhysioNet EEGMAT has no built-in MNE loader, so we fetch the two EDF files
for one subject directly (Subject00_1.edf = resting baseline,
Subject00_2.edf = during mental arithmetic), 23-channel, each a short
artifact-free segment.
Run:  .\\.venv\\Scripts\\python.exe scripts\\03_open_eegmat.py
"""
from pathlib import Path
from urllib.request import urlretrieve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eegmat"
OUT = ROOT / "outputs"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

BASE_URL = "https://physionet.org/files/eegmat/1.0.0/"
SUBJECT = "Subject00"
FILES = {"rest": f"{SUBJECT}_1.edf", "arithmetic": f"{SUBJECT}_2.edf"}

local_paths = {}
for label, fname in FILES.items():
    dest = DATA / fname
    if not dest.exists():
        print(f"Downloading {fname} ...")
        urlretrieve(BASE_URL + fname, dest)
    local_paths[label] = dest

for label, path in local_paths.items():
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    print(f"\n=== {label} ({path.name}) ===")
    print("Channels:      ", len(raw.ch_names), raw.ch_names)
    print("Sampling rate: ", raw.info["sfreq"], "Hz")
    print("Duration:      ", round(raw.times[-1], 1), "s")

    fig = raw.copy().pick(raw.ch_names[:8]).plot(duration=10, scalings="auto", show=False)
    fig.savefig(OUT / f"03_eegmat_{label}_10s.png", dpi=120)

print(f"\nSaved plots in {OUT}")
