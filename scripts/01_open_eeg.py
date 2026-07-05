"""Week 1: open one real EEG recording and just look at it.

Downloads ONE subject (a few MB) of the PhysioNet EEG Motor Movement/Imagery
dataset through MNE, prints basic facts, and saves two plots.
Run:  .\.venv\Scripts\python.exe scripts\01_open_eeg.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SUBJECT = 1
RUN = 4  # run 4 = imagined left/right fist movement (motor imagery)

paths = mne.datasets.eegbci.load_data(SUBJECT, [RUN], path=str(DATA), update_path=True)
raw = mne.io.read_raw_edf(paths[0], preload=True, verbose="ERROR")
mne.datasets.eegbci.standardize(raw)

print("\n=== What is in this file? ===")
print("Channels:      ", len(raw.ch_names), raw.ch_names[:8], "...")
print("Sampling rate: ", raw.info["sfreq"], "Hz")
print("Duration:      ", round(raw.times[-1], 1), "s")
print("Annotations:   ", sorted(set(raw.annotations.description)))

# Plot 1: 10 seconds of raw signal, 8 channels
fig = raw.copy().pick(raw.ch_names[:8]).plot(
    duration=10, scalings="auto", show=False)
fig.savefig(OUT / "01_raw_10s.png", dpi=120)

# Plot 2: frequency spectrum (where is the energy?)
fig = raw.compute_psd(fmax=60).plot(show=False)
fig.savefig(OUT / "01_spectrum.png", dpi=120)

print(f"\nSaved plots in {OUT}")
