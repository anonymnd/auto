"""Level 2: preprocess and see what changes.
Band-pass filter (1-40 Hz) + 60 Hz notch on the motor-imagery run opened in
01/02, compare PSD before vs after on the same 8 channels.
Run:  .\\.venv\\Scripts\\python.exe scripts\\06_preprocess_motor_imagery.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SUBJECT = 1
RUN = 4

paths = mne.datasets.eegbci.load_data(SUBJECT, [RUN], path=str(DATA), update_path=True)
raw = mne.io.read_raw_edf(paths[0], preload=True, verbose="ERROR")
mne.datasets.eegbci.standardize(raw)
raw_orig = raw.copy()

# Band-pass 1-40 Hz (mu/beta band of interest sits inside this) + 60 Hz notch
raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
raw.notch_filter(freqs=[60.0], verbose="ERROR")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
raw_orig.compute_psd(fmax=80, verbose="ERROR").plot(axes=axes[0], show=False)
axes[0].set_title("Before filtering")
raw.compute_psd(fmax=80, verbose="ERROR").plot(axes=axes[1], show=False)
axes[1].set_title("After 1-40 Hz band-pass + 60 Hz notch")
fig.tight_layout()
fig.savefig(OUT / "06_psd_before_after.png", dpi=120)
print(f"Saved {OUT / '06_psd_before_after.png'}")

print("\nPeak power before (0-80Hz, mean over channels):")
psd_before, freqs = raw_orig.compute_psd(fmax=80, verbose="ERROR").get_data(return_freqs=True)
psd_after, _ = raw.compute_psd(fmax=80, verbose="ERROR").get_data(return_freqs=True)
import numpy as np
idx60 = (np.abs(freqs - 60)).argmin()
print(f"  at 60 Hz: before={psd_before[:, idx60].mean():.3e}, after={psd_after[:, idx60].mean():.3e}  (ratio={psd_before[:, idx60].mean()/psd_after[:, idx60].mean():.1f}x reduction)")
