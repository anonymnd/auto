"""Follow-up to 01: confirm what T1/T2 mean on this run, and zoom the
45-65 Hz band to check for line noise.
Run:  .\\.venv\\Scripts\\python.exe scripts\\02_check_annotations_and_noise.py
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

print("\n=== Event mapping for run", RUN, "===")
events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
print("event_id:", event_id)
print("Run 4/8/12 = imagined fist movement: T1=left fist, T2=right fist, T0=rest.")

fig = raw.compute_psd(fmin=45, fmax=65).plot(show=False)
fig.suptitle("Zoom 45-65 Hz (line noise check)")
fig.savefig(OUT / "02_spectrum_45_65hz.png", dpi=120)
print(f"\nSaved {OUT / '02_spectrum_45_65hz.png'}")
