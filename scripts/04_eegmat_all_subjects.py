"""Download all 36 EEGMAT subjects, build a metadata table (duration,
channel count, sampling rate per subject/condition), flag inconsistencies
before any modeling.
Run:  .\\.venv\\Scripts\\python.exe scripts\\04_eegmat_all_subjects.py
"""
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError
import mne
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eegmat"
OUT = ROOT / "outputs"
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

BASE_URL = "https://physionet.org/files/eegmat/1.0.0/"
N_SUBJECTS = 36

rows = []
for i in range(N_SUBJECTS):
    subj = f"Subject{i:02d}"
    for cond, suffix in (("rest", "_1.edf"), ("arithmetic", "_2.edf")):
        fname = f"{subj}{suffix}"
        dest = DATA / fname
        if not dest.exists():
            try:
                urlretrieve(BASE_URL + fname, dest)
            except HTTPError as e:
                print(f"MISSING {fname}: {e}")
                rows.append({"subject": subj, "condition": cond, "error": str(e)})
                continue
        raw = mne.io.read_raw_edf(dest, preload=False, verbose="ERROR")
        rows.append({
            "subject": subj,
            "condition": cond,
            "n_channels": len(raw.ch_names),
            "sfreq": raw.info["sfreq"],
            "duration_s": round(raw.times[-1], 1),
            "error": None,
        })
    if i % 6 == 0:
        print(f"...{i}/{N_SUBJECTS}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "04_eegmat_metadata.csv", index=False)

print("\n=== Summary ===")
print("Total files:", len(df))
print("Errors:", df["error"].notna().sum())
ok = df[df["error"].isna()]
print("\nChannel counts:", ok["n_channels"].value_counts().to_dict())
print("Sampling rates:", ok["sfreq"].value_counts().to_dict())
print("\nDuration by condition (s):")
print(ok.groupby("condition")["duration_s"].describe())
print(f"\nSaved {OUT / '04_eegmat_metadata.csv'}")
