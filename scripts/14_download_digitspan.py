"""Level 7: download a few subjects of the Pavlov et al. 2022 digit-span
working-memory EEG dataset (OpenNeuro ds003838), to attempt reproducing
their reported finding: frontal midline theta (~6 Hz) increases during
the memory condition vs. the listen-only control condition (their Fig. 3d,
no numbers given in the paper text).
Run:  .\\.venv\\Scripts\\python.exe scripts\\14_download_digitspan.py
"""
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ds003838"
DATA.mkdir(parents=True, exist_ok=True)

BASE = "https://s3.amazonaws.com/openneuro.org/ds003838"
SUBJECTS = ["sub-032", "sub-033", "sub-034"]  # first 3 with EEG_excluded=no

FILES_PER_SUBJECT = [
    "eeg/{s}_task-memory_eeg.set",
    "eeg/{s}_task-memory_events.tsv",
    "eeg/{s}_task-memory_channels.tsv",
]

def download_resumable(url, dest, max_retries=15):
    head = requests.head(url, timeout=30)
    total = int(head.headers.get("content-length", 0))
    for attempt in range(max_retries):
        existing = dest.stat().st_size if dest.exists() else 0
        if total and existing >= total:
            return
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        mode = "ab" if existing else "wb"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, mode) as f:
                    downloaded = existing
                    last_print = 0
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_print >= 100_000_000:
                            pct = 100 * downloaded / total if total else 0
                            print(f"    {downloaded/1e6:.0f} MB / {total/1e6:.0f} MB ({pct:.1f}%)")
                            last_print = downloaded
            if not total or dest.stat().st_size >= total:
                return
        except (requests.exceptions.RequestException, ConnectionError) as e:
            print(f"    attempt {attempt+1}/{max_retries} failed: {e}. retry in 5s")
            time.sleep(5)
    raise RuntimeError(f"failed to download {url} after {max_retries} attempts")

for s in SUBJECTS:
    sdir = DATA / s / "eeg"
    sdir.mkdir(parents=True, exist_ok=True)
    for rel in FILES_PER_SUBJECT:
        rel_s = rel.format(s=s)
        url = f"{BASE}/{s}/{rel_s}"
        dest = DATA / s / rel_s
        print(f"{s}: {rel_s}")
        download_resumable(url, dest)

print("\nAll subjects downloaded:")
for s in SUBJECTS:
    setf = DATA / s / "eeg" / f"{s}_task-memory_eeg.set"
    print(f"  {s}: {setf.stat().st_size/1e6:.0f} MB" if setf.exists() else f"  {s}: MISSING")
