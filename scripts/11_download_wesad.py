"""Level 6: download WESAD (wearable stress/affect, 15 subjects, chest+wrist
sensors). Non-commercial use, credit Schmidt et al. 2018 ICMI.
Run:  .\\.venv\\Scripts\\python.exe scripts\\11_download_wesad.py
"""
from pathlib import Path
import time
import requests
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
ZIP_PATH = DATA / "WESAD.zip"
EXTRACT_DIR = DATA / "WESAD"

def download_resumable(url, dest, max_retries=20):
    head = requests.head(url, allow_redirects=True, timeout=30)
    total = int(head.headers.get("content-length", 0))
    for attempt in range(max_retries):
        existing = dest.stat().st_size if dest.exists() else 0
        if total and existing >= total:
            print("Already fully downloaded.")
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
                        if downloaded - last_print >= 50_000_000:
                            pct = 100 * downloaded / total if total else 0
                            print(f"  {downloaded/1e6:.0f} MB / {total/1e6:.0f} MB ({pct:.1f}%)")
                            last_print = downloaded
            if total and dest.stat().st_size >= total:
                print("Download complete.")
                return
        except (requests.exceptions.RequestException, ConnectionError) as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in 5s...")
            time.sleep(5)
            continue
    raise RuntimeError(f"Failed to complete download after {max_retries} attempts.")

if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 2_000_000_000:
    print(f"Downloading WESAD.zip (~2.2 GB) to {ZIP_PATH} (resumable) ...")
    download_resumable(URL, ZIP_PATH)
else:
    print(f"{ZIP_PATH} already exists ({ZIP_PATH.stat().st_size/1e6:.0f} MB), skipping download.")

if not EXTRACT_DIR.exists():
    print(f"Extracting to {EXTRACT_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DATA)
    print("Extraction complete.")
else:
    print(f"{EXTRACT_DIR} already exists, skipping extraction.")

subjects = sorted([p.name for p in EXTRACT_DIR.glob("S*") if p.is_dir()]) if EXTRACT_DIR.exists() else []
print(f"\nSubjects found: {len(subjects)} -> {subjects}")
