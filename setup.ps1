# Run once from this folder:  powershell -ExecutionPolicy Bypass -File setup.ps1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Done. Next: .\.venv\Scripts\python.exe scripts\01_open_eeg.py"
