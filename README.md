# EAI — EEG evaluation experiments

Local-only project. Plan: weeks 1–2 open and look at data; weeks 5–8 build a
CSP+LDA baseline and compare random vs subject-wise evaluation.

## Setup (once)
    powershell -ExecutionPolicy Bypass -File setup.ps1

## Week 1
    .\.venv\Scripts\python.exe scripts\01_open_eeg.py
Plots appear in `outputs\`. Then fill `notebook\lab_notes.md`.

Data: PhysioNet EEG Motor Movement/Imagery (public, consented, anonymised).
