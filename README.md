# EAI — EEG and wearable-sensor generalization experiments

Small research experiments on public EEG and wearable-sensor datasets, built
around one question: how much does a model's apparent accuracy depend on
evaluating it on people it has already seen, versus people it hasn't?

Full write-ups of each experiment, with figures and discussion, are posted
as part of a blog research thread (not linked here — see the repo owner's
site). This repo is the code and the raw decision log behind those posts.

## What's here

- `scripts/` — one script per experiment, numbered in the order they were run
- `notebook/lab_notes.md` — dated, informal notes from first opening the data
- `DECISIONS.md` — a dated log of every methodological choice, including
  mistakes caught and fixed along the way (e.g. a class-imbalance bug in an
  early accuracy metric, and a CSP-pooling failure that scored at chance
  before being diagnosed and fixed)
- `outputs/` — small CSV results and a few PSD plots; raw EEG/sensor data
  itself is not committed (see below)

## Experiments, roughly in order

1. Open and inspect one real EEG recording (PhysioNet EEG Motor
   Movement/Imagery dataset)
2. Confirm and filter 60 Hz line noise (1-40 Hz band-pass + 60 Hz notch)
3. A within-subject CSP+LDA baseline (left vs. right imagined fist movement)
4. Random vs. subject-wise cross-validation, on two EEG tasks (EEGMAT
   rest-vs-arithmetic, and the motor-imagery task above) — the core "does
   splitting matter" result
5. A calibration curve: how much of the cross-subject gap closes when the
   model sees a handful of the new person's own trials
6. The same split-evaluation question on a second signal type (WESAD
   wrist/chest wearable sensors, baseline vs. stress)
7. Wrist-only vs. chest-sensor accuracy cost, same WESAD data
8. A reproduction attempt of a published EEG finding (Pavlov et al. 2022,
   frontal midline theta in a digit-span task) — reported honestly,
   including that it did not replicate on the small subset tried here

## Setup

```
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Installs a `.venv` with MNE, NumPy, SciPy, scikit-learn, pandas, and the
extra packages (`h5py`, `pymatreader`) needed to read one dataset's
MATLAB v7.3 files.

## Data

No raw data is committed — `data/` is gitignored. Every script downloads
its own dataset on first run, from public sources:

- PhysioNet EEG Motor Movement/Imagery (ODC-By 1.0)
- PhysioNet EEGMAT (ODC-By 1.0)
- WESAD, via the dataset authors (Schmidt et al. 2018; non-commercial use,
  credit required)
- OpenNeuro ds003838 (Pavlov et al. 2022)

See each dataset's own license before reuse.
