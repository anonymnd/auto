# Decisions log (dated, one entry per choice that affects results)

Format: date — decision — why — what it rules out.

## 2026-07-03 — Line noise: 60 Hz confirmed on EEGBCI motor-imagery data
Zoomed PSD (45-65 Hz) on subject S001/run R04 shows a clear 60 Hz spike
(~16 dB above baseline), consistent with US power-line interference. See
`notebook/lab_notes.md`. Decision pending: whether to notch-filter before
baseline modeling, or rely on band limiting (mu/beta ~8-30 Hz) being far
enough from 60 Hz to skip it for the first baseline. Revisit at the CSP+LDA
baseline step.

## 2026-07-03 — Dataset order: motor imagery first, EEGMAT second
Start with PhysioNet EEG Motor Movement/Imagery (109 subjects, 64 ch, 160 Hz)
because it is the dataset the leakage experiment (random vs subject-wise split)
will be built on. EEGMAT (36 subjects, rest vs arithmetic) is the one-week
warm-up for subject-wise CV mechanics on a smaller, simpler two-class problem
before scaling up. Rules out: starting modeling before the split logic is
tested on the small set.

## 2026-07-03 — Split discipline: group by subject, fit preprocessing per split
Any cross-validation must use subject ID as the group key
(`sklearn.model_selection.GroupKFold` / `LeaveOneGroupOut`), and any
`fit`/`fit_transform` step (scaler, CSP, ICA) is fit only on the training
fold, never on the full dataset before splitting. Why: Brookshire et al.
2024 found only 17 of 63 DNN-EEG studies reviewed clearly avoided
subject-level leakage; a mistake here invalidates every downstream result.
Rules out: computing any per-subject-dependent statistic (mean, CSP filters)
on data outside the training fold.

## 2026-07-04 — EEGMAT actual format, correcting plan assumptions
Opened Subject00 (`03_open_eegmat.py`, fetched manually — no MNE loader for
this set). Actual: 500 Hz (plan doc marked this unconfirmed), 21 channels
(20 EEG + 1 ECG, not the "23 channels" in the plan), rest file 182 s,
arithmetic file 62 s for this subject — lengths may vary per subject, needs
checking across all 36 before assuming a fixed 60 s. Channel names carry
`EEG `/`ECG ` prefixes (e.g. `EEG Fp1`), will need stripping before montage
assignment. Rules out: assuming EEGMAT config without opening the files.

## 2026-07-04 — EEGMAT full 36-subject metadata pull, correcting fixed-duration assumption
All 72 files (36 subjects × rest/arithmetic) downloaded, 0 errors, 21 ch /
500 Hz consistent across all. Durations are NOT fixed as the plan doc
assumed ("60 s artifact-free segments"): arithmetic ranges 35-62 s
(mean 61.25, std 4.5), rest ranges 80-188 s (mean 179, std 17). Shortest
arithmetic trial is Subject10 (35 s) — verified not truncated by reading
with MNE directly. Implication: epoching must use a window short enough to
fit the shortest trial (<=35 s usable, likely cut to e.g. 15-30 s epochs)
or drop/flag short-trial subjects; cannot assume uniform-length windows
across subjects. See `outputs/04_eegmat_metadata.csv` for per-subject table.

## 2026-07-04 — First leakage result, with an imbalance caveat
`05_leakage_experiment.py`: 10 s epochs, log band power (5 bands x 20 EEG
ch = 100 features), LogisticRegression, scaler fit inside each fold.
850 epochs total (637 rest, 213 arithmetic — imbalanced, since rest trials
run ~3x longer than arithmetic trials). Random 5-fold: 0.914 +/- 0.022.
Subject-wise (GroupKFold): 0.739 +/- 0.069. Gap = 0.175, same direction as
Brookshire et al. 2024. Null (shuffled labels, grouped split): 0.654 +/- 0.033.
CAVEAT: with 75% majority class, a trivial "always predict rest" baseline
scores ~0.75 — higher than the null model got, and close to the
subject-wise score. Raw accuracy is not trustworthy here.

FIX applied same day: reran with `scoring="balanced_accuracy"`. Corrected
numbers: random split 0.871 +/- 0.025, subject-wise 0.690 +/- 0.048, null
(shuffled, grouped) 0.496 +/- 0.015 — null now sits exactly at chance (0.5)
as it should. Gap = 0.182. Subject-wise score is clearly above chance
(0.690 vs 0.496 null), so the classifier has real signal, it just loses
~18 points of balanced accuracy going from random to subject-wise split.
This is the week-1 warm-up result (plan section D, step 1). See
`outputs/05_leakage_results.csv` for the final numbers.
