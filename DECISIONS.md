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
