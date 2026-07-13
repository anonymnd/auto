# Decisions log (dated, one entry per choice that affects results)

Format: date — decision — why — what it rules out.

## 2026-07-13 — New side-thread: does the leakage pattern hold outside neuroscience?
Prompted by the plan's own suggestion (section P) to try interaction-log
data before any brain signal for the adaptive-learning idea. Could not use
my own study-session logs (none exist yet, no logging tool in place), so
switched to a real public dataset with the same kind of signal instead of
fabricating data: ASSISTments 2009-2010 skill-builder (classic knowledge-
tracing dataset, 525,534 interactions, 4213 students, CC BY 4.0, mirrored
on figshare by Zhijie Liang — original ASSISTments 2017 release needed an
email-gated signup form, skipped). Literature pass first: learning-styles
"matching" theory has weak-to-no support (Pashler et al. 2008; a 2024
meta-analysis found only a small effect, g=0.31, "too small to warrant
adoption"); a more defensible individual-difference source is
"structure-building" ability (McDaniel, Marsh & Gouravajhala 2022);
knowledge tracing (Corbett & Anderson 1994 onward) is the standard
technique behind adaptive tutors.

## 2026-07-13 — ASSISTments: predicting correctness, random vs student-wise split
`16_assistments_leakage.py`: features are purely behavioral, no skill
content — log(first-response time), attempt count, hint count, hint
total, log(opportunity = times this student has practiced this skill
before), bottom-hint flag. Target: correct (0/1). Logistic regression,
same random/group/null structure as every prior experiment in this
project. 514,304 usable interactions after dropping bad timestamps,
4213 students, positive rate 0.681.
Random split: 0.866 +/- 0.002. Student-wise split: 0.865 +/- 0.006. Null
(shuffled, grouped): 0.500 +/- 0.000 exactly. Gap = 0.001 — essentially
none, the smallest of any experiment in this project, on the largest
sample (4213 students vs 10-36 in the EEG/wearable experiments).
WHY THIS IS DIFFERENT FROM EVERY EEG/WEARABLE RESULT SO FAR: the features
here (response time, hint use, attempt count) are generic behavioral
signals of "is this question hard for whoever is answering it right now,"
not signals tied to one person's physiology. A slow response + many hints
+ many attempts predicts a wrong answer almost the same way for any
student. There is very little "who is this" information for a model to
exploit through a random split, so random and student-wise splits land
in almost the same place. This directly supports the "small gap size is
informative about the features, not just the splitting method" point
from the synthesis post: EEG/wearable features carry subject-identity
information (skull geometry, skin conductance baseline, etc.) that these
behavioral features simply don't. See
`outputs/16_assistments_leakage.csv`.

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

## 2026-07-06 — Preprocessing filter choice: 1-40 Hz band-pass + 60 Hz notch
`06_preprocess_motor_imagery.py` on S001/R04. Band-pass 1-40 Hz (covers
mu/beta, 8-30 Hz, with margin) + notch at 60 Hz. Confirms the 60 Hz line
noise found earlier is removed (97,686x power reduction at 60 Hz vs
unfiltered). See `outputs/06_psd_before_after.png`. This settles the open
question from the first lab note: yes, notch filter before modeling.
Decision: use this filter (1-40 Hz band-pass + 60 Hz notch) as the default
preprocessing step for the motor-imagery baseline (Level 3, CSP+LDA), fit
identically on every subject, never fit per-split.

## 2026-07-06 — Level 3 baseline: CSP+LDA, within subject 1, weak result
`07_csp_lda_baseline.py`: runs 4/8/12 concatenated, 1-40 Hz + 60 Hz notch,
epochs -0.5 to 2.5 s around T1 (left fist) / T2 (right fist), CSP(4) + LDA,
5-fold stratified CV (CSP fit inside each fold, not leaked). Only 45 epochs
total (23 left, 22 right) for one subject. Balanced accuracy 0.535 +/- 0.145
(fold range 0.300-0.700) — barely above chance, high variance. Honest
finding: single-subject, 45-trial baseline is weak and noisy; this is
consistent with motor imagery being a genuinely hard, noisy signal, not a
pipeline bug (filter/epoch mapping already verified in prior notes). Do not
oversell this number. See `outputs/07_csp_lda_baseline.csv`.

## 2026-07-06 — Level 4: pooled cross-subject CSP+LDA shows NO signal (honest null)
`08_motor_imagery_leakage.py`: 10 subjects, runs 4/8/12 pooled, same CSP(4)+LDA
pipeline as the Level 3 baseline, CSP fit inside each fold (no leak). 450
epochs total. Random split: 0.452 +/- 0.022. Subject-wise split: 0.504 +/- 0.023.
Null (shuffled labels): 0.490 +/- 0.025. All three sit at chance (~0.5);
gap is -0.052 (negative — opposite direction from the EEGMAT result, within
noise of the null).
WHY THIS HAPPENED (not a bug, a modeling limitation): a single CSP filter
set fit on pooled multi-subject data does not capture each subject's own
spatial pattern — CSP is known to be subject-specific, and naively pooling
epochs across subjects before fitting CSP is a known-weak approach (this is
exactly the generalization problem the whole project is about, showing up
at the baseline stage already). The single-subject Level 3 result (0.535,
barely above chance, n=45) was already weak; pooling made it worse, not
better. Cannot demonstrate the random-vs-subject-wise leakage gap on THIS
pipeline because there is no learned signal to leak in the first place.
CONTRAST with EEGMAT: that pipeline (band power features, not CSP) found
real signal (0.871 random / 0.690 subject-wise / 0.496 null) and a clear
leakage gap. Motor imagery L/R-fist with pooled CSP does not, yet.
NEXT STEP (not yet done): CSP must be fit per-subject (or use a
cross-subject-invariant feature like per-subject band power / Riemannian
geometry, as MOABB/Jayaram & Barachant suggest) before a pooled or
leave-subject-out comparison means anything. Report this null result as-is
rather than hide it — it is itself evidence for the "generalization is
hard" thesis. See `outputs/08_motor_imagery_leakage.csv`.

## 2026-07-06 — Fix for Level 4 null: band-power features instead of pooled CSP
`09_motor_imagery_leakage_bandpower.py`: replaced CSP+LDA with log band
power (theta/alpha-mu/beta/low-gamma x 32 EEG ch = 256 features) +
LogisticRegression, same 10 subjects, same epochs (now 0-2s, no CSP
rank issues). Random split: 0.599 +/- 0.041. Subject-wise: 0.573 +/- 0.043.
Null (shuffled, grouped): 0.456 +/- 0.044. This time null is clearly below
both real scores, so there IS learned signal (unlike the CSP attempt).
Gap (random - subject-wise) = 0.027, small but real, same direction as
expected. Confirms the diagnosis in the prior entry: CSP needs per-subject
fitting to work; band power is more pooling-robust. Weak gap compared to
EEGMAT's 0.182 is itself informative — motor imagery L/R-fist is a harder,
noisier signal than rest-vs-arithmetic, consistent with the Level 3 single-
subject result (0.535) also being weak. See
`outputs/09_motor_imagery_leakage_bandpower.csv`.

## 2026-07-06 — Level 5: calibration curve, real effect
`10_calibration_curve.py`: leave-one-subject-out (10 subjects, same
band-power features as the fixed Level 4 run), train on 9 subjects, add
k in {0,2,5,10,20} of the held-out subject's own trials, evaluate on a
fixed 20-epoch holdout per subject never used for calibration. Mean
balanced accuracy across the 10 held-out subjects: k=0: 0.559 +/- 0.106,
k=2: 0.554 +/- 0.121 (small dip, within noise), k=5: 0.577 +/- 0.122,
k=10: 0.597 +/- 0.127, k=20: 0.634 +/- 0.069 — std also shrinks as k grows,
i.e. calibration both raises the average and makes results more consistent
across subjects. This is the plan's central "sense, interpret, adapt,
measure again" loop in its smallest form (plan section D step 2), on real
data, same pipeline fix as the leakage experiment above. See
`outputs/10_calibration_curve.csv`.

## 2026-07-08 — Level 6: second modality (WESAD wrist sensors), same question
`12_wesad_leakage.py`: WESAD dataset (Schmidt et al. 2018), 15 subjects,
Empatica E4 wrist sensors (BVP 64Hz, EDA 4Hz, TEMP 4Hz, 3-axis ACC 32Hz),
30s windows, mean+std per channel = 8 features, baseline vs stress labels
only (amusement/meditation excluded for this binary comparison). 889 windows.
LogisticRegression, same random/group/null structure as prior experiments.
Random split: 0.722 +/- 0.039. Subject-wise: 0.694 +/- 0.060. Null
(shuffled, grouped): 0.494 +/- 0.010 — clean chance baseline. Both real
scores sit clearly above null, so there is genuine signal in just 8 simple
wrist features. Gap = 0.029, small — similar in size to the motor-imagery
gap (0.027), much smaller than EEGMAT's 0.182.
CROSS-MODALITY PATTERN across the three leakage experiments run so far:
EEGMAT (scalp EEG, rest vs arithmetic) has the largest leakage gap (0.182)
and the strongest raw signal; motor imagery (scalp EEG, L/R fist) and
WESAD (wrist, baseline vs stress) both show smaller, similar-sized gaps
(~0.03) despite being a different task AND a different sensor type. One
honest read: a small gap is not automatically "no leakage risk" — it may
mean the task's between-subject signal is just naturally smaller relative
to its within-subject signal, not that subject-wise evaluation matters
less. Needs more datasets before generalizing. See
`outputs/12_wesad_leakage.csv`.

## 2026-07-08 — WESAD wrist vs chest: the sensor-placement cost
`13_wesad_wrist_vs_chest.py`: same 15 subjects, same baseline-vs-stress
task, same subject-wise evaluation, two feature sets — wrist (Empatica E4,
8 features) vs chest (RespiBAN: ECG/EDA/EMG/Resp/Temp/ACC at 700Hz,
12 features, same 30s windows). Chest: 0.807 +/- 0.066. Wrist: 0.694 +/- 0.060.
Both nulls near chance (0.500/0.494). Cost of wrist-only (vs. chest): 0.114
balanced accuracy — switching to the consumer-grade, comfortable device
instead of the clinical chest rig costs about 11 points. This is a concrete,
reportable answer to "how much do you lose going cheap/wearable," one of
the plan's own candidate projects (section F, "NEXT" tier). See
`outputs/13_wesad_wrist_vs_chest.csv`.

## 2026-07-08 — Level 7: reproduction attempt, honest non-replication
Target: Pavlov et al. 2022 (OpenNeuro ds003838, digit-span working memory
EEG), their qualitative claim of frontal midline theta (~6 Hz, Fz)
increasing in the memory condition vs. listen-only control (their Fig. 3d
-- no numbers given in the paper text). Downloaded 3 subjects
(sub-032/033/034, the first three with EEG_excluded=no per
participants.tsv), memory-task .set files (~1.5-1.7 GB each, EEGLAB v7.3/
HDF5 format -- needed pymatreader+h5py, not in original requirements.txt,
installed ad hoc). `15_digitspan_theta_reproduction.py` reads only the Fz
channel directly via h5py (full 63-channel load needs ~3.5GB RAM per
subject and this machine didn't reliably have that free -- lazy single-
channel read avoids it). Theta (4-8Hz) log power via Welch, computed on
the 3s window right after each trial's last digit (the retention interval
before recall), per trial, memory vs. control.
VALIDATION: trial counts per subject (54 control + 108 memory = 162) match
the paper's stated per-subject counts exactly -- confirms the event
parsing (trial_type text matching, "last digit in sequence" detection) is
correct.
RESULT: mean memory-control theta difference = -0.050 (log power), i.e.
theta was LOWER in memory than control on average -- opposite direction
from the paper's claim. 2 of 3 subjects (032, 033) went the "wrong" way,
1 (034) went the right way but barely. Paired t-test across the 3
subjects: t=-1.190, p=0.356, not significant.
HONEST READ: this does not replicate the paper's claim, on this small
sample. Candidate reasons, none confirmed: (1) n=3 is far too small to
draw any real conclusion either way, the paper's own effect is only shown
as a figure with no numbers, so I don't know its effect size either; (2)
my window choice (fixed 3s after last digit) is a guess at "the retention
interval" from the task description, not validated against the paper's
own epoching; (3) no baseline correction (power relative to the pre-trial
baseline) was applied, which spectral papers normally do and I did not;
(4) FCz was the online reference during recording (per paper's methods),
not re-referenced here, which could shift frontal channel power. This is
reported as a non-replication, not a refutation of the original finding --
the honest conclusion is "my small, quick version didn't show it," not
"the effect doesn't exist." See `outputs/15_digitspan_theta.csv`.
