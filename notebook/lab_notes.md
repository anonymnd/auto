# Lab notes (private, dated, honest)

## 2026-07-03 — first EEG file
- Channels / sampling rate / duration: PhysioNet EEG Motor Movement/Imagery, subject S001, run R04 (imagined left/right fist). 64 channels, 160 Hz, 125 s.
- What the raw trace looks like: noisy oscillating traces on 8 frontocentral channels (FC5, FC3, FC1, FCz, FC2, FC4, FC6, C5). Amplitude roughly 100-130 µV peak-to-peak. No obvious flat or dead channels in this window.
- Annotations: T0 (rest), T1, T2 (left/right fist imagery) mark windows directly on the plot — color-coded blue/green blocks, each run ~4s.
- What the spectrum shows: power falls off steeply with frequency (~40 dB near 0 Hz down to ~0 dB by 50-60 Hz), typical 1/f-like EEG shape. No clear separate line-noise spike visible at 50/60 Hz in this plot — needs closer zoom to confirm. No distinct alpha (~10 Hz) peak stands out above the general slope at this view.
- What surprised me: how much the 1/f slope dominates the spectrum — alpha/line-noise features, if present, are subtle against it.
- Follow-up (same day, script 02): zoomed 45-65 Hz confirms clear 60 Hz line-noise spike (~16 dB peak), invisible at the 0-60 Hz full-range scale. event_id = {T0:1, T1:2, T2:3}; run 4 is imagined fist movement, so T1=left fist, T2=right fist, T0=rest.
- What I do not understand yet: whether 60 Hz line noise this size will need a notch filter before any modeling, or whether the band of interest (mu/beta, ~8-30 Hz) is far enough away to ignore it for now.
