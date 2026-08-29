# Methodology notes (Blueprint V4)

**Title:** Trajectory-Adaptive Power-Profile Conformal Beam Tracking from Synchronized RGB and GPS for Reliable V2V mmWave Communications

**Data scope:** DeepSense 6G Scenarios 36 (this run). US 60 GHz V2V testbed. Bangladesh is motivating context only — not a data or operator-deployment claim.

## Guarantee language (Section 5a)

- **Static CRC (Phase 6):** finite-sample conformal guarantees need exchangeability (or a stated extension). Chronological non-overlapping trajectory splits prevent *leakage*; they do **not** by themselves give exchangeability. Trajectories are temporally correlated.
- **Online ACI / PID (Phase 7):** Adaptive Conformal Inference (Gibbs & Candès, 2021) and Conformal PID Control (Angelopoulos, Candès, Tibshirani, 2023) give long-run *time-averaged* risk control under their stated assumptions (bounded loss, the implemented update). That is a different object from split-conformal per-instance coverage. We report rolling empirical miss rate on the test stream.
- **Shift (RQ5, Stretch):** held-out day/night or scenario results are empirical robustness, not “guaranteed coverage,” unless a weighted-CRC density-ratio argument is supplied.

If a formal theorem does not apply to the exact implementation, report measured rolling / worst-trajectory miss rate only.

## Implemented pipeline

Phases 0–8 Core: reconstruction gate, leakage-safe 55/15/15/15 trajectory blocks, GPS+RGB branches, B1/B3, P1/P3 with smoothness on the profile head, fixed Top-k, static power CRC, exact-label CRC, ACI with \(q_{\min}=0\) and \(q_{\max}=\) calibration 99th-percentile gap, \(\eta\) chosen on validation only, block bootstrap CIs.

Stretch coded: B0/B2/B4, P2, Conformal PID vs ACI.
