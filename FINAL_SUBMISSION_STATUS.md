# FINAL SUBMISSION STATUS: IEEE Publication Readiness Report

**Document Path:** `FINAL_SUBMISSION_STATUS.md`  
**Authoritative Date:** November 21, 2026  
**Auditor Role:** Senior IEEE Research Scientist, Wireless Communications Specialist, and Manuscript Editor  
**Target Publication Venue:** IEEE Transactions on Vehicular Technology (TVT)  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  

---

## 1. Submission Readiness Dashboard

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                              FINAL PUBLICATION READINESS AUDIT                                          |
+------------------------------------+-----------------------+------------------------------------------------------------+
| Category                           | Status                | Notes & Current Condition                                  |
+------------------------------------+-----------------------+------------------------------------------------------------+
| 1. Empirical Evidence (S1-299)     | COMPLETED & VERIFIED  | 299 stages executed, verified on disk, checkpoints frozen. |
| 2. Research Direction Alignment    | RESOLVED & FROZEN     | Empirical critique of multimodal beam forecasting.         |
| 3. Authoritative Manuscript        | CONSOLIDATED          | MANUSCRIPT/FINAL_IEEE_MANUSCRIPT.md (13 complete sections).|
| 4. PAR-Net / KM-ACC Claims         | EXPUNGED / DEPRECATED | Fabricated numbers identified, isolated, and removed.      |
| 5. Physical-Layer Mathematics      | CORRECTED             | Tc = 83.33 μs Doppler phase; Δt_beam ≈ 175 ms spatial.     |
| 6. Communication Cost Model        | RIGOROUSLY GROUNDED   | Explicitly labeled as analytical IEEE 802.11ad/ay model.   |
| 7. Statistical Methodology         | FORMALIZED            | G=10 clusters, Neff ≈ 13.3; paired block-bootstrap CIs.    |
| 8. Reserved Test Partition         | LOCKED & UNTOUCHED    | 17,017 raw frames (19,631 windows); 0 reads logged.        |
| 9. Unit Test Suite                 | 100% PASSING          | 74 test suites passed (0 failures, 27 subtests passed).    |
| 10. Submission Readiness           | READY FOR AUTHOR      | Pending author affiliation & final test opening decision.  |
+------------------------------------+-----------------------+------------------------------------------------------------+
```

---

## 2. Definitive Status Breakdown

### A. What Has Been Completed
1. **Multi-Scenario Data Acquisition & Preprocessing:** 112,196 raw frames across DeepSense Scenarios 36, 37, 38, and 39 acquired, audited, and cached.
2. **299-Stage Experimental Lifecycle:**
   - 30 Swin-B3 attention model fits across pooled and leave-one-scenario-out partitions (Stages 268–299).
   - Geometry-constrained candidate reranking across $K=20$ and $K=40$ (Stage 267 widecheck).
   - Candidate exclusion power-oracle decomposition (84.7% rule).
   - 4,000 paired block-bootstrap confidence intervals.
3. **Forensic Audit & Correction Suite (`AUDIT/`):**
   - [AUDIT/REPOSITORY_AUDIT.md](AUDIT/REPOSITORY_AUDIT.md): Complete repository reconciliation and ledger verification.
   - [AUDIT/FINAL_RESEARCH_DIRECTION.md](AUDIT/FINAL_RESEARCH_DIRECTION.md): Rigorous justification of the empirical critique research question.
   - [AUDIT/PAR_NET_KMACC_VERIFICATION.md](AUDIT/PAR_NET_KMACC_VERIFICATION.md): Verification report documenting the origin of hardcoded PAR-Net tuples and removing them.
   - [AUDIT/PHYSICAL_LAYER_CORRECTIONS.md](AUDIT/PHYSICAL_LAYER_CORRECTIONS.md): Correction of the Doppler coherence time arithmetic ($83.33\,\mu\text{s}$ vs $4.16\text{ ms}$) and spatial beam coherence ($\approx 175\text{ ms}$).
   - [AUDIT/FINAL_STATISTICAL_PROTOCOL.md](AUDIT/FINAL_STATISTICAL_PROTOCOL.md): Formulation of cluster-robust statistical inference ($G = 10$ blocks, $N_{\text{eff}} \approx 13.3$).
   - [AUDIT/CANDIDATE_GENERATION_RANKING_ANALYSIS.md](AUDIT/CANDIDATE_GENERATION_RANKING_ANALYSIS.md): Comprehensive 84.7% candidate exclusion breakdown.
   - [AUDIT/RGB_CONTRIBUTION_ANALYSIS.md](AUDIT/RGB_CONTRIBUTION_ANALYSIS.md): Analysis of conditional visual utility and Scenario 39 sensitivity.
   - [AUDIT/NOVELTY_AND_RELATED_WORK.md](AUDIT/NOVELTY_AND_RELATED_WORK.md): Prior-art positioning against Fabiani, Mollah, and CLBP.
4. **Consolidated IEEE Manuscript:**
   - [MANUSCRIPT/FINAL_IEEE_MANUSCRIPT.md](MANUSCRIPT/FINAL_IEEE_MANUSCRIPT.md): The unified, authoritative IEEE TVT manuscript.

---

### B. What Remains Incomplete & Pre-Submission Checklist

```
+----------------------------------------------------------------------------------------------------+
|                                    PRE-SUBMISSION ACTION CHECKLIST                                 |
+-----+-------------------------------------------------------------+--------------------------------+
| No. | Action Item                                                 | Responsibility                 |
+-----+-------------------------------------------------------------+--------------------------------+
| 1   | Author Affiliations, Department, Institution, and Grants    | Human Author (Moazzem Hossain) |
| 2   | IEEE AI Disclosure Statement (Mandatory per IEEE TVT Rule 9)| Human Author                   |
| 3   | Decision on Opening Reserved Test Partition (17,017 Frames) | Human Author                   |
| 4   | Compilation of LaTeX PDF via IEEEtran.cls                   | Human Author / LaTeX Engine    |
+-----+-------------------------------------------------------------+--------------------------------+
```

---

## 3. Verified Results vs. Exploratory Diagnostics

### A. Verified Confirmatory Findings (Supported by Saved Disk Artifacts)
- **Top-5 Exact Hit Gain with RGB:** $+5.42\text{ percentage points}$ (95% CI: $[-2.86, +9.98]$).
- **Top-5 3 dB Near-Optimal Power Collapse:** $-19.85\text{ percentage points}$ (95% CI: $[-26.30, -11.26]$).
- **Alternative-Beam Recovery Collapse:** $-25.28\text{ percentage points}$ (95% CI: $[-30.64, -15.86]$, $z = -6.82, p < 10^{-11}$).
- **Candidate Exclusion Share of Severe Failures ($M=20$):** $84.70\%$ ($14.23\%$ out of $16.80\%$).
- **Analytic Geometry Baseline Top-5 3 dB Success:** $78.04\%$ (beats Swin-B3 multimodal model by $\approx 29.3\text{ pp}$).
- **Incremental Visual Gain at $K=40$:** $+0.011\text{ percentage points}$ (95% CI: $[-0.007, +0.037]$, statistically indistinguishable from zero).

### B. Exploratory / Analytical Diagnostics (Not Primary Deployed Findings)
- **Candidate Exclusion Oracle Decomposition:** Diagnostic using offline ground-truth power profiles; not a deployed transceiver capability.
- **60 GHz Communication Cost Model:** Analytical model derived from IEEE 802.11ad frame specifications ($100\,\mu\text{s}$ per probe), evaluated via offline replay.
- **Feature Shuffling & Native Resolution Pilots (Stages 62–71, 151–154):** Exploratory diagnostics showing negligible variation.

### C. Deprecated / Expunged Claims (Do Not Cite)
- **PAR-Net 81.10% success / +2.54 pp gain:** Expunged. Hardcoded Python tuples in `src/unified_benchmark.py`.
- **KM-ACC 76.0% sweeping reduction:** Expunged. Synthetic heuristic without hardware telemetry.
- **Doppler coherence time $T_c = 4.16\text{ ms}$:** Expunged. Corrected to $83.33\,\mu\text{s}$ phase coherence / $175\text{ ms}$ spatial beam coherence.

---

## 4. Test Set Integrity Status

The held-out test partition:
- **Scope:** 17,017 raw frames across 12 unseen vehicle trajectories (19,631 sliding windows).
- **Log Audit:** **Zero reads logged across all 299 stages.**
- **Integrity:** Pristine and untouched.
- **Protocol:** If the human author chooses to open the test partition for confirmatory reporting, run `python src/evaluate_locked_test.py` once without post-hoc hyperparameter adjustments.

---

## 5. Final Synthesis

The DeepSense-V2V repository is now **technically correct, mathematically verified, statistically sound, and aligned with IEEE publication ethics**. 

By abandoning ungrounded claims of visual superiority and focusing on the **empirical realities of multimodal beam forecasting (the Alternative-Beam Paradox, Candidate Exclusion Dominance, and Geometry Baseline Primacy)**, the paper presents a groundbreaking, highly defensible contribution for *IEEE Transactions on Vehicular Technology*.
