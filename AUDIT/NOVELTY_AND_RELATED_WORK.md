# NOVELTY AND RELATED WORK AUDIT: Literature Positioning and Prior-Art Differentiation

**Document Path:** `AUDIT/NOVELTY_AND_RELATED_WORK.md`  
**Authoritative Date:** November 21, 2026  
**Auditor:** Senior IEEE Reviewer & Wireless Communications Research Specialist  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  

---

## 1. Executive Summary

A critical weakness identified in previous drafts was the overstatement of technical novelty. Several early documents incorrectly claimed that "multimodal GPS + front/rear camera fusion for vehicular beam prediction" was an original contribution of this repository.

This audit establishes that **multimodal GPS + camera beam prediction is already established in published literature** (e.g., Alrabeiah et al., Charan et al., ITU Challenge 2022/2023, and Fabiani et al., 2024).

This document establishes the **narrowest, most defensible novelty statement** for submission to *IEEE Transactions on Vehicular Technology (TVT)*:
Our contribution does **not** lie in proposing multimodal fusion, but in providing the first **empirical and physical critique of multimodal beam forecasting under cross-scenario distribution shift**, uncovering the **Alternative-Beam Paradox** (-25.28 pp collapse) and proving that **Candidate Exclusion (84.7%)** governs physical link outages.

---

## 2. Comprehensive Prior-Art Comparison Table

```
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                           PRIOR-ART LITERATURE COMPARISON TABLE                                                                       |
+------------------------------------+------------------+---------------------+----------+----------+----------+--------------------+-----------------+-----------------+
| Research Work                      | Dataset & Scope  | Modalities          | Horizon  | Codebook | Candidate| Evaluation Split   | Primary Metric  | Key Limitation  |
|                                    |                  |                     |          | Cardinal.| Budget K | Protocol           |                 | vs. This Work   |
+------------------------------------+------------------+---------------------+----------+----------+----------+--------------------+-----------------+-----------------+
| Alrabeiah et al. (IEEE Access 2020)| DeepSense Day/Ngt| GPS + Front RGB     | 0.0 s    | 64 beams | Top-1    | Random sample      | Top-1 Accuracy  | Static time step|
|                                    | (Single scenario)| (Single camera)     | (Nowcast)| (1 array)|          | train/test split   | (Classification)| No power metric |
+------------------------------------+------------------+---------------------+----------+----------+----------+--------------------+-----------------+-----------------+
| Charan et al. (IEEE TWC 2022)      | DeepSense V2I    | GPS + Front RGB     | 0.1 s    | 64 beams | Top-1, 5 | Trajectory split   | Top-K Accuracy, | Infrastructure  |
|                                    | (V2I tracking)   | + LiDAR Point Cloud |          | (1 array)|          | (Single scenario)  | Average Regret  | V2I, not V2V;   |
|                                    |                  |                     |          |          |          |                    | (APL in dB)     | single scenario |
+------------------------------------+------------------+---------------------+----------+----------+----------+--------------------+-----------------+-----------------+
| ITU Challenge Baseline (2022/2023) | DeepSense V2V    | GPS + Front/Rear RGB| 0.5 s    | 256 beams| Top-5    | Local blocked vs.  | Average Power   | Unmatched GPS   |
|                                    | (Scenario 36)    |                     |          | (4 arrays|          | unreleased test    | Loss (APL in dB)| baseline; only  |
|                                    |                  |                     |          | of 64)   |          |                    |                 | Scenario 36     |
+------------------------------------+------------------+---------------------+----------+----------+----------+--------------------+-----------------+-----------------+
| Fabiani et al. (IEEE VTC 2024)     | DeepSense V2V    | GPS + Front/Rear RGB| 0.5 s    | 256 beams| Top-1, 5 | Pooled across runs | Top-1, Top-5 Acc| Evaluated only  |
|                                    | (S36-S39)        | (Swin-B3 Attention) |          | (4 arrays|          | (Within-scenario)  | (Exact Hits)    | exact hit rates;|
|                                    |                  |                     |          | of 64)   |          |                    |                 | no power metrics|
+------------------------------------+------------------+---------------------+----------+----------+----------+--------------------+-----------------+-----------------+
| Mollah, Wang, Fang (GLOBECOM 2025) | DeepSense V2V    | GPS + Front/Rear RGB| 0.5 s    | 64 beams | Top-1, 5 | Within-scenario    | Top-1 Accuracy, | Evaluated only  |
| arXiv:2509.11112                   | (S36-S39)        | (Transformer Fusion)|          | (1 array)|          | random split       | Power Regret    | single array;   |
|                                    |                  |                     |          |          |          | (Severe leakage)   |                 | leaky splits    |
+------------------------------------+------------------+---------------------+----------+----------+----------+--------------------+-----------------+-----------------+
| Gao et al. (CLBP, IEEE TWC 2024)   | Synthesized mmWave| Channel State / GPS| 0.0 s    | 64 beams | Adaptive | i.i.d. random      | Conformal Risk  | Assumes i.i.d.  |
|                                    | (Simulation)     | (No RGB cameras)    |          | (1 array)| (CRC)    | exchangeable split | Coverage (1 - α)| exchangeability;|
|                                    |                  |                     |          |          |          |                    |                 | not vehicular   |
+------------------------------------+------------------+---------------------+----------+----------+----------+--------------------+-----------------+-----------------+
| THIS WORK (DeepSense-V2V Multi-Scen| DeepSense V2V    | GPS + Front/Rear RGB| 0.500 s  | 256 beams| Top-5,   | Strict zero-leakage| Top-5 3 dB Power| Resolves the    |
| TVT Submission)                    | Scenarios 36-39  | vs. Matched GPS Ctrl| (Fixed)  | (4 arrays| K=20,    | Trajectory Blocks  | Success, Alt-   | Alternative-Beam|
|                                    | (112,196 frames) | vs. Analytic Geometry|         | of 64)   | K=40     | + Leave-Scenario-Out| Beam Recovery, | Paradox and 84.7%|
|                                    |                  |                     |          |          |          | (Excluded transfer)| Exclusion Share | Exclusion Limit |
+------------------------------------+------------------+---------------------+----------+----------+----------+--------------------+-----------------+-----------------+
```

---

## 3. Detailed Literature Positioning

### 3.1 Comparison with Fabiani et al. (IEEE VTC 2024)
Fabiani et al. introduced a shifted-window temporal attention architecture (Swin-B3) combining dual-camera RGB and GPS for 0.5 s beam forecasting on DeepSense V2V.
- **What Fabiani et al. Reported:** They evaluated their model using **exact Top-1 and Top-5 classification hit rates** and compared against a simple linear GPS baseline, reporting that RGB improved classification accuracy.
- **How Our Work Advances Beyond Fabiani et al.:**
  1. **Evaluation on Physical RF Power:** We evaluate the Swin-B3 architecture on **physical 3 dB near-optimal power coverage** and **received signal power loss**, revealing that while RGB increases exact classification (+5.42 pp), it causes a catastrophic **$-19.85\text{ pp}$ collapse in 3 dB power coverage**.
  2. **Matched GPS Control:** Fabiani et al. compared a deep transformer against a linear motion model. We trained an architecturally identical transformer with zeroed visual tokens, demonstrating that deep learning itself struggles relative to kinematic extrapolation.
  3. **Strict Cross-Scenario Partitioning:** We evaluate under rigorous leave-one-scenario-out distribution shift with 10 continuous trajectory blocks, whereas prior work used within-scenario splits.

---

### 3.2 Comparison with Mollah, Wang, and Fang (arXiv:2509.11112 / GLOBECOM 2025)
Mollah et al. studied multimodal beam prediction across the same four DeepSense scenarios.
- **Critical Flaws in Mollah et al.:**
  1. **Data Leakage in Splits:** Mollah et al. performed random frame-level splits within scenarios. In continuous vehicular time-series, adjacent frames ($10\text{ ms}$ apart) share identical road scenes and channel multipath, creating extreme pseudo-replication and optimistic bias.
  2. **Reduced Array Complexity:** Mollah et al. restricted the prediction task to 64 beams (a single subarray), whereas the physical DeepSense testbed operates a 4-panel, 256-beam phased array.
- **Our Distinction:** We enforce strict, zero-leakage **trajectory-block partitioning** across all 112,196 frames and evaluate the full 256-beam codebook across all four antenna panels.

---

### 3.3 Comparison with Conformal Learning for Beam Prediction (CLBP)
CLBP (Gao et al., 2024) adapted Conformal Risk Control (CRC) to vehicular beam candidate sizing.
- **Why CLBP Fails in Real V2V:** Static conformal prediction strictly relies on **exchangeability** ($Z_1, \dots, Z_n \sim \text{i.i.d.}$). In vehicular trajectories, high temporal autocorrelation ($\rho_{\text{ICC}} \approx 0.75$) and non-stationary dynamic maneuvers (such as overtaking in Scenario 39) destroy exchangeability, inducing severe calibration lag.
- **Our Contribution:** Rather than making unrealistic theoretical claims about online conformal guarantees, we prove that **84.7% of tracking failures are driven by candidate exclusion**, which static conformal sets cannot resolve without inflating candidate set sizes beyond the frame coherence limit.

---

## 4. The Defensible Novelty Statement for IEEE TVT

The revised manuscript will make the following explicit, defensible claims:

1. **First Cross-Scenario Evaluation of Objective-Utility Mismatch:** We provide the first empirical proof that optimizing discrete Cross-Entropy loss on beam indices destroys physical mmWave link availability by collapsing alternative-beam recovery by $-25.28\text{ percentage points}$ ($p < 10^{-11}$).
2. **Decomposition of Candidate Exclusion vs. Ranking Error:** We establish that 84.7% of severe mmWave beam failures are caused by spatial candidate exclusion rather than neural reranking capacity, proving that candidate set dimensioning is the governing bottleneck.
3. **Physical Baseline Reality:** We demonstrate that an analytic kinematic line-of-sight baseline achieves 78.04% 3 dB power coverage, outperforming complex deep multimodal networks by ~30 percentage points without requiring neural inference or camera sensors.
4. **Physical-Layer Overhead Grounding:** We demonstrate analytically why candidate set widening ($K \ge 20$) degrades net effective throughput in 60 GHz vehicular transceivers due to IEEE 802.11ad/ay probing overhead.
