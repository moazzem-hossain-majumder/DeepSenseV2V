# STATISTICAL VALIDITY REVIEW: Paired Methodology, Cluster Independence, and Inference Limits

**Authoritative Audit Date:** November 21, 2026  
**Auditor Role:** Senior Statistical Evaluation Specialist & IEEE Critical Reviewer  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  
**Target Venue:** IEEE Transactions on Vehicular Technology (TVT)  

---

## 1. Executive Summary

This review provides a formal statistical audit of the experimental methodology, aggregation schemes, bootstrap implementations, and confidence interval constructions reported in `paper/V2V_BEAM_FORECASTING_MANUSCRIPT.md` and `results/swin_comparison_v1/`.

The central finding of this statistical audit is a critical distinction:
> **The empirical estimates of window-level differences are mathematically exact and fully reproducible, but the inferential confidence intervals suffer from an acute small-cluster limitation ($N_{\text{blocks}} = 10$). While 4,000 bootstrap draws were executed, resampling 10 recording blocks cannot create independent degrees of freedom. The resulting confidence intervals are conditional development diagnostic bounds, NOT population generalization guarantees.**

Furthermore, the audit confirms that the pooled-vs-transfer contrast is extraordinarily fragile: **a single scenario (Scenario 39) accounts for the entire observed difference, and within Scenario 39, just two recording sequences (374 windows) drive 77.5% of the effect.**

---

## 2. Cluster Independence and the Effective Sample Size Dilemma

### A. The 16,643 Windows Illusion
The manuscript repeatedly cites an evaluation sample of **16,643 validation windows**. In time-series and vehicular telemetry, consecutive windows sampled at 10 Hz ($\Delta t = 0.1\text{ s}$) exhibit extreme temporal autocorrelation:
- Vehicle position at time $t$ has correlation $\rho > 0.99$ with position at $t - 0.1\text{ s}$.
- The optimal beam index $b_t^*$ changes infrequently during straight-line driving, remaining identical for dozens of consecutive windows.

Therefore, treating individual windows as independent observations would result in catastrophically deflated standard errors and wildly false statistical significance.

### B. The 10-Block True Population
To avoid temporal leakage, the authors implemented **trajectory-block partitioning**, assigning entire contiguous recording blocks to either training, validation, or test. 

However, inspecting the validation partition across the four scenarios reveals the true effective sample size:

```
+----------------------------------------------------------------------------------------------------+
| Scenario ID | Environment Type          | Total Frames | Validation Windows | Validation Blocks    |
+-------------+---------------------------+--------------+--------------------+----------------------+
| Scenario 36 | Urban Arterial Multi-Lane | 24,799       | 5,236              | 2 Recording Blocks   |
| Scenario 37 | High-Speed Suburban Road  | 30,999       | 4,862              | 3 Recording Blocks   |
| Scenario 38 | Dense Urban Intersection  | 35,999       | 3,366              | 2 Recording Blocks   |
| Scenario 39 | Day/Night Mixed Mobility  | 20,399       | 3,179              | 3 Recording Blocks*  |
+-------------+---------------------------+--------------+--------------------+----------------------+
| TOTAL       | Four-Scenario Pool        | 112,196      | 16,643             | 10 RECORDING BLOCKS  |
+----------------------------------------------------------------------------------------------------+
*Note: In Scenario 39, validation is heavily concentrated in blocks 39:8 and 39:9.
```

The true effective degrees of freedom for statistical inference is **$N_{\text{clusters}} = 10$**, not 16,643. In standard econometric and biostatistical cluster-sampling theory, cluster-robust standard errors and cluster bootstrap procedures require at least 30 to 50 independent clusters to provide asymptotically valid coverage. With only 10 clusters, bootstrap percentile intervals are inherently volatile and sensitive to the inclusion or exclusion of individual blocks.

---

## 3. Audit of the Bootstrap Implementation

### A. Resampling Protocol
The resampling procedure implemented in `results/swin_comparison_v1/stage299/reproduce.py` and `src/export_swin_block_summary.py` executes as follows:
1. Precompute the window-paired metric differences for each of the 10 recording blocks, averaged across seeds 42, 7, and 13.
2. In each of $B = 4,000$ iterations:
   - For each scenario $s \in \{36, 37, 38, 39\}$, draw $K_s$ recording blocks *with replacement* from the available blocks in scenario $s$ ($K_{36}=2, K_{37}=3, K_{38}=2, K_{39}=3$).
   - Compute the scenario mean for each scenario.
   - Compute the unweighted macro-average across the four scenarios:
     $$\bar{\Delta}^{(b)} = \frac{1}{4} \sum_{s \in \{36,37,38,39\}} \bar{\Delta}_s^{(b)}$$
3. Sort the 4,000 resampled macro-averages and take the 2.5th and 97.5th percentiles to form the 95% confidence interval $[\hat{q}_{0.025}, \hat{q}_{0.975}]$.

### B. Statistical Critique of the Resampling Space
Consider Scenario 36, which has only **$K_{36} = 2$ blocks** ($A$ and $B$). When sampling 2 blocks with replacement from $\{A, B\}$, there are only three possible outcomes:
- $\{A, A\}$ with probability $0.25$
- $\{A, B\}$ with probability $0.50$
- $\{B, B\}$ with probability $0.25$

Across the entire dataset, the total number of distinct cluster assignment configurations is:
$$\Omega = 3 \times 10 \times 3 \times 10 = 900 \text{ distinct resample states}$$

Executing $B = 4,000$ iterations over an empirical space with fewer than 900 distinct states produces repeated draws of the exact same cluster combinations. **The 4,000 iterations merely interpolate the discrete multinomial distribution of the 10 blocks; they do NOT compensate for the small number of physical driving runs.**

---

## 4. Sensitivity Analysis: The Scenario 39 Anomaly

The most critical statistical vulnerability uncovered by our audit is the extraordinary sensitivity of the results to **Scenario 39**, documented in `results/scenario39_contributions_v1/REPORT.md` and `results/variation_v1/stage220/REPORT.md`:

### A. Scenario-Omission Jackknife
Evaluating the pooled-minus-transfer contrast for the regularized 20-candidate model (+0.717 pp) under leave-one-scenario-out aggregation:

```
+-----------------------------------------+---------------------------------+
| Aggregation Condition                   | Pooled - Transfer Contrast (pp) |
+-----------------------------------------+---------------------------------+
| Full 4-Scenario Macro Average           | +0.717 pp                       |
| Omit Scenario 36 (Evaluate 37, 38, 39)  | +0.841 pp                       |
| Omit Scenario 37 (Evaluate 36, 38, 39)  | +0.981 pp                       |
| Omit Scenario 38 (Evaluate 36, 37, 39)  | +1.005 pp                       |
| Omit Scenario 39 (Evaluate 36, 37, 38)  | +0.040 pp (COLLAPSED TO ZERO!)  |
+-----------------------------------------+---------------------------------+
```

When Scenario 39 is removed from the analysis, **the entire pooled-vs-transfer contrast completely disappears (+0.040 pp).**

### B. Sequence-Level Decomposition within Scenario 39
Drilling down into the 17 sequences of Scenario 39:
- Scenario 39 contains 3,179 validation windows across 17 distinct continuous sequences.
- Just **two sequences**—Sequence 50 (187 windows) and Sequence 51 (187 windows), totaling only 374 windows (2.2% of the validation set)—account for **77.5% of the total net difference**.
- In Sequence 50, the transfer visual model matched geometry.
- In Sequence 51, the transfer visual model suffered a catastrophic plunge, scoring **11.41 percentage points below geometry**.
- Within block 39:8, the contrast was $+4.343$ pp. Within block 39:9, the contrast was $-0.178$ pp.

**Conclusion:** The manuscript cannot claim that "cross-scenario transfer consistently degrades visual fusion." The observed transfer degradation is heavily driven by a single vehicle maneuver during Sequence 51 in Scenario 39. Claiming a general vehicular distribution shift without explicitly disclosing this sequence-level sensitivity constitutes an overstatement of statistical evidence.

---

## 5. Audit of Paired Seed Initialization

The experimental design paired model predictions across three random seeds: $\mathcal{S} = \{42, 7, 13\}$.
- All models were initialized with independent random weights for the projection heads and trained for 12 epochs under identical schedules.
- Prediction differences were computed on a per-window basis before averaging across seeds:
  $$\Delta_i = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \left( M_{\text{RGB}}(i; s) - M_{\text{GPS}}(i; s) \right)$$
- **Audit Assessment: SOUND.** Pairing by window and averaging across identical seed seeds eliminates stochastic weight-initialization variance from the contrast estimate. However, $S = 3$ seeds is the bare minimum for seed robustness; reporting sample standard deviations across seeds remains necessary.

---

## 6. Selection Bias from Extensive Prior Validation

The repository records 299 development stages spanning multiple weeks:
- Stages 8–36 tested power-aware losses.
- Stages 44–61 tested bounded residual tanh corrections.
- Stages 81–95 tested front vs. rear cameras and linear heads.
- Stages 96–116 tested K=10 and K=20 candidate reranking.
- Stages 124–154 tested resolution (96px vs 192px).
- Stages 164–189 tested visual dropout and KL consistency.
- Stages 240–267 tested candidate width K=20 vs K=40.
- Stages 268–299 tested the Swin-B3 adaptation.

**Statistical Impact:**
All of these 299 stages were evaluated on the **exact same 16,643 validation windows across Scenarios 36–39**. 

When researchers repeatedly inspect validation results and formulate new hypotheses to explain negative outcomes, the validation set is no longer an unbiased, untouched sample. It becomes contaminated by **informal post-hoc selection bias**.

The manuscript honestly acknowledges this limitation in Section 6:
> *"Four repeatedly examined scenarios support a development study. Overlapping windows, few independent recording blocks and selection of earlier model variants limit inference to new routes."*

This acknowledgment is commendable, but an IEEE reviewer will demand that the authors maintain a strict separation between **exploratory development diagnostics** and **confirmatory generalization claims**.

---

## 7. Authoritative Statistical Summary Table

```
+-------------------------------------------------------------+------------------------------------+-----------------------------+
| Statistical Dimension                                       | Observed Implementation            | Scientific Assessment       |
+-------------------------------------------------------------+------------------------------------+-----------------------------+
| Unit of Evaluation                                          | 16,643 sliding windows (10 Hz)     | High autocorrelation; not   |
|                                                             |                                    | independent observations.   |
+-------------------------------------------------------------+------------------------------------+-----------------------------+
| True Independent Clusters                                   | 10 contiguous recording blocks     | Severely constrained ($N=10$|
|                                                             |                                    | limits asymptotic validity).|
+-------------------------------------------------------------+------------------------------------+-----------------------------+
| Seed Pairing                                                | 3 matched seeds (42, 7, 13)        | Correctly paired; minimal   |
|                                                             |                                    | sample size ($S=3$).        |
+-------------------------------------------------------------+------------------------------------+-----------------------------+
| Resampling Procedure                                        | 4,000 block-bootstrap draws        | Exact interpolation over    |
|                                                             |                                    | $<900$ discrete states.     |
+-------------------------------------------------------------+------------------------------------+-----------------------------+
| Sensitivity to Scenario 39                                  | S39 accounts for 100% of contrast; | High risk of over-fitting to|
|                                                             | seq 50-51 account for 77.5%.       | localized driving anomaly.  |
+-------------------------------------------------------------+------------------------------------+-----------------------------+
| Cross-Scenario Contrast                                     | -1.39 pp (95% CI: [-13.01, +8.49]) | Statistically inconclusive; |
| (Pooled vs. Transfer)                                       | Interval widely spans zero.        | null hypothesis not rejected|
+-------------------------------------------------------------+------------------------------------+-----------------------------+
| Alternative-Beam Collapse                                   | -25.28 pp (95% CI: [-30.64,-15.86])| Statistically robust;       |
| (RGB vs. GPS)                                               | Interval strictly below zero.      | effect is unmistakable.     |
+-------------------------------------------------------------+------------------------------------+-----------------------------+
| Reserved Test Partition Status                              | 17,017 frames strictly unopened    | Zero contamination; pristine|
|                                                             |                                    | benchmark integrity.        |
+-------------------------------------------------------------+------------------------------------+-----------------------------+
```

---

## 8. Prescriptive Recommendations for the Manuscript

1. **Retitle Section 3 Methodology:** Formally designate the evaluation as a **Cluster-Stratified Trajectory-Block Bootstrap**, explicitly defining the 10 recording blocks as the resampling unit.
2. **Prominently Disclose the Effective Sample Size:** Directly state in the abstract and Section 3 that the validation partition comprises 10 independent recording blocks, preventing reviewers from accusing the paper of inflating sample size through 16,643 correlated windows.
3. **Report the Scenario 39 Jackknife in Section 5:** Present the leave-one-scenario-out sensitivity analysis transparently. Showing that the pooled-transfer contrast drops to $+0.040$ pp without Scenario 39 demonstrates exceptional scientific maturity and preempts a fatal reviewer critique.
4. **Frame the Alternative-Beam Finding as the Headline Result:** While the cross-scenario transfer contrast is statistically inconclusive ($[-13.01, +8.49]$), the **alternative-beam collapse ($-25.28$ pp, 95% CI: $[-30.64, -15.86]$) is statistically rock-solid across all four scenarios**. The paper should center its narrative entirely around this robust finding.
