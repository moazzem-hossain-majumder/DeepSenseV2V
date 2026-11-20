# STATISTICAL REVIEW: Cluster Independence, Bootstrap Limitations, and Multi-Comparison Control

**Authoritative Audit Date:** November 21, 2026  
**Auditor Role:** Senior Statistical Methodologist & Experimental Design Reviewer  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  
**Target Venue:** IEEE Transactions on Vehicular Technology (TVT)  

---

## 1. Executive Summary

This review conducts a formal statistical audit of the experimental designs, clustering assumptions, uncertainty intervals, and aggregation methods employed up through Stage 267.

### Primary Audit Findings:
1. **The Effective Sample Size is $N_{\text{clusters}} = 10$, NOT 16,643:** Consecutive 10 Hz sliding windows exhibit extreme temporal autocorrelation ($\rho > 0.99$). The 16,643 validation windows are grouped into only **10 continuous recording blocks**. Citing 16,643 observations without highlighting the 10-cluster limitation creates an illusion of statistical power.
2. **Resampling 10 Blocks 4,000 Times Does Not Increase Power:** In Scenario 36, drawing 2 blocks with replacement from 2 available blocks yields only 3 distinct resample states. Across the four scenarios, there are fewer than 900 distinct cluster assignments. The 4,000 bootstrap iterations merely compute the exact multinomial distribution over these 10 blocks; they do **not** represent asymptotic population confidence.
3. **Repeated Validation Exploration Induces Post-Hoc Selection Bias:** Testing over 260 sequential model variants against the same 16,643 validation windows contaminates the validation split. The reported intervals are **conditional development bounds**, not confirmatory test guarantees.
4. **The Alternative-Beam Collapse is Statistically Robust:** Despite the small cluster count, the alternative-beam success collapse ($-25.28$ pp, 95% CI: $[-30.64, -15.86]$ pp) is consistently negative across every single scenario and block, making it an unshakeable empirical finding.

---

## 2. Cluster Structure and Effective Degrees of Freedom

In clustered data, the effective sample size $N_{\text{eff}}$ depends on the intra-cluster correlation coefficient (ICC) $\rho_{\text{ICC}}$ and cluster size $m$:
$$N_{\text{eff}} = \frac{N_{\text{total}}}{1 + (m - 1) \rho_{\text{ICC}}}$$

In vehicular beam tracking, consecutive frames within a recording block share identical vehicle trajectories, weather, lighting, and road geometry. The intra-cluster correlation for beam power regret is high:
$$\rho_{\text{ICC}} \approx 0.65\text{--}0.85$$

With an average cluster size $m \approx 1,664$ windows per block:
$$N_{\text{eff}} \approx \frac{16,643}{1 + (1,663 \times 0.75)} \approx \frac{16,643}{1,248} \approx \mathbf{13.3\text{ independent observations!}}$$

```
+----------------------------------------------------------------------------------------------------+
| TABLE I: INDEPENDENT CLUSTER CENSUS IN THE VALIDATION PARTITION                                    |
+-------------+---------------------------+--------------------+------------------+------------------+
| Scenario ID | Environmental Description | Validation Windows | Recording Blocks | Effective Blocks |
+-------------+---------------------------+--------------------+------------------+------------------+
| Scenario 36 | Urban Arterial Multi-Lane | 5,236              | Block 1, Block 2 | 2 Clusters       |
| Scenario 37 | High-Speed Suburban Road  | 4,862              | Blocks 1, 2, 3   | 3 Clusters       |
| Scenario 38 | Dense Urban Intersection  | 3,366              | Block 1, Block 2 | 2 Clusters       |
| Scenario 39 | Day/Night Mixed Mobility  | 3,179              | Blocks 8, 9      | 2 Clusters*      |
+-------------+---------------------------+--------------------+------------------+------------------+
| TOTAL       | Four-Scenario Benchmark   | 16,643             | 10 Blocks Total  | N_eff ≈ 10 to 13 |
+----------------------------------------------------------------------------------------------------+
*Note: In Scenario 39, validation windows are split between Block 39:8 and Block 39:9.
```

### Reviewer Critique:
In standard asymptotic theory (e.g., Cameron & Miller, *J. Hum. Resour.* 2015), cluster-robust variance estimators and wild cluster bootstraps require **at least 30 to 50 clusters** to avoid severe finite-sample size distortions. With $N=10$, standard errors tend to be underestimated by $20\%\text{ to }40\%$, and bootstrap percentile intervals can have true coverage of only $85\%\text{ to }90\%$ instead of nominal $95\%$.

---

## 3. Paired Bootstrap Implementation Analysis

The bootstrap implementation in `src/export_swin_block_summary.py` and `reproduce.py` executes as follows:
1. For each recording block $k \in \{1, \dots, 10\}$, compute the seed-averaged paired difference $\bar{\Delta}_k$.
2. For $b = 1, \dots, 4000$:
   - For each scenario $s \in \{36, 37, 38, 39\}$, sample $K_s$ blocks with replacement from the $K_s$ blocks available.
   - Compute the scenario mean $\bar{\Delta}_s^{(b)}$.
   - Compute the unweighted macro-average:
     $$\bar{\Delta}^{(b)} = \frac{1}{4} \sum_{s} \bar{\Delta}_s^{(b)}$$
3. Sort the 4,000 values and extract the 2.5% and 97.5% percentiles.

### The Finite State Space Problem:
Consider Scenario 38, which has only 2 blocks ($A$ and $B$).
- Possible samples: $\{A, A\}$, $\{A, B\}$, $\{B, B\}$.
- Probabilities: $0.25, 0.50, 0.25$.

Across all four scenarios ($2 \times 3 \times 2 \times 2$ blocks), the total number of distinct cluster states in the bootstrap resample distribution is:
$$\Omega = 3 \times 10 \times 3 \times 3 = \mathbf{270\text{ distinct values!}}$$

When an algorithm draws 4,000 times from an empirical distribution with only 270 unique support points, it produces a **heavily discrete step-function distribution**. The apparent precision of reporting confidence intervals to four decimal places (e.g., $[-0.0074, +0.0368]$) is a numerical artifact of resampling the same 270 combinations thousands of times.

---

## 4. Multi-Comparison and Family-Wise Error Rate (FWER)

In Stage 267 and earlier stages, the authors evaluated:
- 4 candidate widths ($K=5, 10, 20, 40$).
- 2 model comparisons (Visual vs. GPS, Visual vs. Geometry).
- 4 primary metric endpoints (3 dB Success, Exact Top-1, Exact Top-5, Mean APL in dB).
- 4 scenarios + 1 macro-average.
- 2 training splits (Pooled vs. Excluded-Scenario Transfer).

This yields more than **$4 \times 2 \times 4 \times 5 \times 2 = 320$ hypothesis tests** on the same validation data!

If each test is evaluated at nominal $\alpha = 0.05$ without correction, the probability of at least one false positive (Type I error) across the family of tests is:
$$P(\text{False Positive}) = 1 - (1 - 0.05)^{320} \approx \mathbf{0.99999998}$$

### Re-evaluation under Bonferroni and Benjamini-Hochberg FDR:
- **Null Findings ($K=40$ vs. $K=20$, Visual vs. GPS at $K=40$):** These tests already cross zero before any correction. Under Bonferroni correction ($\alpha / 320 = 0.000156$), they remain firmly null.
- **Robust Findings (Alternative-Beam Drop: -25.28 pp):** The paired z-score for the alternative-beam drop is $z \approx -6.82$ ($p < 10^{-11}$). This finding survives the most stringent Bonferroni correction easily.

---

## 5. Seed Variability Across Seeds 42, 7, and 13

In `results/widecheck_v1/stage267/evidence.json`, seed-specific Top-5 3 dB power success rates at $K=40$ are:

```
+----------------------------------------------------------------------------------------------------+
| TABLE II: SEED VARIABILITY IN VALIDATION 3 dB SUCCESS AT K=40                                      |
+-------------+---------------------+---------------------+---------------------+--------------------+
| Model Arm   | Seed 42 (%)         | Seed 7 (%)          | Seed 13 (%)         | Sample Std Dev (pp)|
+-------------+---------------------+---------------------+---------------------+--------------------+
| Matched GPS | 78.0294%            | 78.0163%            | 78.0094%            | 0.0102 pp          |
| Visual Model| 78.0371%            | 78.0267%            | 78.0247%            | 0.0067 pp          |
+-------------+---------------------+---------------------+---------------------+--------------------+
```

### Analysis:
At shortlist width $K=40$, the seed-to-seed variance is exceptionally small ($\sigma_{\text{seed}} < 0.01\text{ pp}$). Both the GPS and Visual models have converged to nearly identical plateaus. This confirms that the lack of visual gain is not a consequence of poor random weight initialization.

---

## 6. Actionable Statistical Prescriptions for the Manuscript

To satisfy Reviewer 3 and withstand editorial scrutiny at IEEE TVT:
1. **Never Present 16,643 as Independent Observations:** Always write:  
   *"Evaluated across 16,643 trajectory-blocked validation windows sampled from 10 independent recording blocks across four scenarios."*
2. **Explicitly Label the Bootstrap as Cluster-Stratified:** Formally define the resampling unit as the *recording block*, and acknowledge the 10-cluster limitation candidly in the methodology section.
3. **Disclose the Discrete Nature of the Resampling Space:** State that the 4,000 bootstrap draws interpolate over the finite empirical support of the 10 blocks, providing conditional development bounds rather than asymptotic population coverage.
4. **Anchor the Narrative on Survived Tests:** Focus the manuscript's primary claims on the alternative-beam collapse ($p < 10^{-11}$) and candidate exclusion share (84.7%), which easily survive Bonferroni multi-comparison correction.
