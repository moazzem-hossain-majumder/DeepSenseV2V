# IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY: PEER REVIEW REPORT

**Reviewer Profile:** Reviewer 3 (Senior Specialist in Statistical Experimental Design, Applied Econometrics, and Research Reproducibility)  
**Manuscript Title:** *Does Camera Input Improve the Power of a Five-Beam Shortlist? A Paired Study of Vehicle-to-Vehicle Beam Forecasting*  
**Tracking Number:** TVT-2026-AUDIT-R3  
**Recommendation:** **MAJOR REVISION**  
**Overall Score:** **5.5 / 10**

---

## 1. Summary of the Contribution

The manuscript presents an empirical evaluation of multimodal beam forecasting on the DeepSense-V2V dataset across Scenarios 36–39. The authors use a paired window-level experimental design comparing GPS and GPS+RGB models across both pooled and leave-one-scenario-out transfer splits. The paper reports paired difference estimates and 95% block-bootstrap confidence intervals for Top-5 exact accuracy and 3 dB power success. The authors conclude that adding cameras reduces alternative-beam recovery by 25.28 percentage points, and that candidate exclusion accounts for 84.7% of severe power misses in a geometry-constrained model.

---

## 2. Strengths

1. **Window-Level Paired Design:** Evaluating models on the exact same sequence windows and averaging paired differences across identical seeds ($\Delta_i = M_{\text{RGB}}(i) - M_{\text{GPS}}(i)$) eliminates sample-composition variance.
2. **Recognition of Temporal Autocorrelation:** The authors correctly rejected random frame splitting and implemented trajectory-block partitioning to prevent adjacent-frame leakage.
3. **Pristine Reproducibility Artifacts:** The accompanying review package and reproduction scripts (`reproduce.py`) allow every single reported number and interval to be recomputed deterministically from block summaries in seconds.
4. **Candid Acknowledgment of Limitations:** The authors explicitly disclose that their intervals condition on four observed scenarios and that each scenario contributed only two or three validation blocks.

---

## 3. Major Concerns (Fatal Methodological Flaws if Not Addressed)

1. **The Small-Cluster Pseudo-Replication Fallacy:**  
   The manuscript repeatedly highlights an evaluation sample of **"16,643 trajectory-blocked validation windows"** and **"4,000 paired resamples"**. This creates a deeply misleading impression of statistical power.  
   The true units of independent sampling are the continuous **recording blocks**, not the 10 Hz sliding windows. In the validation partition, there are only:
   - Scenario 36: 2 blocks
   - Scenario 37: 3 blocks
   - Scenario 38: 2 blocks
   - Scenario 39: 3 blocks (effectively 2 primary blocks)  
   **Total independent clusters: $N = 10$.**  
   In statistical theory, a cluster bootstrap with $N=10$ cannot achieve nominal 95% coverage; empirical coverage is typically distorted, and intervals are highly unstable. Resampling 10 blocks 4,000 times merely draws from a finite set of fewer than 900 discrete cluster combinations. The authors must stop headlining "16,643 windows" and explicitly state that the effective sample size is 10 recording blocks.
2. **Extreme Sensitivity to a Single Driving Anomaly in Scenario 39:**  
   A rigorous sensitivity audit reveals that the paper's pooled-vs-transfer contrast (+0.717 pp) is an artifact of **Scenario 39**:
   - If Scenario 39 is omitted from the macro-average, the contrast **collapses from $+0.717$ pp to $+0.040$ pp**.
   - Within Scenario 39, just two recording sequences (Sequences 50 and 51, containing only 374 windows) drive **77.5% of the total difference**.
   - In Sequence 51, the visual model crashed 11.41 percentage points below geometry.  
   Drawing broad scientific conclusions about "cross-scenario generalization" when the entire statistical effect is driven by 374 frames in one driving sequence is unacceptable in an IEEE Transactions paper. This sensitivity must be prominently analyzed in the main text.
3. **Severe Selection Bias from 299 Prior Development Stages:**  
   The repository history indicates that the validation partition of these four scenarios was probed, diagnosed, and iterated upon across nearly 300 sequential stages over several weeks. When researchers formulate new hypotheses and model adjustments after viewing validation failures, the validation partition ceases to be an independent holdout. It becomes an exploratory training set. The confidence intervals reported in the paper are conditional development bounds, not population generalization intervals. The authors must clarify this distinction unambiguously.

---

## 4. Minor Concerns

1. **Percentile vs. BCa Bootstrap Intervals:** The authors use simple percentile intervals from the empirical bootstrap distribution. Given the extreme skewness and small cluster count ($N=10$), an adjusted bootstrap (e.g. bias-corrected and accelerated [BCa]) or a Wild Cluster Bootstrap would be more theoretically appropriate.
2. **Multiple Comparisons without FWER Control:** The authors evaluate multiple metrics (Top-1, Top-5, 3 dB success, APL, Top-1 loss, Best-of-5 loss) across two splits and four scenarios without adjusting for family-wise error rate (FWER) or false discovery rate (FDR).

---

## 5. Missing Statistical Analyses

1. **Leave-One-Block-Out Jackknife Analysis:** Provide a complete table showing the point estimate and interval when each of the 10 recording blocks is systematically left out.
2. **Permutation Test against the Null Hypothesis:** Conduct a formal block-level permutation test (swapping RGB and GPS labels within blocks) to compute an exact non-parametric p-value for the alternative-beam success difference ($-25.28$ pp).

---

## 6. Novelty Assessment

From a statistical methodology perspective, the paper's strength is its **rigorous paired contrast framework and transparent error decomposition**. However, the statistical power is severely constrained by the small number of available scenarios and recording blocks.

---

## 7. Reviewer Recommendation and Concluding Remarks

**Recommendation:** **MAJOR REVISION.**  
The empirical calculations are correct and fully reproducible, but the statistical narrative overstates the generalizability of the findings by conflating 16,643 autocorrelated windows with independent evidence. The authors must reframe the manuscript to honestly confront the 10-block limitation, report the Scenario 39 sensitivity jackknife, and state clearly that these results represent exploratory development findings rather than universal vehicular distribution laws.
