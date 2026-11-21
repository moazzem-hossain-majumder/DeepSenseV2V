# FINAL STATISTICAL PROTOCOL & BENCHMARK METHODOLOGY

**Document Path:** `AUDIT/FINAL_STATISTICAL_PROTOCOL.md`  
**Authoritative Date:** November 21, 2026  
**Auditor:** Senior Statistical Evaluation Specialist & Research Methodology Auditor  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  

---

## 1. Executive Summary

This protocol establishes the definitive, mathematically rigorous statistical evaluation standards for the DeepSense-V2V research project. 

It explicitly addresses the **pseudo-replication constraint** arising from temporally correlated sliding windows, defines the **paired trajectory-block bootstrap procedure**, formalizes all communication and accuracy metrics mathematically, accounts for **model-selection bias on the development partition**, and enforces the **strict lock on the 17,017-frame test partition**.

---

## 2. Statistical Degrees of Freedom & Pseudo-Replication

### 2.1 The Nominal vs. Effective Sample Size Problem
The dataset comprises 16,643 sliding validation windows generated from synchronized telemetry at 0.1 s steps. However, adjacent windows share overlapping histories (1.0 s observation span) and targets (0.5 s prediction horizon). 

These 16,643 windows originate from only **$G = 10$ continuous vehicle recording blocks** across four DeepSense scenarios:
- **Scenario 36:** 2 continuous recording blocks (5,236 windows)
- **Scenario 37:** 3 continuous recording blocks (4,862 windows)
- **Scenario 38:** 2 continuous recording blocks (3,366 windows)
- **Scenario 39:** 3 continuous recording blocks (3,179 windows)

#### Intra-Cluster Correlation and Effective Sample Size:
With an estimated intra-cluster correlation coefficient $\rho_{\text{ICC}} \approx 0.75$ and average cluster size $\bar{m} = 1,664.3$ windows, the cluster design effect (variance inflation factor) is:
$$\text{VIF} = 1 + (\bar{m} - 1)\rho_{\text{ICC}} \approx 1 + (1663.3 \times 0.75) \approx 1,248.5$$
The effective independent sample size is:
$$N_{\text{eff}} = \frac{N}{\text{VIF}} = \frac{16,643}{1,248.5} \approx \mathbf{13.3\text{ independent observations}}$$

**Mandatory Reporting Rule:**  
Authors must never report $N = 16,643$ as the number of independent samples for statistical tests. In all IEEE tables and manuscripts, the sample size must be reported as:
$$\text{Sample Size: } 16,643\text{ sliding windows clustered across } G = 10\text{ trajectory blocks in } S = 4\text{ scenarios}$$

---

### 2.2 Paired Trajectory-Block Bootstrap Specification
To compute confidence intervals that accurately reflect cluster-level variability without violating independence:

1. **Resampling Unit:** The resampling unit is the **continuous vehicle trajectory block** (identified by `block_key`), **never individual sliding windows**.
2. **Stratification:** Resampling is stratified within each scenario. If Scenario $s$ has $B_s$ blocks, exactly $B_s$ blocks are sampled with replacement in each bootstrap replicate $b \in \{1, \dots, B\}$.
3. **Paired Pairing:** All comparative contrasts ($\Delta = \text{Model}_A - \text{Model}_B$) must be computed at the window level on identical window instances, using identical seeds and initializations, before block aggregation:
   $$\Delta_{b} = \frac{1}{4} \sum_{s \in \{36,37,38,39\}} \left[ \frac{1}{\sum_{i \in \mathcal{B}_b^{(s)}} |\mathcal{W}_i|} \sum_{i \in \mathcal{B}_b^{(s)}} \sum_{w \in \mathcal{W}_i} (Y_{w, A} - Y_{w, B}) \right]$$
4. **Bootstrap Replicates:** $B = 4,000$ paired resamples.
5. **Confidence Interval Derivation:** 95% two-sided percentile interval: $[\Delta_{(0.025)}, \Delta_{(0.975)}]$.

---

## 3. Mathematical Metric Definitions

For a given window $w$ with true 256-beam power vector $\mathbf{P}_w = [P_{w, 1}, \dots, P_{w, 256}] \in \mathbb{R}^{256}$ (in dBm), optimal beam index $y_w^* = \arg\max_{b} P_{w, b}$, and predicted candidate shortlist $\mathcal{C}_{w, K} = \{c_{w, 1}, \dots, c_{w, K}\}$ of size $K$:

### 3.1 Primary Metric: Macro Top-K 3 dB Power Success Rate
The power loss incurred by the candidate set is:
$$L_w(\mathcal{C}_{w, K}) = \max_{b \in \{1,\dots,256\}} P_{w, b} - \max_{c \in \mathcal{C}_{w, K}} P_{w, c} \ge 0\text{ dB}$$
A window is successful within 3 dB if:
$$\mathbb{I}_{w,\text{succ}} = \mathbb{I}\left( L_w(\mathcal{C}_{w, K}) \le 3.0\text{ dB} \right)$$
The macro-averaged metric gives equal weight to all four scenarios:
$$\text{Success}_{\text{macro}}(K) = \frac{1}{4} \sum_{s \in \{36,37,38,39\}} \frac{1}{|\mathcal{W}_s|} \sum_{w \in \mathcal{W}_s} \mathbb{I}_{w,\text{succ}}$$

---

### 3.2 Secondary Metrics

1. **Top-K Exact Classification Hit Rate:**
   $$\mathbb{I}_{w,\text{hit}} = \mathbb{I}\left( y_w^* \in \mathcal{C}_{w, K} \right)$$
2. **Alternative-Beam Power Success Rate (Resilience):**
   A window succeeds via an alternative beam if it delivers within 3 dB of optimal power despite missing the labeled argmax beam:
   $$\mathbb{I}_{w,\text{alt}} = \mathbb{I}_{w,\text{succ}} \land (\neg \mathbb{I}_{w,\text{hit}})$$
   *(Note: $\mathbb{I}_{w,\text{succ}} = \mathbb{I}_{w,\text{hit}} + \mathbb{I}_{w,\text{alt}}$, forming three mutually exclusive outcomes: Exact Hit, Alternative Success, and Outage).*
3. **Average Power Loss (APL in dB / Regret):**
   $$\text{APL}_{\text{macro}} = \frac{1}{4} \sum_{s=36}^{39} \frac{1}{|\mathcal{W}_s|} \sum_{w \in \mathcal{W}_s} L_w(\mathcal{C}_{w, K})$$
4. **Severe Miss Rate (>10 dB Outage):**
   $$\text{SevereMiss}_{\text{macro}} = \frac{1}{4} \sum_{s=36}^{39} \frac{1}{|\mathcal{W}_s|} \sum_{w \in \mathcal{W}_s} \mathbb{I}\left( L_w(\mathcal{C}_{w, K}) > 10.0\text{ dB} \right)$$
5. **Candidate Exclusion Share ($E_{\text{cand}}$):**
   The fraction of severe misses where every candidate in the set already exceeds 10 dB loss:
   $$E_{\text{cand}} = \frac{\sum_{w} \mathbb{I}\left( \min_{c \in \mathcal{C}_{w, K}} [P_{w, y^*} - P_{w, c}] > 10.0\text{ dB} \right)}{\sum_{w} \mathbb{I}\left( L_w(\mathcal{C}_{w, K}) > 10.0\text{ dB} \right)}$$
6. **Effective Communication Throughput ($\eta_{\text{eff}}$):**
   $$\eta_{\text{eff}}(K) = \left( 1 - \frac{K \cdot T_{\text{probe}} + T_{\text{proc}} + T_{\text{feedback}}}{T_{\text{frame}}} \right) \cdot R_{\text{max}} \cdot \text{Success}_{\text{macro}}(K)$$

---

## 4. Partition Integrity & Test Partition Governance

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                              DATASET SPLIT INTEGRITY MANIFEST                                            |
+--------------------------+-----------------------+---------------------+-------------------+----------------------------+
| Partition                | Number of Windows     | Number of Raw Frames| Number of Blocks  | Access Status Across S1-299|
+--------------------------+-----------------------+---------------------+-------------------+----------------------------+
| Training                 | 54,604                | 58,544              | 28 blocks         | Standard Fitting           |
| Validation               | 16,643                | 16,643              | 10 blocks         | Audited (Selection Bias)   |
| Calibration (Reserved)   | 14,025                | 14,025              | 8 blocks          | STRICTLY UNTOUCHED (0 READS|
| Test (Locked)            | 19,631                | 17,017              | 12 blocks         | STRICTLY UNTOUCHED (0 READS|
+--------------------------+-----------------------+---------------------+-------------------+----------------------------+
Total Validated Pipeline: 104,903 Windows (112,196 Raw Rows across Scenarios 36, 37, 38, and 39).
```

### 4.1 Development Model-Selection Bias Disclosure
Because the 16,643 validation windows were examined across Stages 1–299 to compare architectures, hyperparameters, candidate widths, and objectives, **the validation set exhibits cumulative selection bias**. 

**Mandatory Methodological Disclosure for Manuscript Section III:**
> *"All comparative results reported in this paper condition on the observed development validation trajectories. Because validation data were utilized for architectural screening and hyperparameter tuning across development stages, reported confidence intervals reflect uncertainty over observed trajectory blocks rather than independent generalization to unobserved routes."*

### 4.2 Locked Test Set Policy
The 17,017 raw frames (19,631 sliding windows) reserved for final testing must remain **strictly locked and unopened**. 

Under no circumstances should any script execute on the test partition until:
1. All manuscript text, hypotheses, figures, and reviewer responses are frozen.
2. The author explicitly authorizes a single, locked confirmatory pass.
3. No hyperparameter tuning or architecture adjustment is permitted post-test.
