# RESOLUTION REPLICATION PLAN: Controlled Protocol for 96px vs. 192px Evaluation (Stage 155+)

**Authoritative Plan Date:** November 21, 2026  
**Auditor Role:** Senior IEEE Experimental Auditor & Systems Architect  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  
**Target Stages:** Proposed Stages 155–163 (Replication of Native Resolution Features)  

---

## 1. Executive Rationale: Is Resolution Replication Scientifically Justified?

### The Decision: **CONDITIONALLY JUSTIFIED AS A BOUNDED FALSIFICATION STUDY**
In Stage 154, evaluating 192-pixel native features against 96-pixel features under Seed 42 yielded an apparent gain of **+0.3086 percentage points** (79.5434% vs. 79.2348%), but its paired trajectory-block confidence interval crossed zero ($[-0.2890, +0.7763]$ pp).

Replicating this comparison across two additional random seeds (Seeds 7 and 13) with strictly matched GPS controls is scientifically justified **only** to settle the resolution question definitively:
- If Seeds 7 and 13 fail to show a statistically significant positive lower bound, the hypothesis that higher spatial resolution improves V2V beam forecasting is definitively falsified.
- If Seeds 7 and 13 replicate the gain with a strictly positive paired confidence interval, the finding is admitted as a validated development diagnostic.

Under no circumstances should this replication be used as an open-ended hyperparameter tuning exercise.

---

## 2. Pre-Declared Scientific Hypotheses

1. **Null Hypothesis ($H_0$):**  
   Doubling native spatial resolution from 96 to 192 pixels produces zero true difference in three-seed macro-average Top-5 3 dB power success:
   $$\mathbb{E}[\Delta_{\text{192 - 96}}] = 0$$
   *Criterion for Failure to Reject:* The 95% paired trajectory-block bootstrap confidence interval for $\bar{\Delta}_{\text{192 - 96}}$ includes zero.

2. **Alternative Hypothesis ($H_1$):**  
   192-pixel features consistently improve spatial vehicle localization, yielding a three-seed mean gain $\bar{\Delta} \ge +0.25$ percentage points over 96-pixel features, and the 95% paired trajectory-block bootstrap lower bound is strictly greater than zero ($\hat{q}_{0.025} > 0$).

---

## 3. Strict Experimental Design and Protocol Matrix

To ensure absolute comparability, all experimental variables must remain strictly locked across both resolution arms and matched controls:

```
+---------------------------------------------------------------------------------------------------------------------------------+
| TABLE I: EXPERIMENTAL PROTOCOL SPECIFICATION (STAGES 155–163)                                                                   |
+--------------------------+------------------------------------------------------------------------------------------------------+
| Dimension                | Prescribed Value & Implementation Rule                                                               |
+--------------------------+------------------------------------------------------------------------------------------------------+
| Dataset Scope            | DeepSense 6G Scenarios 36, 37, 38, 39 (112,196 frames total).                                        |
| Evaluation Windows       | 54,604 training windows; 16,643 validation windows (trajectory-blocked, stride=2, horizon=0.5 s).     |
| Partitioning Rule        | Zero-leakage trajectory blocks (train/val/cal/test strictly isolated).                               |
| Calibration & Test Set   | STRICTLY LOCKED & UNOPENED (Zero reads permitted).                                                   |
| Tested Random Seeds      | Seed 7 and Seed 13 (Seed 42 completed in Stages 152/153 and frozen).                                 |
| Visual Encoders          | ResNet-18 (ImageNet pretrained, frozen convolutional layers, global average pooling).                |
| Feature Caches           | `data/processed/resolution_v1/s{scenario}_96.npy` and `s{scenario}_192.npy` (verified sha256).      |
| Input Representation     | Side-by-side concatenation of Front (RGB5) and Rear (RGB6) camera views.                             |
| Augmentations            | ZERO brightness augmentation (pure deterministic feature evaluation).                                |
| Scoring Architecture     | Unbounded hard-objective candidate reranking head over K=20 geometry candidates.                     |
| Loss Function            | Hard candidate cross-entropy + expected-regret coefficient (0.05) + ZERO residual L2 penalty.        |
| Optimizer                | AdamW (Initial LR: 1e-4, Cosine decay to 1e-5, Weight decay: 1e-3).                                   |
| Batch Size & Budget      | Batch size 16, exactly 12 full training epochs per run.                                              |
| Matched GPS Control      | Zeroed visual feature tokens with identical projection head and optimizer budget.                    |
| Primary Metric           | Macro-average Top-5 received power success within 3 dB of optimum (equal scenario weighting).         |
| Secondary Metrics        | Top-1 exact accuracy, Top-5 exact accuracy, Average Power Loss (APL in dB).                         |
| Uncertainty Evaluation   | 4,000 paired scenario-stratified trajectory-block bootstrap resamples.                               |
+--------------------------+------------------------------------------------------------------------------------------------------+
```

---

## 4. Execution Sequence: Stages 155 through 163

The execution queue consists of exactly seven training jobs and one reporting stage:

```
+--------------------------------------------------------------------------------------------------------------------+
| TABLE II: STAGE-BY-STAGE EXECUTION QUEUE                                                                           |
+-------+-----------+---------+---------------+------------+---------------------------------------------------------+
| Stage | Target    | Seed    | Input Modality| Resolution | Description & Verification Target                       |
+-------+-----------+---------+---------------+------------+---------------------------------------------------------+
| 155   | Preflight | N/A     | N/A           | N/A        | Verify feature cache integrity (receipt.json) & tests.  |
| 156   | Train     | Seed 7  | GPS Control   | N/A        | Matched GPS baseline (zeroed visual tokens, 12 epochs). |
| 157   | Train     | Seed 7  | RGB + GPS     | 96 pixels  | 96px visual model (12 epochs, cosine decay).            |
| 158   | Train     | Seed 7  | RGB + GPS     | 192 pixels | 192px visual model (12 epochs, cosine decay).           |
| 159   | Train     | Seed 13 | GPS Control   | N/A        | Matched GPS baseline (zeroed visual tokens, 12 epochs). |
| 160   | Train     | Seed 13 | RGB + GPS     | 96 pixels  | 96px visual model (12 epochs, cosine decay).            |
| 161   | Train     | Seed 13 | RGB + GPS     | 192 pixels | 192px visual model (12 epochs, cosine decay).           |
| 162   | Harvest   | All     | All           | Both       | Aggregate saved validation predictions across 3 seeds.  |
| 163   | Report    | All     | All           | Both       | Compute 4,000 block-bootstrap CIs; generate REPORT.md.  |
+-------+-----------+---------+---------------+------------+---------------------------------------------------------+
```

---

## 5. Pre-Declared Stopping Rules and Decision Gates

To prevent open-ended experimentation, the research team pre-declares the following decision rules:

```
                                    STAGE 163 DECISION GATE
                                               |
                  +----------------------------+----------------------------+
                  |                                                         |
        [Condition A: CI Lower Bound > 0]                         [Condition B: CI Includes Zero]
        Mean gain >= +0.25 pp AND                                 CI crosses zero OR
        Bootstrap lower bound > 0.00 pp                           Mean gain < +0.25 pp
                  |                                                         |
                  v                                                         v
        [DECISION: ACCEPTED]                                      [DECISION: FALSIFIED]
        • Resolution gain confirmed.                             • Higher resolution provides NO benefit.
        • Proceed to cross-scenario                               • Permanently freeze visual resolution
          transfer audit (Stages 164+).                             at 96 pixels (or drop vision entirely).
        • Report as positive development                          • Reframe paper around candidate
          diagnostic.                                               exclusion and alternative-beam trade-off.
```

---

## 6. Computational Resource Budget

- **Feature Caching:** The 96px and 192px features are already pre-extracted in `data/processed/resolution_v1/` (~0.5 GB). Zero image decoding is required during training.
- **Runtime Estimate:** Each 12-epoch training run on cached features requires approximately **3.5 minutes** on the NVIDIA RTX 4060 GPU.
- **Total Compute Duration:** 6 training runs $\times 3.5\text{ min} \approx \mathbf{21\text{ to }25\text{ minutes}}$ total execution time.
- **Risk Assessment:** Extremely low risk. Local disk space is abundant (>130 GB free), memory footprint is $<2\text{ GB}$, and the job runs fully offline.
