# Scientifically Rigorous Experimental Improvement Plan

**Repository:** [moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  
**Lead Scientist:** Senior IEEE Research Scientist & ML Systems Engineer  
**Date:** November 2026  
**Status:** Pre-Execution Experimental Protocol (Phase 4 Deliverable)

---

## 1. Executive Statement and Scientific Objectives

The comprehensive repository audits (Phases 1–3) revealed four critical methodological bottlenecks in the current implementation:
1. **The Deep Learning Deficit:** The multimodal neural network achieves only $\sim 49\text{--}51\%$ 3 dB power success, losing to a simple classical GPS vector geometry baseline (**78.56%**) by over 27 percentage points.
2. **Visual Modality Collapse:** Full-frame global average pooling destroys vehicle spatial coordinates, rendering RGB representations insensitive to scene content ($\Delta < 0.12\%$ on shuffle/gray tests).
3. **Loss Function Dysfunction:** Unweighted MSE over 256 beams expends $97\%$ of its capacity fitting noise-floor sidelobes, while smoothness regularization penalizes legitimate transitions across physical 64-beam antenna panels.
4. **The Alternative-Beam Trade-Off:** Categorical cross-entropy sharpens predictions on the exact peak beam at the expense of broad, high-power main-lobe coverage, causing a 19.85 percentage-point collapse in 3 dB power success.

**The Objective of this Plan:** Rather than engaging in ad-hoc hyperparameter tuning, we design a mathematically principled, physically anchored research protocol that directly rectifies these four failure mechanisms and establishes a publication-grade submission for IEEE Transactions on Vehicular Technology (TVT) or IEEE Transactions on Wireless Communications (TWC).

---

## 2. Testable Scientific Hypotheses

### Hypothesis 1: Physical Geometry Prior with Bounded Residual Learning
* **Null Hypothesis ($H_{0,1}$):** A learned multimodal residual network added to the analytic geometry prior does not improve 3 dB top-5 power coverage over analytic geometry alone on unseen vehicular scenarios.
* **Alternative Hypothesis ($H_{1,1}$):** Structuring the neural network to predict a bounded residual angular offset $\Delta \theta \in [-\Delta \theta_{\text{max}}, +\Delta \theta_{\text{max}}]$ and residual power weights over the geometry-generated candidate set strictly dominates pure geometry and pure end-to-end deep learning:
  $$\mathbb{P}(L(\mathcal{A}_{\text{Residual}}) \le 3\text{ dB}) > \mathbb{P}(L(\mathcal{A}_{\text{Geometry}}) \le 3\text{ dB})$$
  with a paired block-bootstrap 95% confidence interval strictly greater than zero.

### Hypothesis 2: Heading-Directed Spatial Cropping vs. Global Pooling
* **Null Hypothesis ($H_{0,2}$):** Heading-directed spatial Region-of-Interest (ROI) visual feature extraction provides no statistically significant improvement in beam ranking correlation over global average pooling.
* **Alternative Hypothesis ($H_{1,2}$):** Extracting visual features from a dynamically computed spatial crop centered on the projected GPS bearing of the partner vehicle preserves vehicle aspect angle and distance cues, restoring sensitivity to visual inputs under temporal shuffle tests ($\Delta_{\text{shuffle}} > 3.0\%$).

### Hypothesis 3: Panel-Aware Main-Lobe Huber Loss
* **Null Hypothesis ($H_{0,3}$):** Restricting profile loss to beams within 10 dB of the optimal beam and enforcing smoothness strictly within physical 64-beam panels does not reduce Average Power Loss (APL) compared to global 256-beam MSE.
* **Alternative Hypothesis ($H_{1,3}$):** Panel-aware main-lobe loss eliminates noise-floor gradient dilution, reducing Average Power Loss (APL) by $\ge 0.5\text{ dB}$ across all four scenarios.

### Hypothesis 4: Trajectory-Adaptive Conformal PID Control under Vehicle Dynamics
* **Null Hypothesis ($H_{0,4}$):** A trajectory-adaptive conformal PID controller achieves the same worst-trajectory miss rate as static conformal risk control.
* **Alternative Hypothesis ($H_{1,4}$):** By coupling online PID feedback with vehicle turning rate $\dot{\theta}$ from GPS, the adaptive controller bounds the worst-trajectory miss rate to $\le \alpha + 2\%$ while reducing average candidate set size by $\ge 20\%$ relative to static CRC.

---

## 3. Proposed Architectural Improvements

### A. Physically Anchored Residual Architecture (PAR-Net)
Instead of forcing the network to memorize absolute beam indices from scratch, the system factorizes the problem:

```
[Historical GPS] ───► [Analytic Geometry Prior] ───► Base Center Beam b_geo & Candidate Set C_20
         │                                                      │
         ▼                                                      ▼
[Kinematic BiGRU] ──┐                                  [Candidate Mask]
                    ├──► [Cross-Attention Fusion] ───► [Residual Offset Head: Δθ ∈ [-15°, +15°]]
[Visual ROI Crop] ──┘                                  [Main-Lobe Power Scorer: ΔP_b for b ∈ C_20]
```

1. **Geometry Prior Module:** Computes forward LOS bearing $\theta_{\text{LOS}} = \text{arctan2}(\Delta y, \Delta x)$ from extrapolated GPS positions. Maps $\theta_{\text{LOS}}$ to the antenna array codebook to obtain the base center beam $b_{\text{geo}}$ and the top-20 candidate directions $\mathcal{C}_{20}$.
2. **Visual ROI Module:** Projects the forward bearing onto the front camera image plane to define a $128\times 128$ Region-of-Interest (ROI) crop centered on the predicted vehicle location, preventing background dilution.
3. **Residual Prediction Head:** The network outputs:
   - A continuous angular correction $\Delta \theta \in [-\Delta \theta_{\text{max}}, +\Delta \theta_{\text{max}}]$ via a $\tanh$ activation.
   - A 20-dimensional relative power-gap score for candidate beams inside $\mathcal{C}_{20}$.

### B. Panel-Aware Main-Lobe Loss Function
The total loss is reformulated to respect physical hardware constraints:
$$\mathcal{L} = \mathcal{L}_{\text{CE}}(\hat{y}_{\text{res}}, y^*) + \lambda_1 \mathcal{L}_{\text{lobe}}(P, \hat{P}) + \lambda_2 \sum_{k=1}^4 \mathcal{L}_{\text{smooth}}^{(k)}(\hat{P})$$
where:
- **Main-Lobe Masked Loss:**
  $$\mathcal{L}_{\text{lobe}} = \frac{1}{|\mathcal{B}_{\text{active}}|} \sum_{b \in \mathcal{B}_{\text{active}}} \text{Huber}\left( P(b) - \hat{P}(b) \right)$$
  where $\mathcal{B}_{\text{active}} = \{b : P(b) \ge \max_{b'} P(b') - 10\text{ dB}\}$. Beams deeper than 10 dB below the peak are ignored.
- **Intra-Panel Smoothness:**
  $$\mathcal{L}_{\text{smooth}}^{(k)} = \frac{1}{63} \sum_{b=64(k-1)+1}^{64k - 1} \left( \hat{P}_{b+1} - \hat{P}_b \right)^2 \quad \text{for } k \in \{1, 2, 3, 4\}$$
  Crucially, smoothness is **never** penalized across panel transitions ($b=64 \to 65$, $128 \to 129$, $192 \to 193$).

---

## 4. Controlled Experimental Matrix and Baseline Suite

Every experiment must be conducted on identical trajectory-blocked splits with frozen seeds ($42, 7, 13$) across all four DeepSense-V2V Scenarios (36, 37, 38, 39):

| Model / Arm ID | Description | Sensory Inputs | Architectural Components | Supervision Loss |
| :--- | :--- | :--- | :--- | :--- |
| **M0 (Baseline)** | Analytic Geometry | GPS History | Non-parametric vector extrapolation + fold offset | None (Deterministic) |
| **M1 (Learned GPS)** | Kinematic BiGRU | GPS History | 3-layer BiGRU ($d=256$) + Linear Head | Cross-Entropy |
| **M2 (Learned RGB)** | Vision Encoder | Front RGB (5 frames) | ResNet-18 ($192\times 192$) + Linear Head | Cross-Entropy |
| **M3 (Standard Fusion)**| Multimodal Transformer | GPS + Front RGB | ResNet-18 + BiGRU + PreLN Transformer | Cross-Entropy + 256-MSE |
| **M4 (Swin Adaptation)**| Shifted-Window Attention| GPS + Front/Rear RGB | EfficientNet-B3 + Shifted-Window Attention | Cross-Entropy (Fabiani et al.) |
| **M5 (PAR-Net: Proposed)**| Physically Anchored Res.| GPS + Front/Rear RGB | Geometry Prior + ROI Crop + Residual Scorer | Panel-Aware Main-Lobe Huber |

### Rigorous Ablation Matrix (Isolating Every Component of M5)
1. **Ablation A1 (w/o RGB):** PAR-Net with GPS and Geometry only (measuring true visual gain).
2. **Ablation A2 (w/o Geometry Prior):** Unconstrained end-to-end model with same capacity (demonstrating the role of the physical anchor).
3. **Ablation A3 (w/o Main-Lobe Mask):** Profile loss computed across all 256 beams (demonstrating noise-floor dilution).
4. **Ablation A4 (w/o Intra-Panel Constraint):** Enforcing smoothness across panel boundaries (measuring hardware mismatch penalty).
5. **Ablation A5 (Visual Sensitivity Probe):** Solid gray frames and temporal shuffling on M5 (verifying non-zero visual gradient flow).

---

## 5. Evaluation Protocol and Statistical Testing

### A. Evaluated Metrics
1. **Primary Communication Utility Metric:**
   - **Top-5 3 dB Power Success Rate:** $\mathbb{P}(L(\mathcal{A}_5) \le 3\text{ dB})$ where $L(\mathcal{A}) = \max_{b} P(b) - \max_{b \in \mathcal{A}} P(b)$.
2. **Secondary Prediction Metrics:**
   - Top-1, Top-3, Top-5 exact-beam accuracy.
   - Decomposition of successes into: (a) *Exact Hits*, (b) *Alternative-Beam Successes* ($L \le 3\text{ dB}$ but peak missed), and (c) *Misses* ($L > 3\text{ dB}$).
3. **Communication Overhead & Quality Metrics:**
   - Average Power Loss (APL in dB).
   - Candidate set size $|C_t|$ and Search Space Reduction ($1 - |C_t|/256$).
   - Worst-Trajectory Miss Rate (measuring resilience during sharp turns or NLOS shadowing).

### B. Statistical Rigor Standards
- **Multi-Seed Execution:** Every model must be trained across three fixed seeds ($42, 7, 13$) with fresh parameter initializations.
- **Trajectory-Block Bootstrap:** Report point estimates alongside **95% percentile confidence intervals** computed from 4,000 paired block resamples. Individual-frame bootstrap is strictly prohibited due to temporal autocorrelation.
- **Formal Significance Tests:**
  - *McNemar's Test* (paired binary classification outcomes on Top-1 and Top-5 hits).
  - *Wilcoxon Signed-Rank Test* (paired non-parametric comparison of continuous dB power loss).
  - A claim of improvement is valid **if and only if** $p < 0.01$ and the paired block-bootstrap 95% lower bound is strictly positive.

---

## 6. Execution Roadmap & Compute Resource Plan

### Phase Execution Timeline
1. **Phase 5 (Implementation):**
   - Implement `src/geometry_prior.py` (vector extrapolation and fold-calibrated index mapping).
   - Implement `src/par_net.py` (Physically Anchored Residual Network with ROI spatial extraction).
   - Implement `src/panel_losses.py` (Panel-aware main-lobe Huber loss).
   - Add comprehensive unit tests in `tests/test_par_net.py` validating tensor dimensions, gradient flow, and panel boundaries.
2. **Phase 6 (Controlled Training & Verification):**
   - Execute controlled pilot training on Scenarios 36–39 using the RTX 4060 GPU (8 GB VRAM, BF16 mixed precision).
   - Training budget: 12 epochs per run, batch size 64, cosine decay LR $10^{-4} \to 10^{-5}$.
   - Store all metrics, epoch logs, and model receipts atomically in `results/par_net_v1/`.
3. **Phase 7 (Statistical Analysis & Ablations):**
   - Run 4,000 paired trajectory-block bootstrap resamples.
   - Generate `ablation_results.csv` and `statistical_analysis.md`.
4. **Phase 8 (Publication Figures & Tables):**
   - Generate vector PDF plots: architecture diagram, beam outcomes decomposition, APL vs. candidate size, trajectory-level tracking.
5. **Phase 9 & 10 (Manuscript Revision & Peer Review):**
   - Overhaul `paper/IEEE_manuscript.md` following standard IEEE TVT/TWC transactions structure.
   - Complete final simulated IEEE peer review.

---

## 7. Explicit Decision Gate (Before Opening Test Split)

To maintain absolute scientific integrity and avoid p-hacking, the locked test split remains strictly sealed until the following pre-declared criteria are met on the validation partition:
1. PAR-Net (M5) must achieve a **3-seed mean top-5 3 dB power success rate $\ge 82.0\%$** (exceeding analytic geometry's 78.56% by at least +3.4 percentage points).
2. The paired trajectory-block bootstrap 95% lower bound against analytic geometry must be **strictly positive** ($\text{CI}_{\text{lower}} > 0.0\%$).
3. PAR-Net must outperform pure GPS (M1) on at least 3 of the 4 held-out scenario transfer folds.
4. If these criteria are not met on validation, the test split will **NOT** be opened, and the paper will be written as a definitive empirical study of V2V multimodal failure modes and candidate-exclusion limits.
