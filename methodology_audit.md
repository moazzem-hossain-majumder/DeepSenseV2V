# Comprehensive Technical and Methodological Audit

**Repository:** [moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  
**Lead Auditor:** Senior IEEE Research Scientist & ML Systems Engineer  
**Date:** November 2026  
**Status:** Complete Technical Audit (Phase 2 Deliverable)

---

## 1. End-to-End Pipeline Architectural Trace

The research repository implements a multi-stage machine learning and statistical signal processing pipeline for vehicle-to-vehicle (V2V) mmWave beam forecasting. The complete pipeline is traced below:

```
[Raw DeepSense V2V Data]
 (Scenarios 36-39: GPS, PWR, RGB180)
            │
            ▼
[Data Extraction & Validation]
 (CRC verify, CSV/Pickle row alignment, 256-beam argmax check)
            │
            ▼
[Spatial & Temporal Feature Engineering]
 (GPS: 9 kinematic features; RGB: 96x96 / 192x192 / 224x224 cached tensors)
            │
            ▼
[Trajectory-Blocked Partitioning]
 (Strict block splitting: Train / Val / Calib / Test; Zero frame overlap)
            │
            ▼
[Predictive Model Training]
 (P1: GPS BiGRU | P2: Profile Huber | P3: Multimodal Transformer Fusion)
            │
            ▼
[Multi-Task Output Heads]
 (Head A: 256-class Logits | Head B: 256-dim Power Profile)
            │
            ▼
[Candidate Set Generator]
 (Fixed Top-k | Power Gap Threshold C_q(x) | Probability Prefix | Geometry Top-20)
            │
            ▼
[Statistical Calibration & Online Control]
 (Split Conformal Risk Control on Calib split | Online ACI / PID Controller on Test stream)
            │
            ▼
[Wireless Communication Utility Evaluation]
 (Top-k Acc, APL in dB, 1 dB/3 dB Miss Rate, Worst-Trajectory Risk, Probing Overhead)
```

---

## 2. Discrepancy Analysis: Documentation vs. Implementation

A rigorous cross-examination of the documentation (`README.md`, `DEEPSENSE_PROJECT_BLUEPRINT_V4.md`), source code (`src/`), and stored experimental artifacts (`results/`) reveals critical discrepancies:

### A. Documented vs. Implemented Architectures
1. **Model Parameterization:** Early documentation (V4 Blueprint) describes P3 as a lightweight model designed for fast mobile deployment. In reality, `src/models.py` implements an 18.2M-parameter model with ResNet-18 (5 layers un-frozen from `layer2`), BiGRU (3 layers, hidden size 256), and a 3-layer 8-head Transformer Encoder with a 512-dim MLP head.
2. **Vision Backbone:** The README references a single ResNet-18 encoder. Later experimental cycles in `src/swin_comparison_training.py` adapted Fabiani et al. using EfficientNet-B3 with shifted-window temporal attention, but froze the backbone and pooled spatial features to a coarse $2\times 2$ grid, deviating from the original published feedback architecture.
3. **Array Continuity:** Documentation describes predicting a continuous "beam-power profile". In reality, the 256 beams belong to **four distinct 64-beam physical antenna panels**. The implementation enforces a smoothness penalty across panel boundaries (e.g., beam 64 to 65), which physically violates the hardware architecture.

### B. Reported vs. Reproducible Metrics
1. **Scenario 36 Cherry-Picking:** The legacy README reported that P3 achieved superior Top-5 accuracy over P1 (GPS-only). Stored multi-seed artifacts in `results/stage7_comparison/REPORT.md` reveal that P3 Top-1 was 15.64% (seed 42), 13.36% (seed 7), and 14.13% (seed 13) — a large sample standard deviation ($1.16\%$). On three-seed mean, P3 Top-5 was inferior to P1.
2. **The "Multimodal Advantage" Myth:** In multi-scenario experiments (Stages 8–36), the reported primary validation metric (3 dB power success at $k=5$) for multimodal models was:
   - Analytic Geometry: **78.56%**
   - Learned GPS: **73.00%**
   - Multimodal Fusion CE: **51.33%**
   - Multimodal Power-Aware Profile: **49.21%**
   The deep learning model performed **27 to 29 percentage points worse** than classical geometry. This reality was completely absent from early project claims.

### C. Intended vs. Actual Dataset Scope
1. **Single vs. Multi-Scenario:** The initial project was built entirely on Scenario 36 with 24,799 samples and only 3 test blocks. This underpowered setup produced illusory statistical significance. Expanding to Scenarios 36–39 (112,196 rows) exposed that the learned models failed to generalize across environments.
2. **Geographical Context:** Early drafts ambiguously referenced deployment in Bangladesh. The data is entirely from a 60 GHz testbed deployed in Tempe/Phoenix/Scottsdale, Arizona.

---

## 3. Deep Technical Audit of Model Architecture & Losses

### A. RGB Vision Feature Extraction
The vision pipeline in `src/models.py` (`ResNet18VisionEncoder`):
```python
x_feat = self.conv1(x_flat)
...
x_feat = self.layer4(x_feat)
x_feat = self.avgpool(x_feat).flatten(1)
return self.proj(x_feat).view(B, S, -1)
```
**Defects Identified:**
1. **Spatial Detail Eradication:** Global average pooling (`avgpool`) condenses the $7\times 7$ spatial feature map into a single 512-dim vector per frame. At $96\times 96$ input resolution, a vehicle at a 30-meter distance occupies approximately $8\times 8$ pixels. Global averaging mixes the vehicle pixels into the vast road/sky background, destroying vehicle orientation and relative angle features.
2. **Nighttime Sensor Degradation:** Scenarios 37 and 39 are recorded at night (20:00–23:00). Camera images contain intense headlight blooming, sensor noise, and low dynamic range. The ResNet-18 pretrained on ImageNet produces degenerate representations dominated by illumination artifacts.
3. **Zero Visual Sensitivity:** Diagnostic ablation experiments (Stages 62–71) proved that replacing images with uniform gray or temporal shuffling produced $\Delta < 0.12\%$ change in validation accuracy, confirming that the vision encoder contributed essentially zero actionable predictive signal.

### B. GPS Kinematic Modeling
The position encoder in `src/models.py` (`BiGRUPositionEncoder`):
- Inputs: 9 features per time step ($t-8, t-6, t-4, t-2, t$):
  - Latitude, Longitude, Altitude (receiver)
  - Relative displacement ($\Delta x, \Delta y, \Delta z$)
  - Relative distance ($d = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$)
  - Heading angle ($\theta = \text{arctan2}(\Delta y, \Delta x)$)
  - Speed ($v$)
- Architecture: 3-layer Bidirectional GRU with hidden dimension 256, projected to $d_{\text{model}} = 256$.

**Defects & Strengths:**
- *Strength:* The feature engineering in `src/dataset.py` is physically sound. Relative bearing and distance directly constrain the line-of-sight (LOS) geometric angle.
- *Defect:* The BiGRU is non-causal across the 5 historical steps (acceptable since all 5 steps are in the past), but the coordinate normalization was initially computed across the full dataset rather than strictly on training trajectories, introducing minor distributional leakage that was later corrected in Stage 8.

### C. Multimodal Fusion Architecture
In `src/models.py` (`PreLNTransformerFusion`):
```python
gps_tok = gps_tokens + self.modality_gps
rgb_tok = rgb_tokens + self.modality_rgb
seq = torch.cat([gps_tok, rgb_tok], dim=1) + self.pos_embed
return torch.mean(self.norm(self.transformer(seq)), dim=1)
```
**Defects Identified:**
1. **Modality Imbalance & Gradient Starvation:** The 5 GPS tokens carry high signal-to-noise ratio (SNR) directional information. The 5 RGB tokens carry noisy background features. In unconstrained self-attention, the Transformer learns to attend heavily to GPS tokens while assigning near-zero attention weights to vision tokens.
2. **Mean Pooling Collapse:** Averaging across all 10 tokens (`torch.mean(..., dim=1)`) further dilutes the temporal evolution of the trajectory, treating time steps as an unordered bag of tokens despite the learned positional embedding.

### D. Multi-Task Power-Profile Loss Function
In `src/models.py`:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(y, \hat{y}) + \lambda_{\text{prof}} \mathcal{L}_{\text{prof}}(P, \hat{P}) + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}(\hat{P}) + \lambda_{\text{rank}} \mathcal{L}_{\text{rank}}(P, \hat{P})$$
where:
- $\mathcal{L}_{\text{CE}}$ is Cross-Entropy on the argmax beam index.
- $\mathcal{L}_{\text{prof}} = \frac{1}{256} \sum_{b=1}^{256} (P_b - \hat{P}_b)^2$ (MSE or Huber).
- $\mathcal{L}_{\text{smooth}} = \frac{1}{255} \sum_{b=1}^{255} (\hat{P}_{b+1} - \hat{P}_b)^2$.
- $\mathcal{L}_{\text{rank}} = \sum_{b, b'} \max(0, - (P_b - P_{b'})(\hat{P}_b - \hat{P}_{b'}) + \gamma)$.

**Critical Mathematical Flaws:**
1. **Noise-Floor Dominance:** 250 out of 256 beams have powers between $-95\text{ dBm}$ and $-105\text{ dBm}$ (pure thermal noise and sidelobe clutter). The MSE loss over 256 beams spends $97.6\%$ of its gradient budget fitting microscopic noise fluctuations across inactive beams rather than resolving the relative power gap between the top 5 candidates.
2. **Hardware-Unaware Smoothness Regularization:** DeepSense-V2V utilizes **four physical antenna arrays** pointing in orthogonal/offset quadrants:
   - Array 1: Beams 1–64
   - Array 2: Beams 65–128
   - Array 3: Beams 129–192
   - Array 4: Beams 193–256
   Penalizing $(\hat{P}_{65} - \hat{P}_{64})^2$ penalizes a boundary transition between physically separate antenna faces, actively corrupting the model's physical understanding of the hardware.
3. **Loss Interference:** Cross-entropy sharpens the softmax distribution toward a one-hot distribution on the single best beam, while profile MSE flattens predictions to match measured dBm values. These two objectives pull the shared representation in diametrically opposing directions.

---

## 4. Conformal Prediction and Online Controller Correctness

### A. Static Conformal Risk Control (CRC)
Implemented in `src/candidate_sets.py`:
$$C_q(x) = \{b \in \{1,\dots,256\} : \hat{P}(b) \ge \max_{b'} \hat{P}(b') - q\}$$
Calibrates $q$ on the calibration split:
$$\hat{q} = \inf \left\{ q \in \mathcal{Q} : \frac{n}{n+1} \hat{R}_n(q) + \frac{1}{n+1} \le \alpha \right\}$$
where empirical risk is the $\Delta\text{ dB}$ miss rate:
$$\hat{R}_n(q) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}\left( \max_{b \in C_q(x_i)} P_i(b) < \max_{b} P_i(b) - \Delta \right)$$

**Rigorous Mathematical Critique:**
- **Validity Condition:** The theoretical guarantee $\mathbb{E}[R(\hat{q})] \le \alpha$ holds **if and only if** calibration and test samples are exchangeable:
  $$(X_1, Y_1), \dots, (X_n, Y_n), (X_{n+1}, Y_{n+1}) \sim \text{Exchangeable}$$
- **Violation:** In vehicular trajectories, $X_t$ and $X_{t+1}$ share high mutual information ($\rho > 0.85$ correlation). A vehicular turn, speed change, or building shadow breaks exchangeability between calibration trajectories and test trajectories.
- **Consequence:** While static CRC is an excellent *heuristic* calibration framework, claiming a "distribution-free finite-sample guarantee" in a V2V paper is mathematically false and will be rejected by any competent mathematical reviewer.

### B. Adaptive Conformal Inference (ACI)
Implemented in `src/online_controller.py`:
$$q_{t+1} = \text{clip}\left( q_t + \eta (\text{err}_t - \alpha), q_{\text{min}}, q_{\text{max}} \right)$$

**Critique:**
- **Asymptotic vs. Finite Trajectories:** Gibbs & Candès (2021) established that ACI provides a **long-run time-averaged** empirical frequency guarantee:
  $$\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T \text{err}_t = \alpha$$
- **Finite-Horizon Pathology:** In vehicular driving sequences ($T \approx 100\text{--}200$ samples), ACI exhibits significant tracking lag. If step size $\eta$ is small ($\eta = 0.005$), a sudden NLOS blockage causes a run of 10–20 consecutive misses before $q_t$ expands sufficiently. Conversely, if $\eta$ is large ($\eta = 0.05$), $q_t$ spikes to $q_{\text{max}} = 30\text{ dB}$, inflating candidate set sizes to all 256 beams.
- **Feedback Assumption Violation:** In real V2V networks, observing $\text{err}_t$ requires that the vehicle *actually probed the true best beam* to measure its power. If the candidate set missed the best beam, the true optimum power is **unobserved**! ACI in this repository assumes oracle full-profile channel state feedback at every step, which is physically impossible in an operational system without scanning all 256 beams.

---

## 5. Wireless Communication Utility Evaluation

### A. Average Power Loss (APL)
The challenge metric:
$$\text{APL} = \frac{1}{N} \sum_{i=1}^N \left( \max_{b \in \{1,\dots,256\}} P_i(b) - \max_{b \in \hat{\mathcal{S}}_i} P_i(b) \right)$$
where $\hat{\mathcal{S}}_i$ is the predicted subset of size $k$ (e.g., $k=5$).
- *Correctness:* Verified. The implementation in `src/evaluate.py` accurately computes the power gap between the true maximum beam power and the maximum power among the proposed candidate set.

### B. The Alternative-Beam Trade-Off (The Central Discovery)
The empirical analysis in Stage 299 uncovered a critical insight into V2V beam selection:
- Under shifted-window multimodal fusion (Fabiani et al. adaptation):
  - **Exact Top-5 Hit Rate:** Increased by **+5.42 percentage points** with RGB.
  - **3 dB Power Success Rate:** Collapsed by **-19.85 percentage points** with RGB!
- *Physical Explanation:* When the neural network is trained with cross-entropy, it sharpens its output on the exact labeled peak beam. In transfer scenarios where the peak beam shifts slightly due to calibration offsets, the multimodal model completely misses the main lobe.
- Meanwhile, the GPS baseline selects broad spatial beams surrounding the LOS vector. Even when it misses the exact peak, its candidate set contains beams in the shoulder of the main lobe that deliver $\ge -3\text{ dB}$ of optimal power.

---

## 6. Synthesis of Identified Deficiencies & Remediation Strategy

| Audit Area | Critical Defect | Required Scientific Remediation |
| :--- | :--- | :--- |
| **Model Structure** | Deep networks severely underperform analytic geometry (49% vs 78%). | **Anchor on Geometry:** Implement a Physically Anchored Residual Architecture where GPS geometry defines the prior base, and the neural network only predicts bounded angular offsets. |
| **Vision Modality** | Global average pooling destroys vehicle spatial coordinates; zero sensitivity on shuffle tests. | **Crop/Spatial Attention:** Implement heading-directed spatial cropping or cross-attention focused on the forward vehicle trajectory rather than full-frame pooling. |
| **Loss Function** | MSE across 256 beams fits noise floor; smoothness regularizer violates panel boundaries. | **Masked Main-Lobe Huber Loss:** Restrict profile supervision to beams within 10 dB of the maximum; enforce smoothness strictly *within* each 64-beam physical panel. |
| **Conformal Inference** | False exchangeability claim on non-stationary V2V trajectories. | **Frame as Empirical Adaptive Control:** Disclose exchangeability limitations honestly; reframe ACI as an online trajectory-adaptive feedback controller with certified time-averaged bounds. |
| **Benchmarking** | Misleading single-seed results on underpowered Scenario 36 split. | **4-Scenario Blocked Evaluation:** Report 3-seed mean, standard deviation, and paired block-bootstrap confidence intervals across all 4 scenarios. |
