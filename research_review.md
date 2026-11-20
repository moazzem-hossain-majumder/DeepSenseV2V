# Critical IEEE Peer Review Report

**Manuscript/Project Title:** Trajectory-Adaptive Power-Profile Conformal Beam Tracking from Synchronized RGB and GPS for Reliable V2V mmWave Communications  
**Repository:** [moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  
**Reviewer Role:** Senior IEEE Research Scientist, ML Engineer, and Critical Peer Reviewer  
**Evaluation Standard:** IEEE Transactions on Vehicular Technology (TVT) / IEEE Transactions on Wireless Communications (TWC)  
**Recommendation:** **REJECT (in current form)** / Requires Comprehensive Methodological Restructuring & Scientific Refinement

---

## Executive Summary & Reviewer Scores

| Evaluation Criterion | Score (out of 10) | Key Review Findings |
| :--- | :---: | :--- |
| **1. Research Problem & Significance** | **8.5 / 10** | Millimeter-wave (60 GHz) V2V beam tracking is a critical problem in 5G-NR-V2X and 6G. Reducing 256-beam exhaustive probing overhead under vehicle mobility is of high practical importance. |
| **2. Novelty & Prior Art Differentiation** | **4.0 / 10** | Significant overlap with published literature. Multiple core claims (conformal subset selection, multi-camera GPS fusion, power-aware loss) have already been published in SCAN-BEST (2025/2026), Fabiani et al. (2024), and Mollah et al. (2025). The remaining gap is narrow and currently unsupported by the empirical results. |
| **3. Technical Soundness & Methodology** | **4.5 / 10** | Multimodal deep learning underperforms a simple analytic geometry baseline across all 4 scenarios. Deep networks achieve ~49–51% validation power success, while classical geometry achieves 78.56%. The visual branch exhibits catastrophic tanh saturation and zero sensitivity on shuffle tests. |
| **4. Conformal & Online Control Rigor** | **5.0 / 10** | Conformal risk control (CRC) is deployed on non-stationary, spatially and temporally correlated V2V trajectories, violating the foundational exchangeability assumption without provable shift correction. ACI time-averaged guarantees are confused with finite-sample coverage. |
| **5. Experimental Completeness & Baselines** | **6.5 / 10** | Commendable data acquisition (112,196 rows across Scenarios 36–39) and trajectory-blocked splitting. However, single-scenario claims in the original V4 README contradict multi-scenario findings. Test partitions remain unopened due to failed development gates. |
| **6. Statistical Evaluation & Significance** | **6.0 / 10** | Block bootstrap procedures are correctly implemented, but multi-seed variance reveals that earlier reported gains (e.g., P3 seed 42) were statistical noise that disappeared across seeds 7 and 13. Paired intervals against geometry include zero or are negative. |
| **7. Reproducibility & Code Engineering** | **8.0 / 10** | Clean repository layout, atomic checkpointing, strict random seed logging, checksum manifests, and zero data leakage across trajectory blocks. |
| **8. Writing Quality & Academic Framing** | **5.5 / 10** | Divergence between the legacy single-scenario README, the V4/V5 blueprints, and the incomplete TVT draft. Inappropriate geographic deployment claims regarding Bangladesh require strict confinement to motivational context. |
| **OVERALL REVIEWER SCORE** | **5.2 / 10** | **REJECT (Substantial revision and restructuring required before resubmission)** |

---

## Detailed Peer Review

### A. Research Problem and Practical Significance

Vehicle-to-Vehicle (V2V) communications operating in the unlicensed 60 GHz mmWave band (IEEE 802.11ad/ay and 3GPP Rel-16/17/18 NR-V2X) face severe path loss, atmospheric attenuation, and rapid link blockage. Narrow directional beams synthesized by high-gain phased antenna arrays (e.g., 256-beam codebooks partitioned across four 64-beam panels) are essential to close the link budget.

However, in high-mobility V2V environments where transmitter and receiver vehicles change relative displacement, heading, and velocity dynamically, exhaustive beam sweep across 256 codebook entries incurs an unacceptable alignment latency ($\sim 100\text{--}250\text{ ms}$). This latency frequently exceeds the channel coherence time ($\approx 5\text{--}15\text{ ms}$ at 60 km/h), inducing severe beam-alignment overhead, packet loss, and frequent radio link failures (RLF).

The problem of utilizing out-of-band sensory observations (such as GNSS/GPS position history and vehicle-mounted RGB camera feeds) to predict optimal future beam directions $500\text{ ms}$ in advance is of paramount importance to the vehicular communications community. The conceptualization of outputting a calibrated candidate set of beams (e.g., Top-5) to restrict in-band probing to a minuscule fraction ($<2\%$) of the codebook while bounding the probability of severe received-power drop ($\le 3\text{ dB}$) is mathematically well-motivated and practically valuable.

---

### B. Novelty Assessment and Literature Positioning

#### 1. The Claimed Novelty (Blueprint V4 / Early Drafts)
The repository originally claimed three primary contributions:
1. Predicting the *complete future 256-beam measured power-profile* (rather than a single categorical class or subset).
2. Pairing profile prediction with an *online trajectory-adaptive risk controller* (ACI / PID) that updates candidate-set thresholds sequentially.
3. Explicit separation of offline exchangeable coverage guarantees from online time-averaged reliability.

#### 2. Critical Prior-Art Collision
A meticulous examination of the literature reveals that this novelty positioning is largely preempted or substantially compromised:

- **SCAN-BEST (2025/2026):** Proposes candidate beam subset selection using Conformal Risk Control (CRC) with relaxed optimality criteria ($\Delta\text{ dB}$ margin), limited in-band probing, and weighted conformal calibration under covariate shift. The core concept of "conformal candidate set selection with dB power margin" was already established by SCAN-BEST.
- **Fabiani et al. (IEEE VTC / Conf 2024, DOI: [10.1109/IEEEConf60004.2024.10942842](https://doi.org/10.1109/IEEEConf60004.2024.10942842)):** Evaluates multimodal V2V beam prediction using synchronized GPS and dual (front + rear) cameras with shifted-window temporal attention on DeepSense-V2V data at a 500 ms forecasting horizon, evaluating both Top-$k$ accuracy and Average Power Loss (APL). The sensory inputs, prediction horizon, and application context are identical.
- **Mollah, Wang, and Fang (IEEE GLOBECOM 2025, DOI: [10.1109/GLOBECOM59602.2025.11432270](https://doi.org/10.1109/GLOBECOM59602.2025.11432270)):** Benchmarks unimodal vs. multimodal fusion across all four DeepSense-V2V Scenarios (36–39) with MaxViT backbones, reporting power loss and accuracy under cross-scenario settings.
- **Wang et al. (IEEE PIMRC 2026, DOI: [10.1109/PIMRC69400.2026.11711825](https://doi.org/10.1109/PIMRC69400.2026.11711825)):** Directly studies cross-environment label-space shift and few-shot adaptation using measured received power for pairwise beam ranking.
- **CLBP (Gao et al., IEEE TMC 2026, DOI: [10.1109/TMC.2026.3674213](https://doi.org/10.1109/TMC.2026.3674213)) & AMBER (2025):** Establish loss-tolerant multimodal fusion and adaptive probing budgets under missing modalities.

#### 3. What Actually Remains Novel?
The genuinely defensible novelty gap in this repository is **NOT** that deep learning predicts beams or uses conformal inference. Rather, the actual scientific value lies in two rigorous, unvarnished empirical discoveries revealed in the repository's deep multi-scenario audit (Stages 8–299):
1. **The Shortlist Alternative-Beam Pathology:** Under cross-entropy or categorical supervision, multimodal neural networks overfit to exact peak-beam labels (+5.42% exact hit rate), but their selection of candidate sets collapses by **-19.85 percentage points** in 3 dB near-optimal power coverage because they fail to capture broad spatial side-lobes that GPS alone preserves.
2. **The Candidate Exclusion Bottleneck:** In constrained geometry rerankers, **84.7% of all severe power failures (>10 dB loss)** are caused by *candidate exclusion* (the near-optimal beam not being in the top-20 candidate generator at all), whereas ranking errors inside the shortlist account for only 2.57 percentage points.

*Reviewer Note:* The authors must abandon grandiose "first-ever conformal beam forecasting" claims and pivot the manuscript around these exact, scientifically defensible failure mechanism analyses and physical prior rectifications.

---

### C. Technical and Methodological Assessment

#### 1. Baseline Dominance over Deep Neural Networks
The most damning technical finding in the repository is that **the proposed deep learning architecture (P3) is substantially outperformed by a simple, non-parametric classical baseline (Analytic Geometry)**:
- **Analytic Geometry Baseline (GPS vector extrapolation):** **78.56%** validation macro-success (within 3 dB at $k=5$).
- **Learned GPS-only network (BiGRU):** **73.00%** ($\Delta = -5.56\%$).
- **Multimodal Fusion Cross-Entropy:** **51.33%** ($\Delta = -27.23\%$).
- **Original Profile Regression Objective (P3):** **48.54%** ($\Delta = -30.02\%$).
- **Power-Aware Soft-Label Objective:** **49.21%** ($\Delta = -29.35\%$).

*Critique:* In an IEEE journal submission, proposing an 18-million-parameter deep multimodal transformer that performs 27 to 30 percentage points *worse* than an analytic geometry baseline without residual anchoring will result in immediate rejection.

#### 2. Visual Modality Collapse & Diagnostic Failures
The repository's diagnostic cycle (Stages 62–71) conducted controlled ablation probes (solid gray frames and temporally shuffled frames):
- When front camera images were replaced with solid gray frames, validation success changed by $< 0.12\%$.
- When frames were temporally shuffled, performance remained statistically unchanged.
- Mechanism visual audit (Stage 117): At $96\times 96$ resolution (and even at $192\times 192$), nighttime frames (Scenarios 37 and 39) depict vehicles as 5–10 pixel indistinct blobs with severe headlamp glare and sensor noise.
- Global spatial pooling (`nn.AdaptiveAvgPool2d(1)`) discards spatial vehicle bounding boxes, turning the image encoder into a day/night ambient brightness detector.

#### 3. Power-Profile Loss Formulation Issues
The multi-task loss function in `src/models.py`:
$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{prof}} \mathcal{L}_{\text{prof}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{rank}} \mathcal{L}_{\text{rank}}$$
suffers from fundamental mathematical and physical flaws:
- **Profile MSE/Huber on 256 beams:** The vast majority of beams ($>240$) operate near the noise floor ($\approx -90\text{ to } -105\text{ dBm}$), while only 3–5 beams form the main lobe. An unweighted MSE over 256 beams forces the optimizer to fit the noise floor of non-transmitting directions, diluting gradients around the narrow beam peak.
- **Smoothness Penalty across Panel Boundaries:** The smoothness loss:
  $$\mathcal{L}_{\text{smooth}} = \frac{1}{255} \sum_{b=1}^{255} (P_b - P_{b-1})^2$$
  incorrectly assumes continuous beam-angle indexing across all 256 beams. In DeepSense-V2V, the 256 beams are composed of **four separate 64-beam planar arrays** (Array 1: beams 1–64, Array 2: 65–128, Array 3: 129–192, Array 4: 193–256). Penetrating across beam 64 to 65 represents a discontinuous switch between physically distinct antenna panels pointing in different directions! Penalizing $(P_{65} - P_{64})^2$ violates the physics of the array architecture.

---

### D. Experimental and Statistical Assessment

#### 1. Dataset Scope and Trajectory Partitioning
The repository deserves strong praise for handling the full raw DeepSense dataset:
- Scenarios 36–39 fully audited and verified: 112,196 samples across 561 sequences.
- 100% agreement between power-argmax and ground-truth beam indices.
- Trajectory-blocked partitioning strictly enforces zero sample or sequence overlap between train, validation, calibration, and test splits.

#### 2. Single-Scenario Seed Instability (The Scenario 36 Flaw)
In the initial Scenario 36 experiments:
- Only 19 trajectory blocks existed, with a meager **3 blocks in the test set**.
- P3 seed 42 appeared superior to P1 (GPS-only) on Top-5 accuracy.
- When replicated across seeds 7 and 13, the three-seed mean Top-5 accuracy of P3 was actually **inferior** to P1, with massive standard deviations ($>3.9\%$).
- The reported "advantage" of multimodal profile prediction in the early README was an artifact of cherry-picked random seed 42 on an underpowered 3-block test split.

#### 3. Hardware & Sensor Timing Provenance
The provenance audit (Stage 62) uncovered that the raw front-camera JPEG timestamps and the GPS/power CSV row timestamps exhibit median offsets ranging from $-1.10\text{ s}$ to $+1.50\text{ s}$ across Scenarios 36–39. Any real-time causal beam tracking claim is compromised if camera frames are asynchronous with the current vehicle state by over a second.

---

### E. Conformal Prediction and Online Controller Assessment

#### 1. Static CRC and Exchangeability Violations
Static Conformal Risk Control (Bates et al., 2021; Angelopoulos et al., 2024) proves that on exchangeable calibration and test data:
$$\mathbb{E}[\mathcal{L}(C_{\hat{q}}(X), Y)] \le \alpha$$
In vehicular beam tracking, this guarantee is mathematically invalid:
- Consecutive frames along a trajectory exhibit Markovian dependence and non-zero temporal autocorrelation.
- When moving from downtown street canyons (Tempe/Phoenix) to open highways, the data distribution undergoes non-stationary covariate and concept shift.
- The repository must explicitly state that static CRC provides *empirical calibration*, not formal finite-sample distribution-free guarantees.

#### 2. Adaptive Conformal Inference (ACI) and PID Correctness
The implementation of ACI:
$$q_{t+1} = \text{clip}(q_t + \eta (\text{err}_t - \alpha), q_{\text{min}}, q_{\text{max}})$$
is mathematically faithful to Gibbs & Candès (2021). However:
- ACI provides a **long-run time-averaged** error rate guarantee:
  $$\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T \text{err}_t = \alpha$$
- On short test trajectories ($T \approx 100\text{--}200$ samples), ACI oscillates violently around $\alpha$. If step size $\eta$ is tuned on validation trajectories of one speed regime, it fails to adapt when vehicle acceleration or turning dynamics spike in the test block.
- In `src/online_controller.py`, tuning $\eta$ on validation trajectories selected $\eta \in [0.001, 0.05]$. But when deployed on test trajectories with sudden NLOS transitions, the controller experiences severe lag, causing bursts of consecutive misses before inflating $q_t$ to its maximum ($q_{\text{max}} = 30\text{ dB}$), which trivially selects all 256 beams.

---

## Final Recommendation & Necessary Roadmap

### Verdict: REJECT (in present form)

### Why this cannot be submitted as-is:
1. The paper cannot claim that multimodal deep learning is a superior solution for V2V beam tracking when the data proves it is **27% worse** than classical GPS vector geometry.
2. The paper cannot claim visual benefits when grayscale and temporal shuffle ablations produce zero drop in performance.
3. The early single-scenario README claims are contradicted by the multi-scenario empirical results.
4. SCAN-BEST, Fabiani et al., and Mollah et al. already claim the headline methods.

### How to turn this into an Acceptable IEEE Journal Paper:
1. **Pivot the Central Hypothesis:** Reframe the paper from a naive "Multimodal SOTA" claim to an authoritative **Scientific Investigation of Multimodal Failure Modes & Physically Anchored Residual Beam Tracking**.
2. **Anchor on Geometry:** Implement a **Geometry-Anchored Residual Architecture** where analytic GPS geometry provides the base probability distribution over candidate beams, and neural network branches only learn bounded local deviations.
3. **Region-of-Interest (ROI) Visual Attention:** Replace global average pooling with vehicle-heading-directed spatial cross-attention to resolve the target vehicle.
4. **Physically Disjoint Array Profile Loss:** Correct the profile loss to respect the four 64-beam physical panels and focus exclusively on the main-lobe power gap ($\le 10\text{ dB}$) rather than the noise floor.
5. **Rigorous Multi-Scenario Benchmark:** Conduct paired trajectory-bootstrap evaluations across all four scenarios with frozen seeds and pre-declared evaluation protocols.
