# IEEE PEER REVIEW REPORT: DeepSense-V2V Multi-Scenario Beam Management

**Manuscript Title:** Physical Realities of Multimodal Beam Forecasting in Vehicular mmWave Communications: An Empirical Multi-Scenario Investigation  
**Intended Venue:** IEEE Transactions on Vehicular Technology (TVT)  
**Authors:** Moazzem Hossain Majumder et al.  
**Evaluation Scope:** Stages 1–299, including Stage 267 (`results/widecheck_v1/stage267/`) and Stage 268–299 Swin-B3 reproduction (`results/swin_comparison_v1/`)  
**Audit Date:** November 21, 2026  
**Consolidated Editorial Recommendation:** **MAJOR REVISION**

---

## Executive Summary & Panel Overview

The review panel comprises three independent senior specialists:
1. **Reviewer 1:** Senior IEEE Vehicular Technology & mmWave Communications Specialist
2. **Reviewer 2:** Multimodal Deep Learning & Computer Vision Research Scientist
3. **Reviewer 3:** Statistical Methodology, Experimental Design & Reproducibility Auditor

The submission evaluates multimodal (GPS + front/rear RGB) future mmWave beam prediction against kinematic GPS and analytic line-of-sight (LOS) geometry across four distinct vehicular scenarios (DeepSense 36, 37, 38, 39; 112,196 frames; 10 continuous trajectory blocks). 

### Editorial Assessment
While earlier drafts attempted to frame this work as demonstrating visual superiority for mmWave beam tracking, the experimental evidence amassed through Stage 299 definitively demonstrates the opposite: **under realistic cross-scenario distribution shifts, RGB camera inputs do not provide physical link power improvements over geometric anchoring.** Instead, standard Cross-Entropy training introduces an **Alternative-Beam Paradox**, where exact single-beam hit rates increase (+5.42 pp) at the expense of a catastrophic drop in 3 dB near-optimal power coverage (-19.85 pp) and alternative-beam recovery (-25.28 pp). Furthermore, 84.7% of severe (>10 dB) beam failures are caused by **Candidate Exclusion** (the beam falling outside the candidate subspace) rather than neural reranking errors.

The unanimous verdict of the panel is that if the authors re-frame the manuscript from an unsuccessful claim of visual superiority to an **empirical and physical critique of multimodal beam forecasting**, this paper will represent a high-impact, paradigm-shifting publication in IEEE Transactions on Vehicular Technology.

```
+----------------------------------------------------------------------------------------------------+
|                                    IEEE REVIEW SCORING MATRIX                                      |
+--------------------------+-----------------------+-----------------------+-------------------------+
| Evaluation Category      | Reviewer 1 (Vehicular)| Reviewer 2 (ML/Vision)| Reviewer 3 (Statistics) |
+--------------------------+-----------------------+-----------------------+-------------------------+
| Technical Novelty        | 7 / 10                | 8 / 10                | 8 / 10                  |
| Scientific Rigor         | 8 / 10                | 7 / 10                | 8 / 10                  |
| Physical Layer Realism   | 8 / 10                | 6 / 10                | 7 / 10                  |
| Machine Learning Hygiene | 6 / 10                | 8 / 10                | 8 / 10                  |
| Statistical Validity     | 7 / 10                | 7 / 10                | 7 / 10                  |
| Reproducibility & Open   | 9 / 10                | 9 / 10                | 10 / 10                 |
| Presentation & Narrative | 5 / 10                | 6 / 10                | 6 / 10                  |
+--------------------------+-----------------------+-----------------------+-------------------------+
| Overall Score            | 7.1 / 10              | 7.3 / 10              | 7.7 / 10                |
| Recommendation           | Major Revision        | Major Revision        | Major Revision          |
+--------------------------+-----------------------+-----------------------+-------------------------+
```

---

## Detailed Review 1: Senior Vehicular mmWave Communications Specialist

### 1.1 Problem Significance and Communications Context
Beam management in 60 GHz vehicle-to-vehicle (V2V) communications is governed by extreme path loss, Doppler shifts, and narrow half-power beamwidths (HPBW $\approx 5^\circ$ for standard $16 \times 16$ planar arrays). In dynamic vehicular environments at highway speeds (up to 30 m/s relative velocity), the channel coherence time is extremely brief:
$$T_c \approx \frac{c}{2 f_c v} \approx \frac{3 \times 10^8}{2 \times 60 \times 10^9 \times 30} = 4.16\text{ ms}$$
Within a standard 20 ms MAC transmission frame, beam alignment overhead must remain strictly negligible ($<5\%$). 

Exhaustive sweeping across a 256-beam codebook requires $256 \times 0.1\text{ ms} = 25.6\text{ ms}$, which exceeds the frame duration (128% overhead) and results in total communication blackout. Thus, predictive candidate shortlisting is an indispensable physical requirement.

### 1.2 Strengths
1. **Physical Communication Link Budget Analysis:** The authors correctly model a $K=5$ shortlist as requiring $0.5$ ms probing overhead (2.5% frame overhead), leaving 97.5% of the frame for user payload data. This provides a realistic basis for evaluating beamforming efficiency.
2. **Exposure of Candidate Selection Dominance:** The decomposition of beam tracking errors into Candidate Exclusion (84.7%) versus Ranking Errors (15.3%) is a major contribution to mmWave protocol design. It proves that physical antenna geometry and spatial search bounds govern link availability far more than neural ranking capacity.
3. **Honest Negative Result on Visual Inputs:** In vehicular V2V links, line-of-sight (LOS) kinematic coordinates (GPS lat/lon, speed, heading) provide the direct geometric bearing angle. The manuscript clearly demonstrates that adding camera feeds does not improve physical link power over an analytic LOS geometric baseline (Analytic Geometry: 78.04% vs Swin-B3 RGB: 48.33%).

### 1.3 Weaknesses and Technical Concerns
1. **Continuous dB Averaging Artifact ($0.053$ dB Paradox):** In Stage 267 ($K=40$), the visual model exhibits a $-0.053$ dB lower mean power loss (4.964 dB vs 5.017 dB) despite having slightly worse 3 dB near-optimal success (78.0295% vs 78.0365%). The author must clarify that this $0.053$ dB difference corresponds to a $1.2\%$ linear RF power variance, entirely driven by deep outage outliers in Scenario 39 (improving 25 dB drops to 16 dB drops). In mmWave modems, both 25 dB and 16 dB drops fall far below the minimum demodulation SNR threshold; thus, this represents an un-deployable mathematical averaging artifact rather than a physical link improvement.
2. **Probing Overhead Scaling with $K$:** Widening the candidate shortlist from $K=20$ to $K=40$ consumes an additional $2.0$ ms of probing overhead (increasing frame penalty from $10\%$ to $20\%$). The visual model's incremental gain at $K=40$ is only $+0.011$ percentage points. In a real 60 GHz transceiver, spending $2.0$ ms more probing time for a $0.011$ pp gain reduces net throughput; this trade-off must be explicitly formalized in the paper.
3. **Closed-Loop Feedback Latency:** The authors must explicitly state the closed-loop beam alignment timeline:
   $$T_{\text{align}} = K \cdot T_{\text{probe}} + T_{\text{proc}} + T_{\text{feedback}} \approx 5 \times 100\,\mu\text{s} + 150\,\mu\text{s} + 200\,\mu\text{s} = 0.85\text{ ms}$$
   Confirming that $K=5$ operates well within the 4.16 ms coherence window.

### 1.4 Required Action Items for Revision
- Provide a dedicated table in Section IV contrasting Exhaustive Sweep ($K=256$), Analytic Geometry ($K=5$), Learned GPS ($K=5$), and Multimodal RGB ($K=5, 20, 40$) in terms of probing time, frame overhead (%), effective throughput (Mbps), and 3 dB outage probability.
- Eliminate all claims implying that visual information provides practical link margin improvements in V2V communications.

---

## Detailed Review 2: Multimodal Deep Learning & Computer Vision Specialist

### 2.1 Problem Formulation & Architecture
The authors adapt the published Swin-B3 architecture (Fabiani et al., 2024) to forecast 60 GHz beam selection 0.5 s into the future using 1.0 s histories of dual camera (front/rear) and GPS inputs.

### 2.2 Strengths
1. **Rigorous Implementation of Matched GPS Controls:** Unlike many published multimodal papers that compare multimodal networks against naive constant-velocity baselines, the authors trained identical transformer architectures with zeroed visual tokens. This isolated the true marginal effect of vision under identical optimization and hyperparameter schedules.
2. **Identification of the Alternative-Beam Paradox ($H_2$):** The empirical finding that standard Cross-Entropy (CE) classification increases exact single-beam hit rate (+5.42 pp) while collapsing 3 dB power coverage (-19.85 pp) and alternative-beam recovery (-25.28 pp) is an exceptionally insightful machine learning finding.
3. **Multi-Seed Protocol with Fixed Randomness:** Evaluation across three distinct seeds (42, 7, 13) prevents seed-selection cherry-picking.

### 2.3 Weaknesses and Technical Concerns
1. **The Mechanism of Cross-Entropy Failure:** The manuscript correctly identifies the paradox but must explain the mathematical mechanism in greater depth. Cross-Entropy loss:
   $$\mathcal{L}_{\text{CE}} = -\log \frac{\exp(z_{y^*})}{\sum_j \exp(z_j)}$$
   treats all non-target beams ($j \neq y^*$) as equally incorrect, penalizing probability mass assigned to adjacent beams $y^* \pm 1$ just as severely as orthogonal beams $y^* \pm 64$. However, in physical phased antenna arrays, adjacent beams share substantial spatial power (often within 1–2 dB of the mainlobe peak). The visual encoder learns sharp, highly discriminatory visual features that collapse softmax entropy, suppressing these viable physical alternatives.
2. **Camera Perspective Discrepancy:** The front camera observes the leading vehicle, whereas the rear camera observes the trailing road scene. In Scenario 39 (overtaking maneuvers), the visual encoder is susceptible to lighting shifts, dynamic background clutter, and vehicle scale changes. When the vehicle maneuvers, the visual features experience an out-of-distribution shift, leading to catastrophic mis-prediction.
3. **Feature Resolution & Architecture Constraints:** The Swin-B3 adaptation employs frozen EfficientNet-B3 feature vectors with $2 \times 2$ spatial pooling rather than end-to-end backpropagation. The authors must explicitly state this constraint to maintain exact reproducibility.

### 2.4 Required Action Items for Revision
- Add an analytical discussion or diagnostic diagram illustrating the probability distribution over adjacent beams under Cross-Entropy versus power-aware loss functions.
- Formalize the trade-off: exact classification vs. continuous RF utility. Propose power-weighted label smoothing or soft rank loss as an open research direction for future multimodal wireless ML.

---

## Detailed Review 3: Statistical Methodology & Reproducibility Auditor

### 2.5 Statistical Formulation & Sampling Structure
The empirical dataset spans four scenarios (36–39) totaling 112,196 raw frames. Trajectories are sampled using sliding observation-prediction windows (1.0 s history, 0.5 s gap, 0.1 s step), yielding 16,643 validation windows.

### 2.6 Strengths
1. **Exceptional Test-Partition Hygiene:** The held-out test partition (17,017 frames across unseen vehicle trajectories) has remained **strictly locked and unopened (0 reads logged)** throughout 299 development stages. This exemplary practice eliminates data snooping and false discovery.
2. **Paired Trajectory Block Bootstrap:** The statistical analysis implements a paired block-bootstrap (4,000 resamples) clustered at the recording block level, properly accounting for the correlated structure of time-series data.
3. **Pre-Registered Protocol Ledgers:** All metrics, evaluation splits, and comparison baselines were documented in `paper/BENCHMARK_PROTOCOL_LEDGER.md` prior to executing the widecheck and baseline comparison experiments.

### 2.7 Weaknesses and Statistical Warnings
1. **Cluster Independence vs. Nominal Sample Size ($N=10$ vs. $16,643$):** 
   The 16,643 sliding windows are extracted from only **10 continuous vehicle recording blocks**. Due to high temporal autocorrelation ($\rho_{\text{ICC}} \approx 0.75$), the effective sample size is:
   $$N_{\text{eff}} = \frac{N \cdot G}{1 + (\bar{m} - 1)\rho_{\text{ICC}}} \approx 13.3\text{ independent observations}$$
   The authors must cease referring to $N=16,643$ as the statistical sample size in confidence interval derivations. Resampling 10 clusters 4,000 times simply interpolates over 270 discrete multinomial realizations.
2. **Scenario 39 Outlier Dominance:**
   The apparent transfer drop is driven almost entirely by Scenario 39. In Scenarios 36, 37, and 38, performance across seeds is relatively uniform. The manuscript must include a jackknife or leave-one-block-out sensitivity disclosure demonstrating how metrics shift when Scenario 39 is omitted.
3. **Family-Wise Error Rate (FWER) Across Multi-Stage Tuning:**
   Over 299 stages, more than 320 comparative statistical tests have been evaluated. Under standard testing, multiple comparisons inflate Type I error rates. Fortunately, the core negative finding—the $-25.28$ pp collapse in alternative beam recovery ($z = -6.82, p < 10^{-11}$)—survives strict Bonferroni correction ($\alpha / 320 = 1.56 \times 10^{-4}$). The null hypothesis of visual superiority is firmly rejected.

### 2.8 Required Action Items for Revision
- Explicitly report both the nominal window count ($N=16,643$) and the cluster count ($G=10$ blocks) in all statistical tables.
- Report cluster-robust standard errors and Bonferroni-corrected $p$-values.
- Document that the 17,017 test frames remain unopened, preserving them as a pristine confirmation dataset.

---

## Synthesis: Required Manuscript Re-structuring

To satisfy the reviewers and guarantee acceptance in IEEE Transactions on Vehicular Technology, the authors must execute the following structural revisions:

```
+----------------------------------------------------------------------------------------------------+
|                               MANUSCRIPT RE-STRUCTURING GUIDELINES                                 |
+--------------------------+-------------------------------------------------------------------------+
| Section                  | Prescribed Narrative & Empirical Focus                                  |
+--------------------------+-------------------------------------------------------------------------+
| Title                    | Change from "Multimodal V2V Beam Prediction" to:                        |
|                          | "Physical Realities of Multimodal Beam Forecasting in Vehicular mmWave  |
|                          | Communications: An Empirical Multi-Scenario Investigation"               |
+--------------------------+-------------------------------------------------------------------------+
| Abstract                 | Highlight the dual empirical discoveries:                               |
|                          | 1) The 84.7% Candidate Exclusion bottleneck.                            |
|                          | 2) The Alternative-Beam Paradox (-19.85 pp power drop with RGB).        |
|                          | Present the work as an essential corrective to multimodal hype.         |
+--------------------------+-------------------------------------------------------------------------+
| Section I (Intro)        | Frame mmWave V2V beam tracking as a physical alignment problem         |
|                          | governed by Doppler coherence (4.16 ms) and probing budgets.            |
+--------------------------+-------------------------------------------------------------------------+
| Section II (Related)     | Differentiate from prior DeepSense-V2V papers (Alrabeiah, Charan) by     |
|                          | identifying that they evaluated only single scenarios or exact hits.   |
+--------------------------+-------------------------------------------------------------------------+
| Section III (Methodology)| Detail the Swin-B3 reproduction, matched GPS control, and the physical |
|                          | 60 GHz link-budget model (overhead, effective throughput).             |
+--------------------------+-------------------------------------------------------------------------+
| Section IV (Results)     | Present the verified data: Top-5 exact gain vs. physical power loss;    |
|                          | K=20 vs K=40 candidate widening; Scenario 39 sensitivity.              |
+--------------------------+-------------------------------------------------------------------------+
| Section V (Discussion)   | Explain the mathematical cause of CE failure on spatial beam manifolds; |
|                          | outline power-aware loss functions as the path forward.                 |
+--------------------------+-------------------------------------------------------------------------+
```

### Recommendation
**MAJOR REVISION.** With this pivot, the manuscript provides an invaluable, reproducible service to the IEEE vehicular communications community.
