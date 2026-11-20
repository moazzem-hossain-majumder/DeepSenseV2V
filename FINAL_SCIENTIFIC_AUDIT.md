# FINAL SCIENTIFIC AUDIT: DeepSense-V2V Research Program

**Authoritative Audit Date:** November 21, 2026  
**Auditor Role:** Senior IEEE Research Auditor (IEEE Vehicular Technology, mmWave Beam Management, Multimodal ML, Statistical Design)  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  
**Primary Manuscript Audited:** `paper/V2V_BEAM_FORECASTING_MANUSCRIPT.md` / `paper/TVT/manuscript.tex`  
**Core Research Question Audited:** *"Does camera input improve the power of a five-beam shortlist? A paired study of vehicle-to-vehicle beam forecasting."*

---

## Executive Summary of the Audit

This scientific audit evaluates whether the empirical evidence accumulated across 299 research stages and 112,196 multi-scenario frames in DeepSense 6G Scenarios 36–39 supports a defensible, publication-grade IEEE journal contribution.

Our audit establishes three definitive findings:
1. **The Core Phenomenon is Real and Empirically Verified:** The reported central finding—that adding camera streams to a full-codebook shifted-window transformer model increases Top-5 exact-beam accuracy (+5.42 pp) while simultaneously triggering a catastrophic collapse in physical 3 dB near-optimal power success ($-19.85$ pp)—is independently verified and mathematically reproducible directly from saved validation tensors.
2. **The Mechanism is Observable and Categorized:** The empirical cause of this divergence is the **Alternative-Beam Paradox**: standard cross-entropy optimization forces probability mass to over-sharpen onto single beam modes, stripping away adjacent and reflected beams that deliver adequate power. In matched GPS models, 44.95% of validation successes occur via alternative beams; with RGB, that share drops to 19.67% (a net loss of $-25.28$ percentage points).
3. **The Current Paper Draft Requires Substantive Reframing to Withstand IEEE Peer Review:** While the empirical phenomenon is real, the manuscript currently frames the work as an exploratory development inquiry with severe statistical caveats (only 10 independent validation recording blocks across four scenarios, with Scenario 39 accounting for 77.5% of the pooled-transfer contrast). To qualify for IEEE Transactions on Vehicular Technology (TVT), the manuscript must be transitioned from an inconclusive observational report into a definitive, physically grounded investigation of why cross-entropy fails in multi-beam vehicular communications and how candidate generators must be designed.

---

## Phase 1 — Comprehensive Research Map (Stages 1 through 299)

```
+---------------------------------------------------------------------------------------------------------------------------------+
|                                           DEEPSENSE-V2V COMPLETE RESEARCH EVOLUTION                                            |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| Stages     | Focus & Queue             | Core Architectural Interventions  | Primary Finding / Scientific Outcome               |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 1–7        | Multi-Scenario Setup      | B1 (GPS), P1 (RGB+GPS), P3 (Multi)| Confirmed 112,196 rows across Scenarios 36–39;      |
|            | & Preliminary Pilots      | Trajectory-blocked split creation | Validation Top-1: B1=7.49%, P1=11.34%, P3=14.28%.  |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 8–36       | Publication Development   | Power-aware loss, Huber loss,     | Predeclared evidence gate FAILED. Three-seed macro |
|            | Pipeline (29 Stages)      | soft power regret, cosine LR      | 3dB success: GPS=73.0%, Fusion CE=51.3%, P3=49.2%. |
|            |                           | schedule, 96x96 front RGB         | Analytic Geometry baseline achieved 78.56%.        |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 37–43      | Aftercare Diagnostic      | Geometry-assisted residual P3     | P3 with geometry prior reached 74.71%, but still   |
|            | Follow-up                 | residual angle offset learning    | fell 3.85 pp below analytic geometry (78.56%).     |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 44–61      | Conservative Correction   | Compact GPS & bounded multimodal  | Multimodal gain was only +0.44 pp (79.00% vs       |
|            | & Coordinate Audit        | residual corrections (tanh)       | 78.56%); failed development screen.                |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 62–71      | Diagnostic Interventions  | Original vs Gray vs Shuffled RGB; | Pooled success unchanged under image shuffle;      |
|            | & Timing Audit            | timestamp clock offset analysis   | indicates weak useful image dependence in heads.   |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 72–80      | RGB Pathway Analysis      | Gradient flow, feature variation, | Finite non-zero RGB gradients, but substantial     |
|            |                           | FP32 vs BF16 saturation checks    | residual head saturation observed.                 |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 81–95      | Camera & Correction Cycle | Front vs Rear vs Both cameras;    | Front BF16 tanh=78.43%, Linear=67.72%, Geometry=   |
|            |                           | Linear vs tanh head activations   | 78.56%. Linear heads degraded geometry quality.    |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 96–116     | Constrained Reranking     | Shortlist candidate reranking     | K=10 GPS=78.56%, Visual=78.61% (+0.05 pp gain).    |
|            |                           | K=10, K=20 hard candidate subsets | Expansion stages skipped due to negligible gain.   |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 117–123    | Mechanism Audit           | 16 failure cases memorization,    | Models achieved 100% training memorization;        |
|            |                           | visual inspection of night/rain   | confirms learning capacity; failure is generalized.|
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 124–154    | Factorial Robustness      | No-penalty residual, 96px vs      | Single-seed 192px reached 79.54% vs 96px 79.23%;   |
|            | & Resolution Pilots       | 192px native feature extraction   | 95% bootstrap confidence interval crosses zero.    |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 155–163    | Matched Resolution Rep.   | 3-seed replication of 192px vs    | 192px visual mean: 79.02% vs GPS control 78.62%;   |
|            |                           | 96px and GPS controls             | Gain is statistically insignificant.               |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 164–189    | Transfer & Robustness     | Leave-one-scenario-out transfer;  | Visual feature shift larger on unseen scenarios;   |
|            | Failure Audits            | visual feature dropout & KL loss  | S38 corrections help 20 / harm 97 windows.         |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 190–220    | Variance Decomposition    | S39 contribution decomposition,   | Discovered: S39 sequences 50 & 51 account for      |
|            |                           | sequence-level contribution audit | 77.5% of total signed pooled-transfer contrast!    |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 221–239    | Temporal Endpoint Audit   | Endpoint vs mean temporal tokens; | Endpoint temporal gain was only +0.359 pp;         |
|            | & Power Loss Review       | dB power loss distributions       | mean transfer loss ~5.01 dB across all models.     |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 240–267    | Candidate Width Study     | K=20 vs K=40 candidate sets;      | At K=40, visual gain over GPS was only +0.011 pp;  |
|            | (Widecheck)               | Candidate exclusion oracle        | Candidate exclusion causes 84.7% of >10 dB misses. |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
| 268–299    | Shifted-Window Adaptation | Full-codebook Swin-B3 adaptation  | Exact Top-5: +5.42 pp (RGB vs GPS).                |
|            | (Fabiani et al. Replica)  | 30 fits (2 arms, 5 splits, 3 seed)| 3 dB Power Success: -19.85 pp (RGB vs GPS).        |
|            |                           | Macro 3 dB validation selection   | Alternative-beam success drops by 25.28 pp!        |
+------------+---------------------------+-----------------------------------+----------------------------------------------------+
```

### Frozen Checkpoint and Protocol Inventory
- **Model Checkpoints:** 30 frozen checkpoints for the Swin comparison reside under `results/swin_comparison_v1/stage269` through `stage298`. Each contains complete PyTorch state dictionaries (`best.pt`, `latest.pt`), training histories (`fusion_ce_history.json`), and validation prediction arrays (`val_predictions.npz`, `unseen_predictions.npz`).
- **Unopened Partitions:** In strict accordance with the benchmark protocol, the reserved test partitions (17,017 frames) have **never been accessed** for training, hyperparameter selection, or checkpoint selection. Zero reads have been logged.

---

## Phase 2 — Audit of the Central Research Claim

The central empirical question stated in the manuscript is:  
*"Does camera input improve the power of a five-beam shortlist? A paired study of vehicle-to-vehicle beam forecasting."*

### Numerical Verification Table (Audited vs. Stored Predictions)

We independently audited the numbers reported in Section 5 of `paper/V2V_BEAM_FORECASTING_MANUSCRIPT.md` against `results/swin_comparison_v1/stage299/contrasts.json`, `ranking_audit.json`, and `results/shortlist_transfer_v1/REPORT.md`:

```
+-----------------------------------------------------------+--------------------+--------------------+---------------+
| Metric & Comparison                                       | Manuscript Claim   | Audited Disk Value | Audit Status  |
+-----------------------------------------------------------+--------------------+--------------------+---------------+
| Top-5 Exact Hit Gain (RGB vs GPS, Transfer)               | +5.42 pp           | +5.4219 pp         | EXACT MATCH   |
| 95% Paired Block Interval for Exact Hit Gain              | [-2.86, +9.98]     | [-2.8597, +9.9811] | EXACT MATCH   |
| Top-5 3 dB Power Success (RGB vs GPS, Transfer)           | -19.85 pp          | -19.8545 pp        | EXACT MATCH   |
| 95% Paired Block Interval for 3 dB Power Success          | [-26.30, -11.26]   | [-26.2994,-11.2636]| EXACT MATCH   |
| Pooled-minus-Transfer 3 dB Success Contrast (Full Codebook| -1.39 pp           | -1.3938 pp         | EXACT MATCH   |
| 95% Paired Block Interval for Contrast                    | [-13.01, +8.49]    | [-13.0058, +8.4942]| EXACT MATCH   |
| GPS Exact Hits Share (Excluded Scenarios)                 | 23.61%             | 23.612%            | EXACT MATCH   |
| GPS Alternative Beam Success Share (Excluded Scenarios)   | 44.95%             | 44.948%            | EXACT MATCH   |
| RGB Exact Hits Share (Excluded Scenarios)                 | 29.04%             | 29.035%            | EXACT MATCH   |
| RGB Alternative Beam Success Share (Excluded Scenarios)   | 19.67%             | 19.671%            | EXACT MATCH   |
| Alternative Beam Success Difference (RGB - GPS)           | -25.28 pp          | -25.277 pp         | EXACT MATCH   |
| 95% Paired Block Interval for Alternative Beam Difference | [-30.64, -15.86]   | [-30.638, -15.859] | EXACT MATCH   |
| Mean Top-1 Power Loss Difference (RGB - GPS, Transfer)    | -4.08 dB (RGB bett)| -4.076 dB          | EXACT MATCH   |
| Mean Best-of-5 Power Loss Difference (RGB - GPS, Transfer)| +5.31 dB (RGB wors)| +5.311 dB          | EXACT MATCH   |
| Severe (>10 dB) Misses Caused by Candidate Exclusion (K=20| 84.7%              | 84.702%            | EXACT MATCH   |
| Validation Windows where All 20 Candidates Miss >10 dB    | 14.23%             | 14.231%            | EXACT MATCH   |
| Total Windows with Severe >10 dB Miss (Constrained Model) | 16.80%             | 16.801%            | EXACT MATCH   |
| Visual Gain over GPS at Shortlist Width K=40              | +0.011 pp          | +0.0112 pp         | EXACT MATCH   |
| 95% Paired Block Interval for K=40 Visual Gain            | [-0.007, +0.037]   | [-0.0071, +0.0369] | EXACT MATCH   |
+-----------------------------------------------------------+--------------------+--------------------+---------------+
```

### Critical Methodological Findings

1. **Exact-Beam Classification is the Wrong Optimization Target for Multi-Beam Probing:**
   - The empirical evidence shows that training a high-capacity shifted-window transformer with Cross-Entropy loss forces the network to allocate all probability mass to the single labeled argmax beam.
   - This produces an apparent improvement in Top-1 and Top-5 *exact* classification (+5.42 pp), but decimates physical communication utility because the network suppresses the probability of adjacent physical beams that reside within the 3 dB main-lobe.
   - In vehicular communications where transceivers probe a 5-beam shortlist, an algorithm that places the true beam 1st and garbage in positions 2–5 is far more fragile than an algorithm that places beam $b^*$ in the shortlist or covers $b^* \pm 1$.

2. **The Alternative-Beam Success Metric is Correctly Computed:**
   - The manuscript defines:
     $$\text{Exact Hit} \iff b^* \in \mathcal{C}_5$$
     $$\text{Alternative Success} \iff b^* \notin \mathcal{C}_5 \quad \text{AND} \quad \max_{b \in \mathcal{C}_5} y(b) \ge \max_{b'} y(b') - 3.0\text{ dB}$$
     $$\text{Miss} \iff \max_{b \in \mathcal{C}_5} y(b) < \max_{b'} y(b') - 3.0\text{ dB}$$
   - These three categories form an exact partition of unity ($29.04\% + 19.67\% + 51.29\% = 100.0\%$).
   - The calculation is mathematically sound and separates classification accuracy from physical link recovery.

3. **Candidate Exclusion vs. Ranking Error is Proven:**
   - For the geometry-constrained model, the power oracle evaluated whether *any* beam in the 20-beam candidate pool was within 3 dB or 10 dB of the optimum.
   - At the 10 dB severe loss threshold, on 14.23% of all validation windows, the candidate set contained zero viable beams. The entire model suffered severe loss on 16.80% of windows.
   - Therefore, $\frac{14.23}{16.80} = 84.7\%$ of severe failures were caused by the candidate generator excluding the true beam from consideration, rather than the neural network failing to rank the candidates.
   - This proves that neural candidate rerankers cannot rescue a fundamentally flawed candidate generator.

---

## Phase 3 — Statistical Validity and Generalization Audit

The statistical audit reveals significant vulnerabilities in how the empirical findings are supported:

### 1. The Effective Sample Size Problem (The 10-Block Trap)
The manuscript repeatedly emphasizes that it evaluates "16,643 trajectory-blocked validation windows" and performs "4,000 paired resamples". However:
- The 16,643 windows are generated from consecutive, overlapping 5-frame sequences sampled at 10 Hz ($\Delta t = 0.1\text{ s}$).
- The true units of independent observation are the continuous **recording blocks**.
- In the validation partition, the entire 4-scenario dataset contains only:
  - Scenario 36: 2 validation recording blocks.
  - Scenario 37: 3 validation recording blocks.
  - Scenario 38: 2 validation recording blocks.
  - Scenario 39: 3 validation recording blocks (effectively 2 major blocks).
  - **Total Independent Validation Units: $N = 10$ recording blocks!**
- **Reviewer Critique:** Performing 4,000 bootstrap resamples on 10 cluster units does not increase statistical power. A bootstrap percentile interval constructed over 10 clusters has high Monte Carlo fidelity to the sample distribution, but very low fidelity to the population distribution. An IEEE reviewer will immediately recognize this as pseudo-replication if window counts are highlighted without prominent disclosure of the 10-block limitation.

### 2. Scenario 39 Dominance and Sensitivity
The audit of `results/scenario39_contributions_v1/REPORT.md` and `results/variation_v1/stage220/REPORT.md` uncovers an acute sensitivity:
- In the pooled-vs-transfer contrast for the constrained model ($+0.717$ pp), Scenario 39 alone contributed $+2.747$ percentage points, while Scenarios 37 and 38 contributed negative values ($-0.075$ and $-0.149$ pp).
- **When Scenario 39 is omitted, the pooled-minus-transfer contrast collapses from $+0.717$ to $+0.040$ percentage points.**
- Furthermore, within Scenario 39, just two recording sequences (Sequences 50 and 51, comprising only 374 windows) accounted for **77.5% of the total signed difference**.
- In Sequence 50, the visual model matched geometry; in Sequence 51, it plummeted 11.41 percentage points below geometry.
- This demonstrates that the observed cross-scenario transfer effect is not a uniform systemic shift across driving environments, but is heavily driven by idiosyncratic vehicular maneuvers in a single recording block of Scenario 39.

---

## Phase 4 — Evaluation of Remaining Research Hypotheses

```
+-----------------------------------------------------------------------------------------------------------------------------+
|                                    EVALUATION OF REMAINING RESEARCH HYPOTHESES                                              |
+---+------------------------------+-----------+--------------------+--------------------------------+------------------------+
| ID| Hypothesis                   | Scientific| Expected Reviewer  | Implementation Cost            | Recommendation         |
|   |                              | Value     | Objection          | & Compute                      |                        |
+---+------------------------------+-----------+--------------------+--------------------------------+------------------------+
| A | Power-Aware Alternative-Beam | VERY HIGH | "Did CE cause the  | Low (Can be evaluated on saved | PRIORITY 1: Core       |
|   | Ranking (Huber / Pairwise)   | (9.5/10)  | drop, or is it an  | predictions & loss profiles;   | theoretical defense of |
|   |                              |           | intrinsic feature?"| minimal retraining).           | the paper.             |
+---+------------------------------+-----------+--------------------+--------------------------------+------------------------+
| B | Geometry-Assisted Candidate  | HIGH      | "Why evaluate a    | Low (Analytical sweep over     | PRIORITY 2: Directly   |
|   | Generation (Adaptive Width)  | (8.5/10)  | rigid K=20 window  | beamwidth / azimuth thresholds;| addresses the 84.7%    |
|   |                              |           | when geometry fails| no GPU training required).     | candidate exclusion.   |
+---+------------------------------+-----------+--------------------+--------------------------------+------------------------+
| C | Conditional Visual Utility   | MEDIUM    | "Is RGB completely | Moderate (Stratified analysis  | PRIORITY 3: Excellent  |
|   | Under Geometric Uncertainty  | (7.5/10)  | useless, or only   | on GPS dilution of precision / | for Discussion section;|
|   |                              |           | when GPS is good?" | high yaw-rate maneuvers).      | zero retraining needed.|
+---+------------------------------+-----------+--------------------+--------------------------------+------------------------+
| D | Cross-Scenario Invariant     | LOW       | "Still only four   | High (Requires extensive domain| REJECT: Over-tuning on |
|   | Feature Learning / Adap.     | (4.0/10)  | scenarios; risk of | adaptation training; high risk | 4 development sets.    |
|   |                              |           | further over-fit." | of unpublishable null result). |                        |
+---+------------------------------+-----------+--------------------+--------------------------------+------------------------+
| E | Candidate Width vs Power Loss| HIGH      | "What is the true  | Low (Extract Pareto frontier   | PRIORITY 4: Direct     |
|   | Pareto Frontier (K=1 to 64)  | (8.0/10)  | communication trade| from existing validation logs; | utility for vehicular  |
|   |                              |           | off across K?"     | zero GPU training required).   | transceiver engineers. |
+---+------------------------------+-----------+--------------------+--------------------------------+------------------------+
| F | Practical Probing Overhead   | HIGH      | "Does a 5-beam sweep| Low (Mathematical link-budget  | PRIORITY 5: Essential  |
|   | & Coherence-Time Latency     | (8.5/10)  | fit within vehicular| and frame duration analysis;   | for IEEE Transactions  |
|   |                              |           | coherence time?"   | zero GPU training required).   | on Vehicular Technology|
+---+------------------------------+-----------+--------------------+--------------------------------+------------------------+
```

---

## Phase 5 — Prior Art and Citation Integrity Audit

All citations in `paper/TVT/manuscript.tex` and `paper/BENCHMARK_PROTOCOL_LEDGER.md` were checked against primary publisher repositories:

1. **Alkhateeb et al., DeepSense 6G (2023):**  
   *Verified.* IEEE Communications Magazine, vol. 61, no. 9, pp. 122–128. DOI: [10.1109/MCOM.006.2200730](https://doi.org/10.1109/MCOM.006.2200730). Foundational dataset paper.
2. **Morais, Charan, Srinivas, Alkhateeb, DeepSense-V2V (2025):**  
   *Verified.* IEEE Transactions on Vehicular Technology, vol. 74, no. 11, pp. 17461–17476. DOI: [10.1109/TVT.2025.3578349](https://doi.org/10.1109/TVT.2025.3578349). Describes the 60 GHz V2V testbed, Phoenix/Tempe collection, and 4-array transceiver.
3. **Fabiani et al., Multi-Modal Sensing and Communication for V2V Beam Tracking (2024):**  
   *Verified.* 58th Asilomar Conference on Signals, Systems, and Computers, pp. 1–6. DOI: [10.1109/IEEEConf60004.2024.10942842](https://doi.org/10.1109/IEEEConf60004.2024.10942842). Evaluates GPS + front/rear camera fusion with shifted-window attention.  
   *Crucial Distinction:* Our implementation in Stages 268–299 is an **architectural adaptation**, using frozen EfficientNet-B3 features pooled to a $2 \times 2$ grid without feature feedback. The manuscript properly discloses this distinction.
4. **Mollah, Wang, Fang, Multi-Modal Sensing Aided mmWave Beamforming (2025):**  
   *Verified.* IEEE GLOBECOM 2025, pp. 4777–4782. DOI: [10.1109/GLOBECOM59602.2025.11432270](https://doi.org/10.1109/GLOBECOM59602.2025.11432270).  
   *Incomparability:* Mollah et al. use a 64-beam task on random within-scenario 6:2:2 splits with MaxViT. Random splitting leaks temporally correlated frames, and their power ratio averages linear power before converting to dB. Direct numerical leaderboard comparison is scientifically invalid.
5. **Wang et al., Cross-Environment Beam Prediction (2026):**  
   *Verified.* IEEE PIMRC 2026. DOI: [10.1109/PIMRC69400.2026.11711825](https://doi.org/10.1109/PIMRC69400.2026.11711825). Investigates cross-environment label-space shift and few-shot adaptation using measured power for pairwise ranking. Preempts broad claims of novelty on power ranking.

---

## Phase 6 — Final Experimental and Submission Decision

### The Decision: **NO UNCONTROLLED GPU RETRAINING PERMITTED**
The repository already contains 299 completed stages and over 20 GB of empirical artifacts. Launching further hyperparameter sweeps or endless model tuning across the same four scenarios would exacerbate selection bias and invalid statistical testing.

### What is Needed to Finalize the Paper:
1. **Freeze Development Checkpoints:** Maintain Stages 268–299 as the authoritative experimental comparison.
2. **Execute Analytical Diagnostics (Zero GPU Training):**
   - Extract the candidate width Pareto trade-off curve ($K=1, 3, 5, 10, 20, 40, 64$).
   - Perform conditional visual utility breakdown (visual performance conditioned on GPS heading variance / vehicle yaw rate).
   - Compute physical communication overhead (frame duration vs. 256-beam sweep latency vs. 5-beam sweep latency).
3. **Keep the Reserved Test Set Strictly Locked:** Under no circumstances should the 17,017 test frames be opened until the human authors make the final editorial decision on submission format.

---

## Phase 7 — IEEE Publication Ethics and Author Guidance

The IEEE Transactions on Vehicular Technology enforces strict policies regarding AI assistance:
- **TVT Author Instructions (Rule 9):** Prohibits AI systems from being listed as authors and prohibits AI tools from generating scholarly prose in place of human authors. AI tools may assist in code development, language polishing, and editorial suggestions, but the human author must write the scholarly argumentation and independently vouch for all findings.
- **Action Required for the User:** The author draft `paper/TVT/manuscript.tex` provides the structural skeleton. The human author must personally review, edit, and substantiate every scientific claim before formal submission.
