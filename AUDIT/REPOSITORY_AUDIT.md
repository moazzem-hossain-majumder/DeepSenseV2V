# REPOSITORY AUDIT: DeepSense-V2V Submission Readiness & Scientific Verification

**Authoritative Audit Date:** November 21, 2026  
**Auditor:** Senior IEEE Research Scientist, Wireless Communications Specialist, and Statistical Reviewer  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  
**Target Venue:** IEEE Transactions on Vehicular Technology (TVT) / IEEE Transactions on Wireless Communications (TWC)  

---

## 1. Executive Audit Overview

This audit provides an exhaustive, forensic examination of the entire DeepSense-V2V repository. It independently traces every major numerical claim, evaluates architectural and algorithmic implementations, audits physical-layer mathematical models, verifies statistical procedures, and classifies all research documents and experimental outputs.

### High-Level Audit Findings
1. **The Core Split in the Repository (Two Incompatible Manuscript Tracks):**  
   The repository currently contains two fundamentally conflicting research narratives:
   - **Track A (Authentic Empirical Critique — Authoritative):** Documented in `paper/V2V_BEAM_FORECASTING_MANUSCRIPT.md` and `paper/TVT/manuscript.tex`, supported by 299 executed development stages (`results/swin_comparison_v1/`, `results/widecheck_v1/stage267/`, `results/shortlist_transfer_v1/`). This track proves that adding RGB via standard cross-entropy fails to improve near-optimal power coverage (-19.85 pp) due to the **Alternative-Beam Paradox** (-25.28 pp drop), that **Analytic Geometry** achieves 78.04% Top-5 3 dB power success (outperforming deep learning by ~30 pp), and that **Candidate Exclusion** accounts for 84.7% of severe (>10 dB) tracking failures.
   - **Track B (Synthetically Injected PAR-Net / KM-ACC — Unverified & Contradictory):** Documented in `paper/IEEE_manuscript.md`, `ablation_results.csv`, `final_peer_review.md`, and `final_research_report.md`. Forensic inspection of `src/unified_benchmark.py` reveals that the reported headline numbers (PAR-Net: 81.10% success, +2.54 pp over geometry, $p = 0.0001$; KM-ACC: 76.0% overhead reduction) were **hardcoded Python tuples and synthetic cosine perturbations** rather than outputs of trained neural network checkpoints. No model weights or checkpoint files exist for PAR-Net.
2. **Physical-Layer Coherence Time Arithmetic Error:**  
   In `COMMUNICATION_COST_AUDIT.md` and `IEEE_REVIEWER_REPORT.md`, the Doppler coherence time was computed as:
   $$T_c \approx \frac{c}{2 f_c v} = \frac{3 \times 10^8}{2 \times 60 \times 10^9 \times 30} = 8.33 \times 10^{-5}\text{ s} = 83.33\,\mu\text{s}$$
   However, the text erroneously asserted **4.16 ms** (a 50x mathematical error). Furthermore, small-scale multipath phase coherence time ($83.3\,\mu\text{s}$) was conflated with spatial beam-alignment coherence time ($100\text{--}200\text{ ms}$).
3. **Communication Cost Grounding:**  
   The communication-cost tables treat theoretical/analytical assumptions derived from IEEE 802.11ad standards (100 $\mu$s per probe, 20 ms MAC frame, 0.5 ms for 5 beams, 25.6 ms for 256 beams) as if they were measured hardware parameters. They must be explicitly labeled as analytical system models.
4. **Statistical Degrees of Freedom:**  
   The 16,643 sliding validation windows are extracted from only **10 continuous vehicle recording blocks**. Due to high autocorrelation ($\rho_{\text{ICC}} \approx 0.75$), the effective independent sample size is $N_{\text{eff}} \approx 13.3$, not 16,643.
5. **Reserved Test Partition Hygiene:**  
   The held-out test partition (17,017 raw frames / 19,631 sliding windows) has **NEVER been accessed or opened (0 reads logged)** across all 299 development stages. The test split remains 100% pristine.

---

## 2. Document Classification & Authority Matrix

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                              DOCUMENT CLASSIFICATION TABLE                                              |
+------------------------------------------------+---------------+--------------------+-----------------------------------+
| Document Path                                  | Status        | Evidence Source    | Primary Role & Integrity Notes    |
+------------------------------------------------+---------------+--------------------+-----------------------------------+
| paper/V2V_BEAM_FORECASTING_MANUSCRIPT.md       | AUTHORITATIVE | Stages 1–299       | Empirically verified manuscript.  |
| paper/TVT/manuscript.tex                       | AUTHORITATIVE | Stages 1–299       | Formal LaTeX draft for IEEE TVT.  |
| paper/BENCHMARK_PROTOCOL_LEDGER.md             | AUTHORITATIVE | Pre-declared specs | Pre-registered benchmark gates.   |
| results/widecheck_v1/stage267/evidence.json    | AUTHORITATIVE | Stage 267 execution| 24 arm/fold/seed verified records.|
| results/swin_comparison_v1/stage299/           | AUTHORITATIVE | Stages 268–299     | 30 Swin-B3 fits, bootstrap contrasts|
| results/shortlist_transfer_v1/REPORT.md        | AUTHORITATIVE | Oracle diagnostic  | 84.7% candidate exclusion proof.  |
+------------------------------------------------+---------------+--------------------+-----------------------------------+
| paper/IEEE_manuscript.md                       | CONTRADICTORY | UNVERIFIED         | Claims PAR-Net 81.10% from csv.   |
| ablation_results.csv                           | FABRICATED    | unified_benchmark  | Hardcoded tuples in Python code.  |
| final_peer_review.md                           | OUTDATED/BIAS | ablation_results   | Reviews fabricated PAR-Net claims.|
| final_research_report.md                       | OUTDATED/BIAS | ablation_results   | Asserts PAR-Net completion.       |
| experimental_improvement_plan.md               | SUPERSEDED    | Conceptual design  | Planned PAR-Net experiments.      |
| src/unified_benchmark.py                       | FLAWED SCRIPT | Script containing  | Injects synthetic cosine & tuples.|
+------------------------------------------------+---------------+--------------------+-----------------------------------+
| COMMUNICATION_COST_AUDIT.md                    | DEFECTIVE     | Analytical model   | Contains 4.16 ms vs 83.3 μs error.|
| IEEE_REVIEWER_REPORT.md                        | PARTIAL       | Review synthesis   | Inherits 4.16 ms arithmetic error.|
| STATISTICAL_REVIEW.md                          | AUTHORITATIVE | Cluster derivation | Proves G=10 blocks, Neff ≈ 13.3.  |
| STAGE267_INDEPENDENT_AUDIT.md                  | AUTHORITATIVE | Stage 267 audit    | Solves -0.053 dB paradox.         |
| FINAL_RESEARCH_STATUS.md                       | AUTHORITATIVE | Master status      | Tracks 299 completed stages.      |
| README.md                                      | AUTHORITATIVE | Root portal        | Unified on 4 scenarios (36–39).   |
+------------------------------------------------+---------------+--------------------+-----------------------------------+
```

---

## 3. Discrepancy & Contradiction Analysis

### Contradiction 1: PAR-Net / KM-ACC Claims vs. Actual Empirical Evidence
- **Document A (`paper/IEEE_manuscript.md`):** Claims PAR-Net achieved an 81.10% Top-5 3 dB power success rate (+2.54 percentage points over geometry, $p = 0.0001$, 95% CI: $[+1.22\%, +3.85\%]$) and that KM-ACC reduced sweeping overhead by 76.0% with a 14.17% miss rate.
- **Document B (`paper/V2V_BEAM_FORECASTING_MANUSCRIPT.md` & `paper/TVT/manuscript.tex`):** Reports that deep learning models achieve 48.71% power success, severely lagging the 78.04% achieved by Analytic Geometry. Shows that multimodal cross-entropy causes a catastrophic drop in alternative-beam recovery (-25.28 pp) and that widening candidate sets from K=20 to K=40 yields only +0.011 pp visual gain.
- **Code Inspection Truth (`src/unified_benchmark.py`):**
  Lines 310–314 of `src/unified_benchmark.py`:
  ```python
  # Refined logit: geometry base + bounded kinematic confidence boost
  boost = 0.35 * np.cos(np.linspace(-np.pi/4, np.pi/4, 20))
  if yaw_rate[i] > 0.05:
      boost *= 1.2
  refined_logits[i, cand] += boost
  ```
  Lines 353–358 of `src/unified_benchmark.py`:
  ```python
  ablations = [
      ("A1", "PAR-Net w/o RGB (GPS + Geometry Only)", 81.10, 4.37, +2.54, "[+1.22, +3.85]", 0.0001),
      ("A2", "PAR-Net w/o Geometry Prior (Unconstrained)", 51.33, 8.42, -27.23, "[-31.45, -23.01]", 0.0001),
      ("A3", "PAR-Net w/o Main-Lobe Mask (Global 256-MSE)", 74.71, 5.02, -3.85, "[-5.42, -2.18]", 0.0001),
      ("A4", "PAR-Net w/o Intra-Panel Constraint (Cross-Panel Smoothing)", 77.20, 4.65, -1.36, "[-2.65, -0.07]", 0.0382),
  ]
  ```
  **Conclusion:** The PAR-Net / KM-ACC numbers were not generated by training neural network models. They are hardcoded values and synthetic cosine perturbations dumped into `ablation_results.csv`. **These claims cannot be trusted and must be expunged from the submission.**

---

### Contradiction 2: Physical-Layer Coherence Time Arithmetic
- **Claimed in `COMMUNICATION_COST_AUDIT.md` (Section 1) and `IEEE_REVIEWER_REPORT.md` (Section 1.1):**
  $$T_c \approx \frac{c}{2 f_c v} \approx \frac{3 \times 10^8}{2 \times 60 \times 10^9 \times 30} = 4.16\text{ ms}$$
- **Exact Mathematical Derivation:**
  $$f_d = \frac{v}{\lambda} = \frac{30\text{ m/s}}{0.005\text{ m}} = 6000\text{ Hz} = 6\text{ kHz}$$
  $$T_c = \frac{1}{2 f_d} = \frac{3 \times 10^8}{2 \times 60 \times 10^9 \times 30} = \frac{3 \times 10^8}{3.6 \times 10^{12}} = 8.333 \times 10^{-5}\text{ s} = 83.33\,\mu\text{s}$$
  The previously reported value of $4.16\text{ ms}$ is exactly 50 times too large!
- **Physical Interpretation Discrepancy:**
  - $83.33\,\mu\text{s}$ is the **small-scale multipath Doppler phase coherence time** for an isotropic channel.
  - In directional mmWave arrays, narrow beamforming filters out angular multipath spread, lengthening channel coherence.
  - More importantly, **spatial beam-alignment coherence time** (the time before relative angular movement causes the transceiver to exit a $5^\circ$ half-power beamwidth) is:
    $$\Delta t_{\text{beam}} = \frac{\text{HPBW}}{\dot{\theta}} \approx \frac{5^\circ \times (\pi / 180)}{v_\perp / d} \approx \frac{0.0873\text{ rad}}{10 / 20\text{ rad/s}} \approx 175\text{ ms}$$
  Conflating small-scale Doppler coherence with beam tracking duration is an immediate red flag for an IEEE reviewer.

---

### Contradiction 3: Deployed Hardware Measurements vs. Offline Replay
- Several documents describe probing overhead as "measured link efficiency."
- In reality, the DeepSense 6G testbed collected offline measurements. The candidate shortlist evaluation is an **offline replay using saved validation beam powers**. The 256-beam sweep latency ($25.6\text{ ms}$) and 5-beam probing latency ($0.5\text{ ms}$) are **analytical estimates** derived from IEEE 802.11ad/ay frame timing specifications ($100\,\mu\text{s}$ per beam probe in a 20 ms MAC transmission frame).
- Offline replay evaluates an **oracle sweep** (where the ground truth of all 256 beams is known to the evaluator). A deployed transceiver only measures the $K$ candidate beams it actually probes.

---

### Contradiction 4: Statistical Degrees of Freedom
- `paper/IEEE_manuscript.md` repeatedly claims: "Evaluated across 16,643 validation windows... providing strong statistical power."
- `STATISTICAL_REVIEW.md` correctly proved that the 16,643 windows are generated via 0.1 s sliding steps from only **10 continuous vehicle recording blocks**:
  - Scenario 36: 2 blocks
  - Scenario 37: 3 blocks
  - Scenario 38: 2 blocks
  - Scenario 39: 3 blocks
- With an intra-cluster correlation $\rho_{\text{ICC}} \approx 0.75$, the effective independent sample size is:
  $$N_{\text{eff}} = \frac{N \cdot G}{1 + (\bar{m} - 1)\rho_{\text{ICC}}} \approx 13.3\text{ independent runs}$$
  Statistical inference cannot claim $N=16,643$ independent degrees of freedom.

---

## 4. Comprehensive Numerical Claim-by-Claim Verification Ledger

```
+-----------------------------------------------------------------------------------------------------------------------------------------------+
|                                                    NUMERICAL CLAIM VERIFICATION LEDGER                                                        |
+-----+-------------------------------+---------------+-----------------------+----------+--------+----------------------+----------------------+
| No. | Claimed Phenomenon / Metric   | Exact Value   | Source File           | Stage    | Split  | Verification Source  | Status               |
+-----+-------------------------------+---------------+-----------------------+----------+--------+----------------------+----------------------+
| 1   | Swin-B3 Transfer Top-5 Exact  | +5.42 pp      | V2V_BEAM_MANUSCRIPT   | Stage 299| Valid  | contrasts.json:L38   | VERIFIED ON DISK     |
|     | Hit Gain with RGB vs GPS      | [-2.86,+9.98] |                       |          |        |                      |                      |
| 2   | Swin-B3 Transfer Top-5 3 dB   | -19.85 pp     | V2V_BEAM_MANUSCRIPT   | Stage 299| Valid  | contrasts.json:L118  | VERIFIED ON DISK     |
|     | Power Success Drop with RGB   | [-26.3,-11.3] |                       |          |        |                      |                      |
| 3   | Swin-B3 Alternative-Beam      | -25.28 pp     | V2V_BEAM_MANUSCRIPT   | Stage 299| Valid  | ranking_audit.json   | VERIFIED ON DISK     |
|     | Power Success Collapse        | [-30.6,-15.9] |                       |          |        |                      |                      |
| 4   | Analytic Geometry Baseline    | 78.04%        | V2V_BEAM_MANUSCRIPT   | Stage 267| Valid  | evidence.json (S267) | VERIFIED ON DISK     |
|     | Top-5 3 dB Power Success      |               |                       |          |        |                      |                      |
| 5   | Candidate Exclusion Share of  | 84.7%         | shortlist_transfer_v1 | Audit    | Valid  | REPORT.md (shortlist)| VERIFIED ON DISK     |
|     | Severe (>10 dB) Failures (K=20| (14.23/16.80%)|                       |          |        |                      |                      |
| 6   | Candidate Width K=40 Visual   | +0.011 pp     | widecheck_v1          | Stage 267| Valid  | REPORT.md (S267)     | VERIFIED ON DISK     |
|     | Gain over Matched GPS         | [-0.007,+0.037|                       |          |        | evidence.json        |                      |
| 7   | Stage 267 K=40 Visual Mean    | -0.053 dB     | widecheck_v1          | Stage 267| Valid  | REPORT.md (S267)     | VERIFIED ON DISK     |
|     | Power Loss Benefit vs Geo     | (4.964 vs 5.02|                       |          |        | (Outage artifact)    |                      |
| 8   | Number of Training Windows    | 54,604        | manifest.jsonl        | Preproc  | Train  | manifest.jsonl count | VERIFIED ON DISK     |
| 9   | Number of Validation Windows  | 16,643        | manifest.jsonl        | Preproc  | Valid  | manifest.jsonl count | VERIFIED ON DISK     |
| 10  | Number of Calibration Windows | 14,025        | manifest.jsonl        | Preproc  | Calib  | manifest.jsonl count | VERIFIED (UNTOUCHED) |
| 11  | Number of Test Windows        | 19,631        | manifest.jsonl        | Preproc  | Test   | manifest.jsonl count | VERIFIED (UNTOUCHED) |
| 12  | Number of Test Raw Frames     | 17,017        | split_manifest.jsonl  | Preproc  | Test   | File system audit    | VERIFIED (UNTOUCHED) |
| 13  | PAR-Net Top-5 3 dB Success    | 81.10%        | IEEE_manuscript.md    | N/A      | N/A    | ablation_results.csv | FABRICATED / UNVERIF |
| 14  | PAR-Net Gain vs Geometry      | +2.54 pp      | IEEE_manuscript.md    | N/A      | N/A    | ablation_results.csv | FABRICATED / UNVERIF |
| 15  | KM-ACC Sweeping Reduction     | 76.0% (61.5 b) | IEEE_manuscript.md    | N/A      | N/A    | unified_benchmark.py | SYNTHETIC FORMULA    |
| 16  | Small-Scale Coherence Time    | 4.16 ms        | COMMUNICATION_COST    | Deriv.   | Theory | Equation arithmetic  | ARITHMETIC ERROR     |
|     | at 60 GHz, 30 m/s             |                |                       |          |        |                      | (True value: 83.3 μs) |
+-----+-------------------------------+---------------+-----------------------+----------+--------+----------------------+----------------------+
```

---

## 5. Experimental Pipeline & Split Taxonomy

### 5.1 Dataset Scope and Partition Structure
- **Dataset:** DeepSense 6G V2V Track (Scenarios 36, 37, 38, 39; collected in Tempe, Arizona, USA).
- **Total Validated Rows:** 112,196 synchronized frames.
- **Window Formulation:** 5 historical observations at $t-8, t-6, t-4, t-2, t$ (1.0 s history sampled at 5 Hz), predicting target at $t+5$ (0.500 s future horizon).
- **Split Distribution (Zero Trajectory Leakage):**
  - **Training:** 54,604 windows across distinct vehicle runs.
  - **Validation:** 16,643 windows across 10 distinct recording blocks.
  - **Calibration (Reserved):** 14,025 windows. Zero reads logged.
  - **Held-Out Test (Locked):** 19,631 windows (17,017 raw frames). Zero reads logged.

### 5.2 Completed vs. Exploratory vs. Confirmatory Experiments
1. **Completed & Confirmatory on Validation Data:**
   - **Stages 268–299 (Swin-B3 Full-Codebook Reproduction):** 30 independent training fits across 5 partitions (pooled + 4 leave-one-scenario-out folds) and 3 random seeds (42, 7, 13). Evaluated under matched GPS controls. Fully reproducible from saved prediction arrays.
   - **Stage 267 (Widecheck K=20 vs. K=40):** 24 multi-seed transfer runs evaluating geometry-constrained candidate set widening and candidate access rates.
   - **Shortlist Transfer Audit:** Oracle power decomposition proving the 84.7% candidate exclusion bottleneck.
2. **Exploratory / Diagnostics (Not for Primary Claims):**
   - Resolution comparisons (96px vs. 192px native features, Stages 151–154).
   - Visual shuffle and grayscale interventions (Stages 62–71).
   - Sequence-level contribution analysis in Scenario 39.
3. **Unexecuted / Non-Existent:**
   - PAR-Net full GPU training across all 4 scenarios.
   - Real hardware RF measurement of probing overhead in a physical vehicle.
   - Any execution against the locked test set.

---

## 6. Software, Checkpoints & File Integrity

### 6.1 Checkpoint Integrity
- **Swin-B3 Checkpoints:** All 30 model checkpoints for Stages 268–299 are stored in `checkpoints/` and `results/swin_comparison_v1/`.
- **Widecheck Predictions:** All prediction `.npz` files for Stage 267 are preserved in `results/widecheck_v1/stage267/`.
- **PAR-Net Checkpoints:** **NONE exist.** Searching `checkpoints/` and `results/` yields zero `.pt` or `.pth` weights for PAR-Net.

### 6.2 Test Suite Status
- Running `python -m pytest tests/ -v` verifies that all **74 unit tests pass** (74 passed, 27 subtests passed, 0 failed).
- All core multi-scenario utilities (`src/multiscenario_windows.py`, `src/publication_data.py`, `src/swin_comparison_data.py`, `src/paths.py`, `src/dataset.py`) execute without errors.

---

## 7. Mandatory Action Items Before Phase 2

1. **Expunge Fabricated PAR-Net & KM-ACC Claims:**
   - Remove or deprecate `ablation_results.csv`, `final_peer_review.md`, and `final_research_report.md`.
   - Freeze `paper/IEEE_manuscript.md` and transition exclusively to `paper/V2V_BEAM_FORECASTING_MANUSCRIPT.md` / `paper/TVT/manuscript.tex` as the sole authoritative manuscript.
2. **Correct Physical-Layer Coherence Time Derivations:**
   - Correct $T_c = 83.33\,\mu\text{s}$ in `COMMUNICATION_COST_AUDIT.md` and `IEEE_REVIEWER_REPORT.md`.
   - Distinguish small-scale Doppler coherence from spatial beam alignment coherence ($\approx 175\text{ ms}$).
   - Re-frame all communication cost tables as standard-compliant analytical models rather than measured deployment statistics.
3. **Formalize Effective Sample Size ($N_{\text{eff}} \approx 13.3$):**
   - Qualify all validation claims by explicitly stating that the data originates from 10 continuous recording blocks across 4 scenarios.
4. **Preserve Test Set Lock:**
   - Ensure the 17,017 test frames remain completely unopened until the final revision protocol is officially approved by the author.
