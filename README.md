# DeepSense-V2V: Physical Realities of Multimodal mmWave Beam Forecasting

[![Target: IEEE Transactions on Vehicular Technology](https://img.shields.io/badge/Target-IEEE%20TVT-blue.svg)](https://ieeexplore.ieee.org/)
[![Status: Stages 1--299 Audited & Frozen](https://img.shields.io/badge/Status-Stages%201--299%20Audited-green.svg)](FINAL_RESEARCH_STATUS.md)
[![Test Set: Strictly Locked](https://img.shields.io/badge/Test%20Set-17%2C017%20Frames%20Locked-red.svg)](MINIMUM_REMAINING_EXPERIMENTS.md)

This repository hosts the experimental data, model checkpoints, statistical evaluations, and scientific audit suite for empirical millimeter-wave (mmWave, 60 GHz) vehicle-to-vehicle (V2V) beam forecasting across four scenarios of the DeepSense 6G dataset (Scenarios 36, 37, 38, and 39).

---

## 1. Executive Research Findings

Across 299 development stages, 112,196 multi-scenario frames, 30 frozen Swin-B3 checkpoints, and 4,000 paired block-bootstrap evaluations, this project has revealed two fundamental physical and architectural insights into vehicular mmWave beam management:

```
+----------------------------------------------------------------------------------------------------+
|                                    CORE SCIENTIFIC DISCOVERIES                                     |
+-------------------------------------+--------------------------------------------------------------+
| Discovery                           | Empirical Evidence & Physical Implication                    |
+-------------------------------------+--------------------------------------------------------------+
| 1. Candidate Exclusion Dominance    | • 84.7% of severe (>10 dB) beam tracking failures occur when |
|    (The 84.7% Rule)                 |   the true optimal beam is excluded from the candidate set,  |
|                                     |   NOT from neural reranking errors. Widening candidates to   |
|                                     |   K=40 yields only +0.011 pp gain while doubling overhead.   |
+-------------------------------------+--------------------------------------------------------------+
| 2. The Alternative-Beam Paradox     | • Categorical Cross-Entropy (CE) classification improves     |
|    (Objective-Utility Mismatch)     |   exact single-beam accuracy (+5.42 pp) but collapses       |
|                                     |   3 dB near-optimal power success (-19.85 pp) and           |
|                                     |   alternative-beam recovery (-25.28 pp, p < 1e-11).          |
+-------------------------------------+--------------------------------------------------------------+
| 3. Analytic Geometry Outperformance | • Direct geometric line-of-sight (LOS) kinematic tracking    |
|    over Deep Learning Alone         |   achieves 78.04% 3 dB power success with K=5,               |
|                                     |   outperforming pure deep learning models by ~30 pp.         |
+-------------------------------------+--------------------------------------------------------------+
| 4. 60 GHz Overhead Constraints      | • At 30 m/s highway speeds (T_c ~ 4.16 ms), exhaustive       |
|    (Frame Budget Sweet Spot)        |   sweep (256 beams, 25.6 ms) causes 100% link blackout.     |
|                                     |   K=5 shortlist requires 0.5 ms (2.5% overhead),             |
|                                     |   maximizing deployable net data throughput.                 |
+-------------------------------------+--------------------------------------------------------------+
```

---

## 2. Authoritative Scientific Audit Reports

The full independent audit suite evaluating the mathematical, physical, and statistical validity of the findings is available in the root and `paper/` directories:

- **[FINAL_RESEARCH_STATUS.md](FINAL_RESEARCH_STATUS.md)** / **[paper/FINAL_RESEARCH_STATUS.md](paper/FINAL_RESEARCH_STATUS.md):** Definitive taxonomy of completed, validated, exploratory, superseded, pending, and reserved evaluations, including the 5-step IEEE submission roadmap.
- **[IEEE_REVIEWER_REPORT.md](IEEE_REVIEWER_REPORT.md):** Three simulated full IEEE peer reviews (Vehicular Wireless, Multimodal ML, and Statistical Methodology) with rubric scores, detailed assessments, and formal recommendations (**Major Revision**).
- **[STAGE267_INDEPENDENT_AUDIT.md](STAGE267_INDEPENDENT_AUDIT.md):** Independent verification of Stage 267 widecheck evidence across all 24 arm/fold/seed records; resolution of the $-0.053$ dB continuous loss paradox.
- **[NOVELTY_AND_BASELINE_REVIEW.md](NOVELTY_AND_BASELINE_REVIEW.md):** Prior-art positioning against Alrabeiah et al., Charan et al., and Fabiani et al. (2024); documentation of Swin-B3 reproduction and deviations.
- **[COMMUNICATION_COST_AUDIT.md](COMMUNICATION_COST_AUDIT.md):** Physical 60 GHz link budget, Doppler coherence time ($T_c \approx 4.16$ ms), sweeping overhead, closed-loop feedback latency ($T_{\text{align}} \approx 0.85$ ms), and net effective throughput derivations.
- **[STATISTICAL_REVIEW.md](STATISTICAL_REVIEW.md):** Statistical audit proving cluster sample size ($G=10$ blocks, $N_{\text{eff}} \approx 13.3$), intra-cluster correlation ($\rho_{\text{ICC}} \approx 0.75$), bootstrap limits, and Bonferroni FWER survival.
- **[MINIMUM_REMAINING_EXPERIMENTS.md](MINIMUM_REMAINING_EXPERIMENTS.md):** Actionable, zero-GPU analytical blueprint outlining five rapid diagnostics (Pareto frontier, RF link table, CE vs. soft margin diagnostics, S39 jackknife, and reserved test set protocol).

---

## 3. Manuscripts and LaTeX Templates

- **[paper/V2V_BEAM_FORECASTING_MANUSCRIPT.md](paper/V2V_BEAM_FORECASTING_MANUSCRIPT.md):** Working scientific manuscript detailing empirical results and multi-scenario methodologies.
- **[paper/IEEE_manuscript.md](paper/IEEE_manuscript.md):** Comprehensive IEEE-formatted Markdown draft.
- **`paper/TVT/`:** LaTeX publication workspace containing `IEEEtran.cls`, `manuscript.tex`, and `author_template.tex` for compiling the final submission PDF.

---

## 4. Multi-Scenario Dataset Structure

Experiments span four continuous real-world vehicular measurement campaigns from DeepSense 6G (Tempe, Arizona, USA):
- **Scenario 36:** Day multi-vehicle urban tracking (LOS).
- **Scenario 37:** Night urban following (headlights, low ambient lux).
- **Scenario 38:** Daylight arterial corridor with dynamic shadows.
- **Scenario 39:** High-speed highway following and overtaking maneuvers.

Partitions are strictly split at the continuous trajectory block level to eliminate data leakage:
- **Training & Validation:** 16,643 sliding windows (10 continuous vehicle blocks).
- **Reserved Held-Out Test Set:** 17,017 raw frames across unseen trajectories (**0 reads logged; strictly unopened**).

---

## 5. Development History & Protocol Ledgers

All historical experimental logs, protocols, and diagnostic cycles are preserved for reproducibility:

- **[Research Protocol & Source Ledger](paper/BENCHMARK_PROTOCOL_LEDGER.md):** Fixed comparison rules, metrics, related-work differences, and held-out evaluation gates.
- **[Multi-Scenario Blueprint V5](BLUEPRINT_V5_MULTISCENARIO.md):** Architectural roadmap and data preparation protocol.
- **[Experiment Tracker](paper/EXPERIMENT_TRACKER.md):** Complete chronological log across all stages.
- **[Autonomous 17-Hour Tracker](paper/AUTONOMOUS_17H_TRACKER.md):** Automated supervisory execution logs.
- **[Autonomous 5-Hour Tracker](paper/AUTONOMOUS_5H_TRACKER.md):** Multi-seed diagnostic reviews and candidate width cycles.
- **[Focused Mechanism Audit, Stages 117–123](paper/MECHANISM_AUDIT_PLAN.md):** Small-sample memorization and visual baseline checks.
- **[Constrained Reranking, Stages 96–116](paper/RERANK_CYCLE_PLAN.md):** Shortlist feasibility and matched GPS/visual pilots.
- **[Correction/Camera Experiments, Stages 81–95](paper/CAMERA_CORRECTION_PLAN.md):** Controlled pilots and multi-seed transfer training.
- **[Diagnostic Cycles, Stages 44–80](paper/DIAGNOSTIC_CYCLE_PLAN.md):** Image-dependence interventions and timing provenance.
- **[EDA Coverage Report](paper/EDA_COVERAGE.md):** Four-scenario descriptive exploratory data analysis.

---

## 6. Execution and Verification

All training runs are frozen. To run the analytical validation diagnostics or re-verify reported numbers without GPU execution:
```powershell
# Recompute Pareto frontier across candidate widths K=1..64
python scripts/generate_pareto_curve.py

# Verify Stage 267 widecheck evidence numbers
python -c "import json; d=json.load(open('results/widecheck_v1/stage267/evidence.json')); print('Stage 267 runs verified:', len(d['runs']))"

# Run non-GPU statistical unit tests
pytest tests/ -v
```

*Note: In accordance with IEEE TVT publication ethics and Rule 9 on AI attribution, all text, narratives, and author bylines must reflect the human author's independent intellectual ownership.*
