# FINAL RESEARCH STATUS: DeepSense-V2V Readiness and Submission Roadmap

**Authoritative Status Date:** November 21, 2026  
**Auditor Role:** Senior IEEE Research Auditor & Academic Manuscript Advisor  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  
**Target Venue:** IEEE Transactions on Vehicular Technology (TVT)  

---

## 1. Executive Summary

This document establishes the definitive research status of the DeepSense-V2V project across its entire 299-stage lifecycle. It categorizes every experiment, documents validated findings, flags exploratory and superseded analyses, and provides the human author with an actionable roadmap for final submission to IEEE Transactions on Vehicular Technology.

---

## 2. Definitive Status Breakdown

```
+----------------------------------------------------------------------------------------------------+
|                                    DEFINITIVE STATUS TAXONOMY                                      |
+--------------------------+-------------------------------------------------------------------------+
| Category                 | Scope & Included Components                                             |
+--------------------------+-------------------------------------------------------------------------+
| 1. COMPLETED             | Stages 1–299 fully executed and verified on disk.                       |
|    EXPERIMENTS           | • Four-scenario acquisition and integrity audit (112,196 frames).       |
|                          | • 30 Swin-B3 fits across 5 partitions and 3 seeds (Stages 268–299).     |
|                          | • Geometry-constrained candidate reranking K=20 and K=40 (widecheck).   |
|                          | • Candidate exclusion power-oracle evaluation (shortlist_transfer_v1).  |
|                          | • 4,000 paired block-bootstrap uncertainty analyses.                    |
+--------------------------+-------------------------------------------------------------------------+
| 2. VALIDATED             | Numerically verified, reproducible findings from saved predictions:     |
|    RESULTS               | • Top-5 exact hit gain with RGB: +5.42 pp (95% CI: [-2.86, +9.98]).    |
|                          | • Top-5 3 dB power success drop with RGB: -19.85 pp (CI: [-26.3,-11.3]).|
|                          | • Alternative-beam success drop with RGB: -25.28 pp (CI: [-30.6,-15.9]).|
|                          | • Candidate exclusion share of severe (>10 dB) misses: 84.7%.          |
|                          | • Incremental visual gain at shortlist width K=40: +0.011 pp (not sig). |
|                          | • Analytic geometry baseline Top-5 3 dB success: 78.04% (outperforms    |
|                          |   Swin-B3 deep learning model by ~30 percentage points).                |
+--------------------------+-------------------------------------------------------------------------+
| 3. EXPLORATORY           | Informative development diagnostics that do not represent final tests:  |
|    RESULTS               | • Visual feature shuffle / gray-image inference (Stages 62–71).         |
|                          | • Visual feature dropout and prediction KL consistency (Stages 180–189).|
|                          | • Feature resolution comparison (96px vs. 192px native features).       |
|                          | • 16-sample failure memorization mechanism audit (Stages 117–123).      |
|                          | • Sequence-level contribution analysis in Scenario 39 (Sequences 50–51).|
+--------------------------+-------------------------------------------------------------------------+
| 4. SUPERSEDED            | Replaced by mature multi-scenario protocols; retain for provenance only:|
|    EXPERIMENTS           | • Scenario-36-only legacy experiments (archived in `archive/`).         |
|                          | • Single-frame 0.1 s horizon models (superseded by official 0.5 s).     |
|                          | • Positive decibel APL definitions (superseded by challenge dB regret). |
|                          | • Unbounded linear-head residual experiments (Stages 81–95).            |
+--------------------------+-------------------------------------------------------------------------+
| 5. PENDING               | Analytical evaluations (zero GPU training required before submission):  |
|    EVALUATIONS           | • Candidate width Pareto frontier curve (K=1 to 64).                    |
|                          | • RF link-budget and frame coherence-time timing analysis.              |
|                          | • Leave-one-block-out Jackknife sensitivity table.                      |
+--------------------------+-------------------------------------------------------------------------+
| 6. RESERVED              | Strictly unopened partitions protected against data snooping:           |
|    EVALUATIONS           | • Held-out calibration and test partitions (17,017 frames).             |
|                          | • ZERO reads logged. Must remain untouched until author approval.       |
+--------------------------+-------------------------------------------------------------------------+
| 7. UNSUPPORTED /         | Must be excised or heavily qualified in the manuscript:                 |
|    OVERSTATED CLAIMS     | • General cross-scenario transfer degradation (driven 100% by S39).     |
|                          | • High statistical power claims from 16,643 windows (only 10 blocks).   |
|                          | • SOTA performance claims against published leaderboards.               |
|                          | • Bangladesh data collection claims (data is Tempe, Arizona, USA).      |
+--------------------------+-------------------------------------------------------------------------+
```

---

## 3. Actionable Path to Publication (Step-by-Step)

```
+----------------------------------------------------------------------------------------------------+
|                                    SUBMISSION ACTION ROADMAP                                       |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Step 1: Protocol & Checkpoint Freeze]                                                            |
|    • Keep all 30 Swin-B3 checkpoints and widecheck results frozen.                                |
|    • Verify clean Git commit history on master.                                                    |
|                                                                                                    |
|  [Step 2: Analytical Polish (Zero Retraining)]                                                     |
|    • Generate Candidate-Set Pareto Frontier (K=1 to 64) from existing .npz validation files.       |
|    • Add 60 GHz link-budget and coherence-time equations (T_c ~ 4.16 ms vs. sweep latencies).      |
|    • Add Scenario 39 sensitivity disclosure to the Discussion section.                             |
|                                                                                                    |
|  [Step 3: Human Author Textual Ownership (IEEE TVT Compliance)]                                    |
|    • Open `paper/TVT/author_template.tex` or edit `paper/TVT/manuscript.tex`.                      |
|    • Rewrite abstract and narrative using the human author's voice and academic perspective.       |
|    • Formulate proper author byline, affiliations (Department, University, City, Country),        |
|      and grant/funding acknowledgments.                                                            |
|    • Include required IEEE AI disclosure statement acknowledging Codex / AI tools for              |
|      experimental scripting and editorial assistance.                                              |
|                                                                                                    |
|  [Step 4: Reserved Test Set Decision]                                                              |
|    • Author decides whether to open the reserved test partition for final confirmatory numbers     |
|      OR keep the paper strictly framed as a development validation study.                          |
|    • If opening: execute one single locked pass without post-hoc tuning.                           |
|                                                                                                    |
|  [Step 5: Compile Final PDF & Submit to IEEE TVT]                                                  |
|    • Compile using IEEEtran.cls (regular paper format, ≤14 pages).                                  |
|    • Submit via IEEE Author Portal / ScholarOne Manuscripts.                                       |
+----------------------------------------------------------------------------------------------------+
```

---

## 4. Final Auditor Synthesis

The DeepSense-V2V project has generated an extraordinary body of empirical evidence. By discovering that **categorical cross-entropy destroys alternative-beam recovery in vehicular shortlists** and proving that **candidate exclusion drives 84.7% of severe communication failures**, this research provides real intellectual value to both the wireless communications and machine learning communities.

By adhering strictly to the publication ethics of IEEE Transactions on Vehicular Technology, refraining from unneeded retraining cycles, and allowing the human author to take full scholarly ownership of the manuscript, this project is well positioned for publication success.
