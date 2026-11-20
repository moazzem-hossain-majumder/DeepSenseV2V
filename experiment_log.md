# Empirical Experiment Log

**Repository:** [moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  
**Execution Environment:** NVIDIA GeForce RTX 4060 (8 GB VRAM), CUDA 12.1, Python 3.11.9, PyTorch 2.5.1+cu121  
**Dataset Scope:** DeepSense 6G V2V Development Data (Scenarios 36, 37, 38, 39: 112,196 total rows; 104,903 5-step windows)  
**Evaluation Partition:** 16,643 validation windows across 80 trajectory blocks (zero frame overlap with training)  
**Random Seeds:** Fixed at 42, 7, 13  

---

## Chronological Experiment & Pipeline Stages

| Stage / Run ID | Model & Configuration | Execution Objective | Key Findings & Metric Outcomes |
| :--- | :--- | :--- | :--- |
| **Stage 1–7** | Multi-Scenario Pilot & Unattended Seeds | Initial multi-scenario baseline & 3-seed variance | Seed 42 Top-1: 15.64%, Seed 7: 13.36%, Seed 13: 14.13%. High inter-seed standard deviation ($1.16\%$). |
| **Stage 8–36** | Publication-Development Pipeline | 5 matched model families across 3 seeds on blocked splits | Geometry baseline: **78.56%** 3 dB power success. Learned GPS: 73.00%. RGB-only: 46.68%. Fusion CE: 51.33%. Profile: 48.54%. Power-aware: 49.21%. **Deep models lost to geometry by 27–30 percentage points.** Development gate failed. |
| **Stage 44–61** | Conservative Correction Cycle | Bounded residual correction ($\pm 0.25$ tanh) on geometry | Validation macro success: 79.00% vs 78.56% (+0.44% gain). Paired bootstrap interval [-2.56, +2.27] spans zero. In unseen Scenario 38, performance dropped to 66.16% vs geometry's 70.71%. |
| **Stage 62–71** | Diagnostic Cycle | Gray image & temporal shuffle ablation on frozen models | Validation accuracy changed by $<0.12\%$ with gray frames; shuffle test showed no degradation. Proved near-zero visual dependence in standard multimodal checkpoints. |
| **Stage 72–80** | Pathway & Gradient Audit | Training gradient flow & activation saturation | Identified severe tanh saturation in visual projection heads and low dynamic range in nighttime frames. |
| **Stage 81–95** | Camera & Multi-View Correction | Front vs. rear vs. dual camera linear corrections | Front linear: 67.72%, Rear linear: 68.21%, Dual linear: 73.19%. All underperformed geometry (78.56%). |
| **Stage 96–116** | Constrained Reranking | Visual reranking of top-10/20 geometry candidates | Top-10 GPS: 78.5617%, Visual reranker: 78.6137% (+0.052% gain; statistically indistinguishable from zero). |
| **Stage 117–123** | Mechanism & Visual Audit | 16 failure cases, memorization tests, image audit | Vehicle occupies only 5–10 pixels at $96\times 96$ resolution. Headlight blooming in night scenarios (37 & 39) overwhelms spatial cues. |
| **Stage 219 & 299** | Shortlist Oracle & Exclusion Audit | Attribution of severe power misses (>10 dB) | With candidate set size 20, **84.7% of all severe misses** were caused by *candidate exclusion* (near-optimal beam omitted from candidate set), whereas ranking errors inside candidate set accounted for only 2.57%. |
| **Stage 268–299** | Shifted-Window Attention (Fabiani et al.) | Full-codebook multi-camera temporal attention | Exact Top-5 hit rate increased (+5.42%), but **3 dB power success collapsed by -19.85 percentage points** (68.56% down to 48.71%) because cross-entropy over-sharpens on peak labels and destroys sidelobe coverage. |
| **New Evaluation: PAR-Net** | Physically Anchored Residual Network | Dual-head kinematic cross-attention + main-lobe loss | Achieved **80.21%** 3 dB success (Delta: +1.65% vs geometry). Ablation without RGB (A1) reached **81.10%** (+2.54% vs geometry, CI: [+1.22, +3.85], $p=0.0001$). Proved that the physical geometry anchor is essential. |
| **New Evaluation: KM-ACC** | Kinematically Modulated Adaptive Conformal Control | Online risk control with vehicle turning rate feedforward | Achieved 14.17% miss rate (target: 10.0%), reducing average candidate set size to 61.51 beams (76.0% search space reduction). Worst-trajectory miss rate: 46.52%. |

---

## Experimental Integrity Affirmation
All metrics reported above reflect execution on genuine physical sensor data (DeepSense-V2V). No data or results were synthetically fabricated. Test splits were kept strictly unopened prior to satisfying the pre-declared development gate.
