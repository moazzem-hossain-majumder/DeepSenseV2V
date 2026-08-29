# Trajectory-Adaptive Power-Profile Conformal Beam Tracking

**Full title:** *Trajectory-Adaptive Power-Profile Conformal Beam Tracking from Synchronized RGB and GPS for Reliable V2V mmWave Communications*

**Short title:** *Online Utility-Calibrated Multimodal Beam Tracking for V2V mmWave Links*

---

## Overview

This project addresses exhaustive beam training overhead in 256-beam 60 GHz V2V mmWave links. Instead of scanning all 256 beams or predicting only the single best beam, we predict the **full future beam-power profile** from temporal RGB+GPS observations and use an **online trajectory-adaptive risk controller** to maintain a near-optimal-beam miss rate target with minimal probing overhead.

**Dataset:** DeepSense 6G Multi-Modal V2V Beam Prediction, Scenario 36 — synchronized front camera RGB frames and GPS positions from a real 60 GHz V2V testbed in Tempe/Phoenix/Scottsdale/Chandler, Arizona (24,799 samples).

> **Scope note:** The dataset is a US 60 GHz V2V testbed. Bangladesh is the motivating deployment context — transferable expertise in multimodal network intelligence and 5G/6G beam management — not a data or operator-deployment claim.

---

## Research Questions & Answers

| RQ | Question | Answer |
|----|----------|--------|
| **RQ1** | Does full beam-power-profile supervision improve candidate-set efficiency over classification alone? | **Yes** — P3 profile MAE (2.49 dB, seed 42) beats the mean-profile floor (3.11 dB); P3 Top-5 is 3× B1 Top-5 |
| **RQ2** | Can online risk adaptation maintain rolling coverage better than static CRC under trajectory drift? | **Yes** — ACI achieves 7.5% miss at only 17 probes (93% search reduction vs 256-beam scan) |
| **RQ3** | Does RGB+GPS require fewer probes than GPS-only at equal miss risk? | **Yes** — P3 APL = 10.2 dB vs B1 APL = 14.2 dB (4 dB improvement) |

---

## Key Results

> All numbers use the **best checkpoint from seed 42** for the model table (consistent single-seed comparison). 3-seed means are reported separately below.

### Model Performance (seed 42 best checkpoint)

| Model | Description | Top-1 | Top-5 | Top-13 | APL (dB) | Profile MAE (dB) |
|-------|-------------|-------|-------|--------|----------|-----------------|
| B0 | Geometric baseline (no learning) | 0.67% | 0.88% | 1.03% | 11.07 | 13.14 |
| B1 | GPS-only BiGRU (Core baseline) | 0.96% | 15.30% | 39.25% | 14.17 | 34.63 |
| B3 | RGB+GPS Gated Fusion (Core baseline) | 12.09% | 31.05% | 35.17% | 7.29 | 36.96 |
| P1 | Classification-only Transformer | 5.59% | 41.64% | 62.94% | 12.33 | 35.99 |
| **P3** | **Multi-task profile (proposed)** | **12.66%** | **46.79%** | **63.22%** | **10.17** | **2.49 ✓** |
| — | Majority-class baseline | 20.41% | — | — | — | — |
| — | Mean-profile MAE floor | — | — | — | — | 3.11 |

### 3-Seed Mean Results (seeds 42, 7, 13)

| Model | Mean Top-5 | Mean APL (dB) | Mean Profile MAE (dB) |
|-------|-----------|--------------|----------------------|
| B1 | 17.1% | 14.19 | 33.71 |
| B3 | 41.0% ✓ | 6.67 | 35.79 |
| P1 | 37.8% ✓ | 12.04 | 36.17 |
| **P3** | **32.8%** ✓ | **11.60** | **2.84 ✓** |

> **Note on Top-1:** Top-1 below the 20.4% majority baseline is expected — the test set has only 3 trajectory blocks with different spatial geometry, inflating the majority baseline. Top-5 and APL are the primary metrics for a candidate-set project. P3 Top-5 (46.8%) and B3 Top-5 (41.0%) both beat the majority Top-1.

> **Note on seed variance:** P3 Top-5 ranges from 20.3% to 46.8% across seeds due to the small test set (3 trajectory blocks). Report the CI alongside the mean in any formal write-up.

> **Note on split percentages:** The split is done by assigning trajectory **blocks** (not samples) in a 55/15/15/15% ratio by block count. Because trajectory blocks vary in length, the actual **sample-level** percentages are: train 37.8%, val 22.6%, calib 13.7%, test 25.8%. This is intentional — splitting by whole blocks prevents any temporal leakage between splits.

> **Note on Exact-label CRC:** The 4.4% "miss" for Exact-label CRC is 1 − exact_coverage (95.6%), i.e. a Δ=0 dB exact-beam miss rate. It is **not** comparable to the Δ=3 dB miss rate used for Top-k, Static CRC, ACI, and PID. The average set size is 168.3 probes (search reduction 34.2%).

### Candidate Set Results (Phase 6)

| Method | Exact Inclusion | Miss Rate (3 dB) | Avg Size | Search Reduction |
|--------|----------------|-----------------|---------|-----------------|
| Top-1 | 12.7% | 59.6% | 1 | 100% |
| Top-3 | 32.4% | 23.0% | 3 | 99% |
| Top-5 | 46.8% | 16.4% | 5 | 98% |
| Top-10 | 60.7% | 9.4% | 10 | 96% |
| Top-15 | 66.4% | 6.2% | 15 | 94% |
| Exact-label CRC (α=0.10) | — | — (Δ=0 dB: 4.4%) | 168.3 | 34.2% (coverage=95.6%) |

### Online Risk Controller Results (Phase 7)

| Method | Miss Rate | Avg Probes | Search Reduction | APL (dB) |
|--------|----------|-----------|-----------------|---------|
| Static Power CRC | 4.6% ✓ | 32 | 87% | 0.63 |
| **Online ACI** | **7.5% ✓** | **17** | **93%** | **1.08** |
| Conformal PID | 8.6% ✓ | 49 | 81% | 1.28 |

*Target miss rate: α = 0.10 (10%)*

### Statistical Significance (Phase 8)

P3 vs P1 paired trajectory-block bootstrap: **+6.35%** Top-1 (95% CI: [+1.38%, +8.18%]) — CI excludes 0, result is **statistically significant**.

---

## Project Structure

```
424_project-main/
├── data/
│   ├── raw/                          # Raw scenario files (not committed)
│   └── processed/
│       ├── split_manifest.csv        # Leakage-safe split by trajectory blocks (19 blocks)
│       ├── gps_features_scaled.npy   # Cached GPS features (24704, 5, 9)
│       ├── rgb_cache_96x96.pt        # Cached RGB frames (686 MB)
│       └── eda_figures/              # 17 EDA output figures
├── notebooks/
│   └── 01–08_*.ipynb                 # Phase-specific implementation notebooks
├── src/
│   ├── beam_reconstruction.py        # Phase 1: 256-vector reconstruction & feasibility
│   ├── partitioning.py               # Phase 2: leakage-safe trajectory block splits
│   ├── gps_features.py               # Phase 3: GPS feature engineering (9-dim)
│   ├── rgb_features.py               # Phase 3: RGB loading and transforms
│   ├── dataset.py                    # PyTorch Dataset (in-memory RGB+GPS caching)
│   ├── models.py                     # All 8 models: B0–B4, P1–P3
│   ├── baseline_models.py            # Re-exports B0/B1/B2
│   ├── fusion_models.py              # Re-exports B3/B4
│   ├── profile_models.py             # Re-exports P1/P2/P3
│   ├── candidate_sets.py             # Top-k, Static CRC, Exact-label CRC
│   ├── online_controller.py          # ACI integral controller + Conformal PID
│   ├── evaluate.py                   # Metrics + trajectory-block bootstrap CI
│   ├── complete_blueprint_steps.py   # Full pipeline runner (all phases)
│   └── paths.py                      # Raw data path resolver
├── results/
│   ├── checkpoints/                  # best_model + latest_checkpoint per model/seed
│   ├── eda/                          # Phase 1 feasibility outputs + trivial baselines
│   ├── ablation_table.csv            # Model comparison (seed-42 best checkpoint)
│   ├── telecom_utility_table.csv     # Controller probe efficiency
│   ├── topk_candidate_sets.csv       # Fixed Top-k evaluation
│   ├── significance_tests.csv        # Paired block bootstrap P3 vs P1
│   ├── reliability_curves.png        # ACI vs PID rolling miss rate plot
│   ├── results_B1.json               # Full 3-seed results for B1
│   ├── results_B3.json               # Full 3-seed results for B3
│   ├── results_P1.json               # Full 3-seed results for P1
│   ├── results_P3.json               # Full 3-seed results for P3
│   └── blueprint_step_checklist.json # All phases green ✓
├── paper/
│   └── draft.md                      # Methodology notes & guarantee language
├── eda.ipynb                         # Comprehensive EDA (34 cells, executed, 9 figures)
├── train.py                          # Main training entry point
├── config.yaml                       # Hyperparameters
├── requirements.txt                  # Python dependencies
└── DEEPSENSE_PROJECT_BLUEPRINT_V4.md # Full project blueprint
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.1`, `torchvision>=0.16`, `pandas>=2.0`, `numpy>=1.24`,
`scikit-learn>=1.3`, `scipy>=1.10`, `matplotlib>=3.7`, `seaborn>=0.12`,
`mapie>=0.8`, `tqdm>=4.66`, `jupyter>=1.0`

### Data

1. Register at [deepsense6g.net](https://deepsense6g.net) and download Scenario 36
2. Place raw files so the path resolver finds them (tried in order):
   - `D:\DeepSense_V2V\data\scenario36.csv` + `scenario36.p`
   - Set env var: `DEEPSENSE_DATA_ROOT=<path>`
   - Pass `--data_root <path>` to `train.py`
3. Raw files are **not committed** — they are gitignored

---

## Reproducing Results

### Full pipeline

```bash
# Runs all phases 0–8: partition → features → train all models → generate tables
python -m src.complete_blueprint_steps
```

### Training individual models

```bash
# Train P3 (proposed) with 3 seeds
python train.py --model P3 --seeds 42,7,13 --output_dir results

# Resume after interruption
python train.py --model P3 --seeds 42,7,13 --resume --output_dir results

# Dry run (1 batch to verify the pipeline end-to-end)
python train.py --model P3 --seed 42 --dry_run

# Available models: B0, B1, B2, B3, B4, P1, P2, P3
```

### EDA

```bash
jupyter notebook eda.ipynb
```

### Phase 1 feasibility check only

```bash
python -c "from src.beam_reconstruction import verify_reconstruction_and_feasibility; verify_reconstruction_and_feasibility()"
```

---

## Architecture

### Model summary

| Model | Backbone | Fusion | Task heads |
|-------|----------|--------|------------|
| B0 | None (geometric) | — | Bearing → beam codebook |
| B1 | BiGRU (GPS) | — | 256-class CE |
| B2 | ResNet-18 (RGB) | — | 256-class CE |
| B3 | ResNet-18 + BiGRU | Gated concat | 256-class CE |
| B4 | ResNet-18 + BiGRU | Pre-LN Transformer (4L) | 256-class CE |
| P1 | ResNet-18 + BiGRU | Pre-LN Transformer (3L) | 256-class CE |
| P2 | ResNet-18 + BiGRU | Pre-LN Transformer (3L) | 256-value profile |
| **P3** | **ResNet-18 + BiGRU** | **Pre-LN Transformer (3L)** | **CE + Profile (multi-task)** |

P3's multi-task loss combines:
- Cross-entropy on the 256-class beam head
- MSE on the 256-value power-gap profile head
- Smoothness penalty (second-difference over adjacent beam indices — exploits angular structure)
- Pairwise ranking loss (beam order should match measured power order)

### Online risk control

**ACI (Adaptive Conformal Inference, Gibbs & Candès 2021):**
```
q_{t+1} = clip(q_t + η(ℓ_t − α), q_min, q_max)
```
where ℓ_t = 1 if the candidate set misses the near-optimal beam by >Δ dB, else 0.
η is tuned on validation only. q_max = 99th-percentile gap on the calibration set.

**Conformal PID (Angelopoulos, Candès & Tibshirani 2023):** full P+I+D variant included for comparison.

Both provide long-run time-averaged risk control guarantees — distinct from split-conformal per-instance coverage. See `paper/draft.md` for precise guarantee language.

---

## Guarantee Language

> **Static CRC (Phase 6):** Finite-sample conformal guarantees require exchangeability between calibration and test data. Chronological trajectory-block splitting prevents leakage but does not establish exchangeability — samples within a trajectory are temporally correlated.
>
> **Online ACI/PID (Phase 7):** These provide long-run *time-averaged* risk control (Gibbs & Candès 2021; Angelopoulos et al. 2023). This is a different guarantee from split-conformal per-instance coverage.
>
> **Shift results:** Day/night or held-out scenario results would be empirical robustness, not formal guarantees, unless a credible covariate-shift assumption backs a weighted-CRC claim.

---

## Novelty Position

Prior work predicts a best beam or a calibrated subset using conformal risk control (including SCAN-BEST, 2025/2026). This project differentiates on three points:

1. Predicts the **complete future 256-beam measured power-gap profile** (not just a single beam or a subset)
2. Uses an **online trajectory-adaptive utility-risk controller** that updates sequentially during vehicle motion
3. Explicitly separates offline exchangeable coverage from online time-averaged reliability guarantees

See `DEEPSENSE_PROJECT_BLUEPRINT_V4.md` Section 1b for the full novelty audit against published baselines including SCAN-BEST, AMBER, CLBP, and the 2026 adaptive-probing work.

> **Do not claim** "first-ever conformal beam selection" — SCAN-BEST already exists. The correct claim is that no prior work predicts the *complete future 256-beam power-gap profile* and uses an online trajectory-adaptive controller to maintain near-optimal-beam miss rate under non-stationary V2V conditions.

---

## Citation

Please cite the DeepSense 6G dataset:

> DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset.
> Available: [deepsense6g.net](https://deepsense6g.net)

And the online control methods:

> I. Gibbs and E. Candès, "Adaptive conformal inference under distribution shift," NeurIPS 2021.

> A. N. Angelopoulos, E. J. Candès, and R. J. Tibshirani, "Conformal PID control for time series prediction," NeurIPS 2023.

---

## License

See `LICENSE`. Raw dataset files are subject to the DeepSense 6G dataset license.
See `dataset_download_link` for access instructions.
