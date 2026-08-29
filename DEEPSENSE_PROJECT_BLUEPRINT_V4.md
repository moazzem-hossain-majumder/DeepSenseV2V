# Trajectory-Adaptive Multimodal Beam Tracking on DeepSense 6G — Course-Thesis Project Blueprint

**A complete, self-contained guide for building this project by hand.**
Everything in this document — the title, problem statement, dataset, folder
structure, novelty positioning, and roadmap — is designed so you can
reconstruct the entire project yourself, understand every design decision,
and defend every claim to a supervisor, examiner, or (per your stated goal)
a telecom-sector recruiter reviewing your portfolio.

**Document status:** This blueprint restates the title, safe novelty claim,
and venue strategy from `DEEPSENSE_IEEE_NOVELTY_REAUDIT_V3.md`, which
remains the source of truth for that language. If this document and the
audit ever diverge, the audit governs.

---

## 0. What this project is, and why it's built this way

This project is built around a genuinely **multimodal RGB + GPS dataset**:
DeepSense 6G's Multi-Modal V2V Beam Prediction development data (Scenarios
36–39) — synchronized front/rear camera frames and GPS positions from a real
60 GHz vehicle-to-vehicle mmWave testbed, with measured 256-beam receive
power as ground truth. This satisfies a multimodal-thesis requirement
directly: every sample is a real (image, GPS, measured-power) tuple, not a
synthetically paired dataset.

Two things to know before you build anything:

1. **The dataset is a US-based testbed, not Bangladeshi data.** This means
   "Bangladesh" must stay a motivating context rather than a data-level
   claim: the study addresses a future-facing wireless network optimization
   problem using a real-world US V2V mmWave testbed, and does not measure a
   Bangladeshi mobile operator. Your BD-recruiter framing belongs in your
   introduction/discussion ("this pipeline's multimodal network-intelligence
   and reliability-aware AI approach is transferable to 5G/6G beam
   management generally"), not in your title or as an implied deployment
   claim.
2. **A version of this project (V2: power-aware static conformal beam
   selection) was already re-audited and found too close to existing
   published work** (Section 1b). The V3 formulation below — full
   power-profile prediction plus online trajectory-adaptive risk control —
   is the one to build. It is defensible and course-appropriate, but its
   novelty is provisional, not guaranteed, and needs periodic re-checking
   (Section 1b).

---

## 1. Formal Thesis / Project Title

**"Trajectory-Adaptive Power-Profile Conformal Beam Tracking from
Synchronized RGB and GPS for Reliable V2V mmWave Communications"**

A shorter version for slides/CV: **"Online Utility-Calibrated Multimodal
Beam Tracking for V2V mmWave Links"**

> **Why not put "Bangladesh" in the title:** the data is a US 60 GHz V2V
> testbed. Say so plainly in your abstract, and put your BD/telecom-industry
> framing in the introduction and discussion, where it belongs as
> motivation and transferable-expertise argument — not as an implied
> deployment or data claim.

---

## 1b. Novelty Check — What's Already Published in This Space

Before you build anything, know what already exists, because your
supervisor and any recruiter with ML/telecom background may already know
some of this.

| What already exists | Source |
|---|---|
| Original DeepSense 6G V2V beam-prediction task and official baseline (RGB/GPS → best beam). | DeepSense 6G challenge/task documentation |
| Multimodal Transformer temporal fusion for beam prediction (general case study, not this exact online-control angle). | "Multimodal Transformers for Wireless Communications: A Case Study in Beam Prediction," 2023 |
| **SCAN-BEST** — predicts a candidate beam subset, uses conformal risk control, a relaxed beam-optimality criterion, limited in-band training inside the candidate set, and weighted conformal calibration under covariate shift. | SCAN-BEST, 2025/2026 |
| Calibrated multimodal priors with adaptive probing budgets and a dB safety margin. | "Sensing-Assisted Adaptive Beam Probing With Calibrated Multimodal Priors and Uncertainty-Aware Scheduling," IEEE WCL, 2026 |
| Cross-modal loss-tolerant beam prediction under missing/degraded modalities. | CLBP, IEEE Transactions on Mobile Computing, 2026 |
| Adaptive multimodal masked-Transformer beam prediction robust to missing modalities. | AMBER, 2025 |
| Lightweight cross-environment vision-aided beam tracking. | "Lightweight Vision-Aided Beam Tracking for Cross-Environment mmWave Communications," 2026 |
| Multi-task learning (joint classification + auxiliary heads) for mmWave beam prediction. | "Multi-Task Learning for mmWave Transceiver Beam Prediction," IEEE OJ-COMS, 2025 |
| Conformal Risk Control and Conformal PID Control (the methodological foundations this project builds on, not beam-prediction-specific). | Methodological literature |

**What this means for you:** "predict the best beam (or a small candidate
set of beams) from RGB+GPS using conformal risk control" is already done,
done well, and already extended with adaptive probing, missing-modality
robustness, and weighted calibration under shift. A project that stops at
this is not separated enough from SCAN-BEST or the 2026 adaptive-probing
work to claim IEEE-level novelty — that was the finding of the V2 re-audit,
and it stands.

**The available, course-appropriate gap (already the basis of this
blueprint):**

1. **No prior work predicts the full future 256-beam power-*profile*** (the
   complete relative-dB-gap vector across all beams) from temporal RGB–GPS
   observations — prior work predicts a best beam or a candidate *subset*,
   not the full measured landscape.
2. **No prior work pairs that profile prediction with an *online*,
   trajectory-adaptive risk controller** — existing conformal work
   (including SCAN-BEST) calibrates a static threshold once; this project
   updates the threshold sequentially as the vehicle's route, lighting, and
   motion regime change.
3. **No prior work explicitly separates offline exchangeable coverage
   guarantees from online time-averaged reliability under trajectory
   dependence and drift** — most papers in this space either assume
   exchangeability implicitly or don't discuss it.

**Your actual differentiating claim, written out:** *"To the best of the
literature search completed on 21 July 2026, prior work has studied RGB–GPS
multimodal beam prediction, missing-modality robustness, conformal candidate
beam selection with relaxed optimality, weighted conformal calibration, and
uncertainty-adaptive probing. We did not find a prior study that predicts
the complete future 256-beam measured power-gap profile from temporal
RGB–GPS observations and uses an online trajectory-adaptive utility-risk
controller to maintain a target near-optimal-beam miss rate under
non-stationary real-world V2V conditions."* Keep this sentence (or your own
close paraphrase) in your abstract and introduction. **Do not** upgrade it
to "first-ever conformal beam selection" or similar — that specific claim is
false given SCAN-BEST's existence.

> **Verification reminder (mandatory, not optional):** re-run this
> literature/citation-chain search at least once every 3 months while
> actively working on the project, and again, mandatorily, 2–4 weeks before
> any submission (proposal, thesis defense, or paper) — regardless of when
> you last checked. If a new paper closes gap #1 or #2 above, update this
> section and your abstract's claim before writing further results.

---

## 2. Problem Statement

Exhaustive beam training over 256 receive beams increases alignment
overhead and latency in high-mobility mmWave V2V links. Existing DeepSense
studies and recent literature have already shown that sensing data can
predict a single optimal beam or a small calibrated candidate subset, using
conformal risk control, missing-modality robustness, weighted calibration
under shift, and adaptive probing budgets.

The unresolved question this project addresses is whether synchronized RGB
and GPS can predict the **future beam-power landscape** — not merely one
class or a static subset — and whether an **online utility-risk controller**
can adapt the candidate-set threshold *during* a non-stationary vehicle
trajectory, so that near-optimal-beam miss risk stays controlled with low
probing overhead as routes, lighting, and motion regimes change.

**Core research question:** Does full future beam-power-profile supervision,
combined with an online trajectory-adaptive risk controller, improve
candidate-set efficiency and rolling reliability over classification-only
prediction with static calibration — and how much of that improvement
survives leakage-safe, blocked, multi-seed evaluation?

**Supporting research questions:**

- **RQ1:** Does full beam-power-profile supervision improve candidate-set
  efficiency over best-beam classification alone?
- **RQ2:** Can online risk adaptation maintain rolling near-optimal coverage
  better than static CRC under trajectory drift?
- **RQ3:** At equal 1 dB or 3 dB miss risk, does RGB+GPS require fewer
  probes than RGB-only or GPS-only prediction?
- **RQ4 (stretch):** Does the proposed method improve effective spectral
  efficiency after probing and inference latency are counted?
- **RQ5 (stretch):** How robust is the method under day/night and
  held-out-scenario shift?

**Scope note (state this in your abstract):** the dataset is a real-world US
60 GHz V2V testbed. Bangladesh is the **motivating deployment context** —
transferable expertise in multimodal network intelligence, reliability-aware
AI, and 5G/6G beam management — not a claim about the data's origin or an
implied operator deployment.

---

## 3. Dataset

| Dataset | Records | Modality | Source | Access |
|---|---|---|---|---|
| **DeepSense 6G — Multi-Modal V2V Beam Prediction, Scenarios 36–39** | Confirmed raw per-scenario sample counts: Scenario 36 ≈ 24,800; Scenario 37 ≈ 31,000; Scenario 38 ≈ 36,000; Scenario 39 ≈ 20,400 (≈112,000 raw timestamped samples total, per prior published work using these scenarios). **This is raw timestamps, not usable training sequences** — once windowed into 5-past-timestep (`x1`–`x5`) → 1-future (`y1`) sequences, the actual sequence count will be smaller; confirm the exact number from the official development CSV in Phase 1, don't assume the raw counts transfer directly. | Front + rear RGB images and GPS positions of receiver and transmitter vehicles, synchronized; measured 60 GHz beam power (four 64-beam vectors = 256 beams) as ground truth. Collected with two moving vehicles at 60 GHz across Tempe, Phoenix, Scottsdale, and Chandler, Arizona, in both day and night conditions. | DeepSense 6G testbed / official challenge release | deepsense6g.net (registration required) |

### Task structure

| Element | Description |
|---|---|
| Input | Five synchronized past timestamps (`x1`–`x5`): receiver/transmitter GPS + front/rear RGB frames |
| Output timestamp | Future target `y1_unique_index` |
| Primary label | `y1_unit1_overall-beam` — the measured best beam index (1 of 256) |
| Prediction targets (this project) | (a) best-beam class; (b) full 256-value relative power-gap profile \(d_b = G^* - G_b\); (c) near-optimal label sets \(Y_\Delta = \{b : d_b \le \Delta\}\) for \(\Delta \in \{0,1,3\}\) dB |

### Annotation structure

Each development-sequence row includes: five input timestamps with GPS and
front/rear images at each; a future output timestamp; the future best-beam
label; and four 64-beam future power vectors, which must be concatenated
into one 256-vector **using the official code's ordering**, not an assumed
ordering (this is a Phase 1 feasibility gate, not optional).

### Required dataset access steps

1. Register and download Scenario 36 first — validate the full pipeline on
   one scenario before pulling 36–39 in full (large RGB archives).
2. Read the dataset's official license/usage terms before redistributing
   anything derived (images, processed embeddings, or model weights).
3. Place raw files under `data/raw/` (never commit RGB archives or GPS raw
   files to git — `.gitignore` them).
4. Cite the official DeepSense 6G testbed/task documentation as the dataset
   source in your methodology and data-availability statement.
5. **Before any modeling:** confirm that each `y1_unique_index` maps to all
   four 64-beam power vectors, reconstruct the 256-vector using the official
   ordering, and confirm the reconstructed argmax equals the supplied
   `y1_unit1_overall-beam` label for a random sample and, ideally, the full
   dataset.

---

## 4. Project Folder Structure

```
deepsense-beam-tracking/
├── data/
│   ├── raw/
│   │   ├── scenario36/ ... scenario39/       # RGB frames + GPS CSVs — DO NOT commit
│   │   └── dev_dataset.csv                    # Official development CSV (x1-x5, y1, beam powers)
│   └── processed/
│       ├── split_manifest.csv                 # Leakage-safe trajectory block assignments
│       ├── train.parquet
│       ├── val.parquet
│       ├── calib.parquet                      # Static calibration / online warm-start block
│       └── test.parquet                       # Chronological online test stream
├── notebooks/
│   ├── 01_eda_beam_reconstruction.ipynb       # Phase 1: vector ordering, argmax check, near-optimal set sizes
│   ├── 02_gps_features.ipynb                  # Phase 3: GPS branch exploration
│   ├── 03_rgb_features.ipynb                  # Phase 3: RGB branch exploration
│   ├── 04_baseline_models.ipynb                # Phase 4: B0-B4
│   ├── 05_profile_models.ipynb                 # Phase 5: P1-P3
│   ├── 06_candidate_sets.ipynb                 # Phase 6: fixed/static CRC
│   ├── 07_online_control.ipynb                 # Phase 7: integral / PID controllers
│   └── 08_evaluation.ipynb                     # Phase 8: metrics, bootstrap, shift tests
├── src/
│   ├── beam_reconstruction.py                  # Phase 1: reconstruct 256-vector, verify argmax vs label
│   ├── partitioning.py                         # Phase 2: leakage-resistant trajectory blocks, split_manifest
│   ├── gps_features.py                         # Phase 3: Cartesian coords, distance/bearing, velocity proxies
│   ├── rgb_features.py                         # Phase 3: image loading, resize, optional cached embeddings
│   ├── baseline_models.py                      # Phase 4: B0 geometric, B1 GPS-only, B2 RGB-only
│   ├── fusion_models.py                        # Phase 4: B3 concat/gated fusion, B4 multimodal Transformer
│   ├── profile_models.py                       # Phase 5: P1 classification-only, P2 profile-only, P3 multi-task
│   ├── candidate_sets.py                       # Phase 6: Top-k, static CRC, power-aware CRC
│   ├── online_controller.py                    # Phase 7: integral update, conformal-PID controller
│   ├── evaluate.py                             # Phase 8: reliability/telecom metrics, bootstrap CIs
│   └── train.py                                # Shared training loop, checkpointing, seeds
├── results/
│   ├── eda/
│   ├── near_optimal_set_sizes.png               # Feasibility-gate output — STOP/GO decision artifact
│   ├── ablation_table.csv
│   ├── reliability_curves.png
│   ├── telecom_utility_table.csv
│   └── significance_tests.csv
├── paper/
│   └── draft.md
├── requirements.txt
└── README.md
```

**How notebooks and `.py` files work together:** experiment in the
notebook, move working code into the matching `.py` file, import it back
in. The notebook stays a scratchpad; the `.py` file is the reproducible
source of truth.

---

## 5. Full Step-by-Step Roadmap

Each phase below is tagged **[Core]** or **[Stretch]** per item.
**Complete all Core items before starting any Stretch item.** Core alone
answers RQ1–RQ3 and is sufficient to defend as a thesis. Stretch items
strengthen the case for an IEEE-level venue (Section 9 of the audit) but
should only be attempted once Core results are clean.

### Phase 0 — Environment Setup **[Core]**
1. Python 3.10+, virtual environment, install `requirements.txt` (Section 6).
2. Confirm GPU access — the RGB CNN encoder is the only GPU-heavy component;
   GPS-only and downstream candidate-set/controller logic run cheaply on CPU
   once features are extracted.
3. Download Scenario 36 only, first, to validate the pipeline end-to-end
   before pulling the full 36–39 set.

### Phase 1 — Data Audit and Feasibility Gates **[Core — mandatory before any modeling]**
1. Confirm each `y1_unique_index` maps to all four 64-beam power vectors.
2. Reconstruct the 256-vector using the official ordering; confirm argmax
   equals the supplied best-beam label (random sample, then full sample).
3. Inspect the units and official average-power-loss (APL) definition
   before any dB transformation.
4. **Plot the number of beams within 1 dB and 3 dB of optimum per sample —
   this is the single most important checkpoint in the whole project.** If
   almost every sample has only one near-optimal beam, the candidate-set
   premise gives little benefit and RQ1/RQ2 need reframing before you build
   anything further.
5. **Check the best-beam class distribution across all 256 beams.** In
   real V2V geometry, usage is almost never uniform — a small number of
   beams likely account for most samples. Note the imbalance now: it
   affects whether P1's cross-entropy needs class weighting, and it means a
   high Top-1 accuracy alone can be misleading if a majority-class predictor
   already scores well. Report a majority-class baseline accuracy alongside
   any model result so readers can judge real lift.
6. Test whether full power-profile prediction is learnable at all, beyond
   **both** a geometry-only baseline **and** a trivial baseline that always
   predicts the training-set's mean power-gap profile (no input dependence
   at all). The mean-profile baseline is cheap to compute and is the true
   floor — if a trained model can't clear it, something in the pipeline is
   broken, not just underpowered.
7. Save all outputs to `results/eda/`.

### Phase 2 — Leakage-Resistant Partitioning **[Core]**
1. Sort by scenario and absolute indices.
2. Connect sequence rows sharing any `x` or `y` timestamp into trajectory
   blocks; merge close components using a guard interval.
3. Assign complete trajectory blocks — never overlapping frames — to
   training (55%), validation (15%), static calibration/warm-start (15%),
   and the chronological online test stream (15%), per scenario.
4. Save `split_manifest.csv`.

### Phase 3 — Input Preprocessing **[Core]**
1. **GPS branch:** local Cartesian coordinates, Tx–Rx relative position,
   distance/bearing, first differences/velocity proxies, optional
   acceleration/turning proxies, missingness flags. Fit normalization on
   training data only.
2. **RGB branch:** validate file paths, resize consistently, preserve
   front/rear identity, moderate train-only augmentation (avoid unrealistic
   horizontal flips unless beam labels/camera geometry are transformed
   consistently), optionally cache encoder embeddings to disk.

### Phase 4 — Baseline Models
1. **[Core] B1 — GPS-only** temporal model (GRU/TCN/Transformer over
   engineered GPS sequence).
2. **[Core] B3 — RGB–GPS fusion** baseline (CNN image features + GPS
   encoder + gated/concatenation fusion).
3. **[Stretch] B0** — geometric/GPS-extrapolation-only baseline.
4. **[Stretch] B2** — RGB-only temporal model.
5. **[Stretch] B4** — multimodal Transformer baseline (stronger prior-art-
   style temporal fusion).

### Phase 5 — Prediction Heads (the heart of RQ1)
1. **[Core] P1 — classification-only** model (256-class cross-entropy).
2. **[Core] P3 — proposed multi-task profile model** — shared multimodal
   backbone with a 256-class beam head, a 256-value power-gap head, and
   (optionally) a ranking loss preserving beam order. Tune task weights on
   validation data only.
3. **[Stretch] P2** — power-profile-only model (Huber/weighted regression
   loss), used to isolate the profile head's standalone contribution in
   ablations.
4. **[Stretch, but worth trying early] Physically-informed profile
   parameterization.** Beam index typically maps to a physical azimuth/
   elevation angle, so the true 256-value power profile is usually a smooth,
   often unimodal-ish curve over beam index rather than 256 independent
   numbers. Predicting 256 free values from a modest training set risks
   overfitting or a noisy profile that doesn't reflect this structure. Two
   cheap ways to exploit it: (a) add a smoothness penalty (e.g., a small
   total-variation or second-difference term over adjacent beam indices) to
   the profile loss; (b) predict a low-dimensional latent (e.g., a peak
   angle plus a small number of basis-function coefficients) and reconstruct
   the 256-value profile from it. Either is a legitimate, citable modeling
   contribution beyond raw regression, and (a) is a one-line loss-function
   change worth trying even if you don't have time for (b).

### Phase 6 — Candidate-Set Construction
1. **[Core]** Fixed Top-1/3/5/10/15 from class probabilities.
2. **[Core]** Static exact-label conformal risk control (CRC), warm-started
   from the calibration block.
3. **[Stretch]** Probability-mass thresholding, entropy-adaptive Top-k,
   APS/RAPS, static power-aware CRC, SCAN-BEST-inspired static CRC baseline.

### Phase 7 — Online Risk Control
1. **[Core]** Integral controller — this is, precisely, **Adaptive
   Conformal Inference (Gibbs & Candès, 2021)** applied to a general risk
   rather than exact-label coverage; cite it as such rather than presenting
   the update rule as original. Warm-start \(q_1\) from the calibration
   block, then update after each probed outcome —
   \(q_{t+1}=\operatorname{clip}(q_t+\eta(\ell_t-\alpha),q_{\min},q_{\max})\),
   where \(\ell_t=\mathbf{1}\{\min_{b\in C_t(q_t)}d_{t,b}>\Delta\}\).
   - Set \(q_{\min}=0\) dB and \(q_{\max}\) to the 99th-percentile gap
     observed in the calibration block (not an arbitrary large number) —
     this keeps the fallback regime data-driven.
   - Set the target risk \(\alpha\) (e.g., 0.1) and step size \(\eta\) on
     the validation block only; report a small grid (e.g., \(\eta \in
     \{0.01, 0.05, 0.1\}\)) and pick by validation rolling risk, not by
     eyeballing the test stream.
2. **[Stretch]** Conformal-PID-style controller — this is **Conformal PID
   Control (Angelopoulos, Candès, and Tibshirani, 2023)**, cite it as such.
   Compare stability, overshoot, average candidate-set size, and rolling
   risk against the integral controller.

### Phase 8 — Evaluation and Statistical Protocol
1. **[Core]** Prediction quality: Top-1/3/5 accuracy, profile MAE/RMSE (dB),
   rank correlation between predicted and measured profiles.
2. **[Core]** Reliability: exact/1 dB/3 dB inclusion, cumulative and
   rolling-window miss rate, deviation from target risk \(\alpha\).
3. **[Core]** Telecom utility: official average power loss, probes per
   decision, search reduction vs. 256 beams, outage probability.
4. **[Core]** Statistics: 3 seeds minimum, block bootstrap CIs on the Core
   comparisons (RQ1–RQ3). **Resample whole trajectory blocks, not
   individual frames** — bootstrapping at the frame level understates
   variance because frames within a trajectory are correlated, and it lets
   long trajectories dominate the resample if not handled at the block
   level. **Do not select \(\alpha\), \(\Delta\), task weights, or
   controller gains (\(\eta\), \(q_{\min}\), \(q_{\max}\)) by looking at
   final-test performance** — all of that is chosen on validation/
   calibration only, exactly once, before the test stream is touched.
5. **[Stretch]** Worst-trajectory/worst-scenario risk, inference/probing
   latency and effective spectral efficiency (RQ4), day/night and
   held-out-scenario robustness (RQ5), 5-seed protocol with paired block
   bootstrap across the full baseline set, SHAP/attention analysis.
6. **[Stretch]** Write the BD/telecom discussion section using the honest
   scope statement in Section 2 above — transferable expertise, not
   deployment.

If Core results are weak (e.g., P3 doesn't beat P1, or the online
controller doesn't beat static CRC after leakage-safe splitting), stop and
treat the project as a solid thesis and portfolio piece rather than pushing
toward a publication claim the data doesn't support.

---

## 5a. Guarantee and Claim Discipline

This section was in the original audit and blueprint and is restored here
because it is easy to accidentally drop once you're focused on
implementation — and getting it wrong is a real correctness error in a
conformal-methods project, not just a framing nitpick.

- **Static/offline CRC (Phase 6):** finite-sample conformal guarantees
  require exchangeability between calibration and test data, or the
  specific assumptions of whatever extension you use (e.g., weighted CRC
  under covariate shift). Chronological, non-overlapping trajectory
  splitting prevents *leakage*, but it does **not** by itself establish
  exchangeability — samples along a single vehicle trajectory are
  temporally correlated, which is a distinct issue from leakage. Don't
  conflate the two in your writeup.
- **Online control (Phase 7):** Adaptive Conformal Inference and Conformal
  PID Control have their own online, distribution-free guarantees on
  long-run *time-averaged* risk — but these are different in kind from the
  finite-sample, per-instance coverage guarantee of split conformal
  prediction. State exactly which theorem you're relying on and under what
  assumptions (e.g., bounded loss, the specific update rule used) before
  claiming any guarantee at all, rather than describing the result as
  "conformal" in a way that borrows the offline method's guarantee by
  implication.
- **Held-out scenario shift (RQ5, Stretch):** don't use "guaranteed
  coverage" language for day/night or held-out-scenario results unless you
  have a credible covariate-shift assumption and a defensible density-ratio
  estimate behind a weighted-CRC claim. Otherwise, report it as *empirical*
  shift robustness — a measured rolling miss rate under shift, not a formal
  guarantee.
- **In every case:** if you're not sure whether a formal guarantee applies,
  default to reporting the empirical, measured reliability (rolling miss
  rate, worst-trajectory miss rate) and describe it as exactly that. An
  honest empirical result is defensible; an overstated theoretical claim is
  the kind of thing an examiner with conformal-prediction background will
  catch immediately.

---

## 6. `requirements.txt`

```txt
torch>=2.1
torchvision>=0.16
opencv-python>=4.9
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.10
statsmodels>=0.14
matplotlib>=3.7
seaborn>=0.12
pyarrow>=14.0
utm>=0.7
tqdm>=4.66
jupyter>=1.0
mapie>=0.8
```

`mapie` (or `crepes` as an alternative) implements standard split-conformal
and CRC machinery — use it for the static/exact-label CRC baselines in
Phase 6 rather than hand-rolling the calibration-quantile logic, and reserve
your own implementation effort for the profile-threshold candidate sets and
the online controllers in Phase 7, which these libraries don't cover.

GPU notes: the RGB CNN encoder (Phase 3–4) is the only consistently
GPU-heavy step; if you cache image embeddings after a single pass, the
profile heads, candidate-set logic, and online controller (Phases 5–7) all
train and run cheaply, even on CPU. A single consumer GPU is enough for the
Core scope — you do not need multi-GPU or large-model fine-tuning budgets
for this project.

---

## 7. Roadblocks to Flag to Your Supervisor (with mitigations)

1. **A prior version of this project (V2) was already found too close to
   published work.** *Mitigation:* Section 1b — the V3 profile-prediction +
   online-control angle is explicitly positioned against SCAN-BEST and the
   2026 adaptive-probing work, not "predict a candidate beam set with
   conformal control." Confirm this framing with your supervisor before
   writing results.
2. **Near-optimal beam sets may turn out to be singleton at 1 dB/3 dB.**
   *Mitigation:* this is the Phase 1 feasibility gate, checked before any
   modeling. If it fails, reframe around a different \(\Delta\) or narrow
   the research question — don't discover this after months of model work.
3. **Beam power-vector ordering is not self-evident from the raw files.**
   *Mitigation:* verify reconstruction against the official code and the
   supplied best-beam label before any modeling (Phase 1, item 2).
4. **The full scope (8 base models, 2 controllers, many ablations) is large
   enough to be a multi-year workplan if treated as uniformly mandatory.**
   *Mitigation:* the Core/Stretch tagging throughout Section 5 — defend on
   Core alone; treat Stretch as time-permitting.
5. **The novelty claim has a shelf life.** *Mitigation:* the mandatory
   re-check schedule in Section 1b (every 3 months, and again 2–4 weeks
   before submission).
6. **BD relevance is motivational, not data-level** — the dataset is a US
   testbed. *Mitigation:* state this explicitly in your abstract and
   introduction every time (Section 2 scope note), and never imply
   compatibility with a named Bangladeshi operator without operator data.
7. **Trajectory dependence complicates formal conformal guarantees.**
   *Mitigation:* Section 5a — report online time-averaged/rolling empirical
   reliability rather than claiming a formal exchangeability-based
   guarantee, unless a cited online-conformal theorem applies under your
   exact implementation.
8. **A 256-value regression target from a modest real-world dataset risks
   overfitting to a noisy, non-physical profile shape.** *Mitigation:* the
   smoothness-penalty option in Phase 5, item 4 — a one-line addition to
   the loss function that exploits the fact that beam index typically
   corresponds to a physical angle, and a good sanity check even if you
   don't build the full low-dimensional parameterization.
9. **It's easy to accidentally tune \(\alpha\), \(\Delta\), or controller
   gains against the final test stream without noticing** (e.g., "trying a
   few \(\eta\) values to see what looks good" while already looking at
   test-stream plots). *Mitigation:* Phase 8, item 4 — fix these on
   validation/calibration only, before the test stream is touched, and keep
   a written log of when the test stream was first evaluated.

---

## 8. What To Do Next

1. Download Scenario 36 and the official development CSV — nothing else
   yet.
2. Run Phase 1 in full: reconstruct the beam-power vectors, confirm the
   argmax matches the official label, and plot near-optimal set sizes at
   1 dB and 3 dB. **Do not proceed past this step until you've looked at
   that plot.**
3. Build leakage-safe trajectory partitions (Phase 2) before any model
   comparison.
4. Train the two Core baselines (B1, B3) and the two Core prediction heads
   (P1, P3) first — get an end-to-end pipeline working end-to-end before
   adding Stretch models or the second controller.
5. Show your supervisor the Section 1b novelty positioning and the Phase 1
   feasibility-gate plot before committing significant time to modeling —
   confirm both clear your course's bar.
6. Come back once you have the Phase 1 plot and a first working baseline
   number, and I can help interpret results or adjust the plan.

---

## 9. Progress Checklist

**Completed:**
- [x] V2 novelty re-audited; found too close to SCAN-BEST/adaptive-probing
      work
- [x] V3 (profile prediction + online risk control) formulated and
      novelty-checked against published baselines
- [x] Core/Stretch scope tiers defined
- [x] Blueprint restructured with folder structure, roadmap, and roadblocks

**Still to do (in order):**
- [ ] Register on the DeepSense 6G portal, download Scenario 36
- [ ] Set up environment, confirm GPU access
- [ ] Write `src/beam_reconstruction.py`, run Phase 1 feasibility gates
- [ ] **Look at the near-optimal-set-size plot and decide go/no-go**
- [ ] Write `src/partitioning.py`, build `split_manifest.csv`
- [ ] Write `src/gps_features.py` and `src/rgb_features.py`
- [ ] Write `src/baseline_models.py` and `src/fusion_models.py` (B1, B3 first)
- [ ] Compute majority-class and mean-profile trivial baselines (Phase 1)
- [ ] Write `src/profile_models.py` (P1 vs P3 comparison — Core deliverable);
      try the smoothness-penalty loss variant (Phase 5, item 4) early
- [ ] Write `src/candidate_sets.py` (fixed Top-k, static CRC via `mapie`)
- [ ] Write `src/online_controller.py` (ACI/integral controller first; set
      \(\alpha\), \(\eta\), \(q_{\min}\), \(q_{\max}\) on validation only)
- [ ] Write `src/evaluate.py`, run Core statistical protocol (trajectory-
      block bootstrap, not frame-level)
- [ ] Write the guarantee-language section of your methodology per Section 5a
      before drafting results
- [ ] Download Scenarios 37–39 and repeat at full scale
- [ ] Attempt Stretch items only if Core results are clean and time remains
- [ ] Final literature re-check immediately before submission
