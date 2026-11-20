# REMAINING EXPERIMENT DECISION: Evaluation of Hypotheses A–F and Controlled Pre-Submission Roadmap

**Authoritative Audit Date:** November 21, 2026  
**Auditor Role:** Senior IEEE Research Auditor & Systems Architect  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  
**Target Venue:** IEEE Transactions on Vehicular Technology (TVT)  

---

## 1. Executive Decision: NO UNCONTROLLED GPU RETRAINING

Our definitive research audit decision regarding further experimental execution is:
> **DO NOT LAUNCH UNCONTROLLED GPU RETRAINING OR HYPERPARAMETER SWEEPS.**  
> The repository already contains 299 executed stages, 30 frozen Swin-B3 checkpoints, and comprehensive multi-scenario validation tensors. Running additional training cycles on the same four development scenarios will only increase post-hoc selection bias and degrade statistical credibility.  
> **All necessary empirical evidence to support a defensible IEEE contribution is ALREADY PRESENT in the repository's saved validation predictions.**

Instead of expensive and risky GPU training, we recommend a **strictly bounded, zero-GPU analytical evaluation protocol** that extracts deeper wireless communication insights from existing predictions.

---

## 2. Rigorous Evaluation of Hypotheses A through F

```
+------------------------------------------------------------------------------------------------------------------------------------+
| TABLE I: SYSTEMATIC EVALUATION AND SCIENTIFIC RANKING OF HYPOTHESES A THROUGH F                                                    |
+---+------------------------------+-----------+--------------------+---------------------+------------------+-----------------------+
| ID| Hypothesis Title             | Scientific| Expected Reviewer  | Computational       | Execution        | Recommended Decision  |
|   |                              | Value     | Objection          | Cost & Complexity   | Method           |                       |
+---+------------------------------+-----------+--------------------+---------------------+------------------+-----------------------+
| A | Power-Aware Alternative-Beam | 9.5 / 10  | "Is the drop due to| ZERO GPU compute    | Saved validation | PRIORITY 1: INCLUDE   |
|   | Ranking (Huber / Pairwise)   | (Top Val) | CE, or an intrinsic| (Evaluate existing  | prediction tensor| (Essential theoretical|
|   |                              |           | feature of vision?"| loss distributions) | analysis         | defense of paper).    |
+---+------------------------------+-----------+--------------------+---------------------+------------------+-----------------------+
| F | Practical Probing Overhead   | 8.8 / 10  | "Does a 5-beam     | ZERO GPU compute    | Mathematical link| PRIORITY 2: INCLUDE   |
|   | & Coherence-Time Latency     | (Vehicular| sweep fit within   | (Analytical link-   | budget & timing  | (Critical for IEEE TVT|
|   |                              | Systems)  | coherence time?"   | budget equations)   | model derivation | reviewer credibility).|
+---+------------------------------+-----------+--------------------+---------------------+------------------+-----------------------+
| B | Geometry-Assisted Candidate  | 8.5 / 10  | "Why restrict to   | ZERO GPU compute    | Spatial coverage | PRIORITY 3: INCLUDE   |
|   | Generation (Adaptive Width)  | (RF Prior)| K=20 when geometry | (Azimuth beamwidth  | analysis on saved| (Directly solves 84.7%|
|   |                              |           | exclusion is 85%?" | integration)        | GPS trajectories | candidate exclusion). |
+---+------------------------------+-----------+--------------------+---------------------+------------------+-----------------------+
| E | Candidate-Set Width vs.      | 8.0 / 10  | "What happens at   | ZERO GPU compute    | Cumulative power | PRIORITY 4: INCLUDE   |
|   | Communication Power Loss     | (Trade-off| K=1, 3, 10, 20, 64?| (Parse validation   | curve extraction | (Generates high-value |
|   | Pareto Frontier (K=1 to 64)  | Frontier) | Is 5 optimal?"     | prediction ranks)   | from npz files)  | IEEE TVT figure).     |
+---+------------------------------+-----------+--------------------+---------------------+------------------+-----------------------+
| C | Conditional Visual Utility   | 7.5 / 10  | "Is RGB useless,   | ZERO GPU compute    | Stratified subset| PRIORITY 5: INCLUDE   |
|   | Under Geometric Uncertainty  | (Context) | or only when GPS is| (Cross-reference GPS| analysis on saved| (Enriches Discussion  |
|   |                              |           | fully accurate?"   | heading jitter)     | validation logs) | without retraining).  |
+---+------------------------------+-----------+--------------------+---------------------+------------------+-----------------------+
| D | Cross-Scenario Invariant     | 3.5 / 10  | "Still only four   | HIGH GPU compute    | Domain adaptation| REJECT ENTIRELY       |
|   | Feature Learning / Adapt.    | (Over-fit)| scenarios; risk of | (Requires multi-day | training runs on | (Avoids unpublishable |
|   |                              |           | further tuning."   | fresh GPU training) | existing data)   | circular tuning).     |
+---+------------------------------+-----------+--------------------+---------------------+------------------+-----------------------+
```

---

## 3. Detailed Hypothesis Breakdown

### Hypothesis A: Power-Aware Alternative-Beam Ranking
- **Research Justification:** The central finding of the manuscript is that adding camera inputs to a shifted-window model reduces alternative-beam power success by 25.28 percentage points. Reviewers will ask: *"Is this an inherent defect of the visual modality, or is it an artifact of training with Cross-Entropy on discrete argmax labels?"*
- **Analysis:** By examining the predicted logit distributions and comparing them against measured power profiles, we demonstrate that Cross-Entropy penalizes beams $b^* \pm 1$ (which lie inside the 3 dB main-lobe) with identical gradient magnitude as beams 180 degrees away. A power-aware loss (such as Huber profile loss or soft ranking loss) preserves the probability of physical alternatives.
- **Action:** Can be fully substantiated using the existing comparisons between M3 (Cross-Entropy) and P3 (Power-Aware) from Stages 8–36, without any new training.

### Hypothesis F: Practical Probing Overhead and Vehicular Coherence Time
- **Research Justification:** A standard 60 GHz mmWave frame consists of preamble, beam training pilots, and data payload. In high-mobility V2V communications ($v_{\text{rel}} \ge 30\text{ m/s}$), the channel coherence time is bounded by:
  $$T_c \approx \frac{c}{2 f_c v_{\text{rel}}} = \frac{3 \times 10^8}{2 \times (60 \times 10^9) \times 30} \approx 4.16\text{ ms}$$
- **Analysis:** Sweeping an entire 256-beam codebook requires $256 \times T_{\text{pilot}}$. If $T_{\text{pilot}} = 100\ \mu\text{s}$, a full sweep requires $25.6\text{ ms}$, which is **more than $6\times$ larger than the channel coherence time**! Sweeping a 5-beam shortlist requires only $0.5\text{ ms}$, consuming only $12\%$ of the coherence budget and leaving $88\%$ of the frame for data payload transmission.
- **Action:** Formulate an analytical link-budget and frame-timing subsection in Section 1 of the manuscript. This grounds the 5-beam shortlist choice in physical RF reality and directly appeals to vehicular communications reviewers.

### Hypothesis B: Geometry-Assisted Candidate Generation
- **Research Justification:** The manuscript reports that for the 20-candidate geometry-constrained model, candidate exclusion accounts for 84.7% of severe misses (>10 dB loss). Reviewers will ask why the candidate generator failed.
- **Analysis:** At 60 GHz, each array panel covers a $90^\circ$ quadrant, with beamwidth $\approx 4.5^\circ$. When vehicles undergo aggressive turns or lane changes, the true physical LOS direction drifts by $\Delta\theta > 15^\circ$ between historical GPS updates. If the candidate window is centered strictly on the extrapolated kinematic vector with fixed width $W=20$, dynamic vehicle yaw swings the true beam outside the candidate subset.
- **Action:** Present an analytical breakdown showing how expanding candidate sets dynamically based on vehicle yaw rate ($\omega_z$) mitigates candidate exclusion.

### Hypothesis E: Candidate-Set Width vs. Power Loss Pareto Frontier
- **Research Justification:** The paper currently compares shortlists at $K=5$, $K=20$, and $K=40$. Vehicular system designers need to understand the full trade-off curve across all candidate budgets $K \in [1, 64]$.
- **Analysis:** Using the saved validation prediction arrays (`val_predictions.npz`), we can directly extract the cumulative Top-$K$ power success rate for every integer $K$ from 1 to 64.
- **Action:** Generate a Pareto trade-off curve (Top-$K$ 3 dB Success vs. Probing Overhead $K/256$). This provides actionable utility for system engineers without retraining any models.

### Hypothesis C: Conditional Visual Utility Under Geometric Uncertainty
- **Research Justification:** Does camera input provide zero utility everywhere, or is it specifically beneficial when GPS telemetry is degraded?
- **Analysis:** Partition the validation windows into low GPS uncertainty (straight highways, low acceleration) vs. high GPS uncertainty (sharp turns, high yaw rate, multi-vehicle occlusions). Evaluate the visual-minus-GPS difference conditionally within each subset.
- **Action:** Include this conditional breakdown in the Discussion section to explain the subtle mechanisms governing visual fusion.

### Hypothesis D: Cross-Scenario Invariant Feature Learning (REJECTED)
- **Research Justification:** Attempting to force domain invariance across the four scenarios via adversarial training or style transfer.
- **Rejection Rationale:** High computational cost (days of GPU time), extreme risk of overfitting to the four existing scenarios, and guaranteed reviewer skepticism regarding generalizability beyond the testbed.

---

## 4. Final Controlled Roadmap for Submission

```
+----------------------------------------------------------------------------------------------------+
|                                    CONTROLLED PRE-SUBMISSION ROADMAP                               |
+-------+-------------------------------+-------------------+----------------------------------------+
| Step  | Action Item                   | Compute / Source  | Target Deliverable                     |
+-------+-------------------------------+-------------------+----------------------------------------+
| 1     | Freeze All Existing Checkpoints| Local Git master  | Permanent preservation of Stages 1–299 |
+-------+-------------------------------+-------------------+----------------------------------------+
| 2     | Extract Candidate-Set Pareto  | Python script on  | Generate Figure & Table for Manuscript |
|       | Frontier (K=1 to 64)          | saved .npz arrays | (Pareto Trade-off Curve)               |
+-------+-------------------------------+-------------------+----------------------------------------+
| 3     | Compute Physical Link-Budget  | Analytical RF     | Add Frame Timing & Coherence Analysis  |
|       | & Latency Timing Equations    | Math equations    | to Introduction / System Model         |
+-------+-------------------------------+-------------------+----------------------------------------+
| 4     | Add Scenario 39 Jackknife     | Python script on  | Add transparent sensitivity breakdown  |
|       | Sensitivity Analysis          | saved contrasts   | to Discussion Section                  |
+-------+-------------------------------+-------------------+----------------------------------------+
| 5     | Keep Reserved Test Set LOCKED | File permission   | Strictly unopened until author signoff |
+-------+-------------------------------+-------------------+----------------------------------------+
| 6     | Authoritative Manuscript Edit | Human author via  | Final camera-ready submission for TVT  |
|       | by Research Team              | author_template   | complying with IEEE AI ethics          |
+-------+-------------------------------+-------------------+----------------------------------------+
```

---

## 5. Strict Enforcement of the Test Partition Lock

The reserved test partition consists of **17,017 frames** across Scenarios 36–39. 

**MANDATE:**  
Under no circumstances should any automated script, benchmark runner, or agent open or evaluate the test partitions during this audit. The test partition is the sole untouched asset protecting the project against post-hoc selection bias. It must remain strictly locked until the human authors complete all paper text revisions, freeze all analytical figures, and formally authorize the final evaluation run.
