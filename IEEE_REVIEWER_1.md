# IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY: PEER REVIEW REPORT

**Reviewer Profile:** Reviewer 1 (Senior IEEE Researcher in Wireless Communications, Vehicular Systems, and mmWave Beam Management)  
**Manuscript Title:** *Does Camera Input Improve the Power of a Five-Beam Shortlist? A Paired Study of Vehicle-to-Vehicle Beam Forecasting*  
**Tracking Number:** TVT-2026-AUDIT-R1  
**Recommendation:** **MAJOR REVISION**  
**Overall Score:** **6.5 / 10**

---

## 1. Summary of the Contribution

This paper addresses the problem of proactive beam selection in 60 GHz vehicle-to-vehicle (V2V) communications using multimodal sensory telemetry (GPS and front/rear RGB cameras). The authors evaluate an architectural adaptation of a shifted-window transformer model across 16,643 validation windows drawn from DeepSense-V2V Scenarios 36–39. 

The primary contribution is the empirical observation of a severe divergence between discrete classification accuracy and physical communication utility: adding cameras to a full-codebook model increases exact Top-5 beam accuracy by +5.42 percentage points, but decreases the probability of including a beam within 3 dB of the optimal power by 19.85 percentage points. The authors attribute this drop to the loss of useful "alternative beams," where GPS alone frequently selects near-optimal neighboring beams within the physical antenna main-lobe. Additionally, for a geometry-constrained candidate reranker, the authors show that candidate exclusion accounts for 84.7% of severe (>10 dB) power outages.

---

## 2. Strengths

1. **Practical Communication Grounding:** The paper wisely shifts the evaluation focus away from academic Top-1 discrete accuracy toward physical communication metrics: 3 dB power coverage and average power loss (APL). In vehicular communications, link margin and SNR dictate throughput and packet error rates; discrete classification metrics can be deceptive.
2. **Identification of the Alternative-Beam Phenomenon:** The empirical demonstration that a model can gain +5.42 pp in exact classification while losing 19.85 pp in 3 dB power success is a vital insight for wireless engineers designing beam management protocols.
3. **Strictly Matched GPS Controls:** Rather than comparing a multimodal transformer against a naive constant-velocity extrapolation, the authors evaluate a matched GPS control that uses the identical transformer architecture with visual tokens zeroed out. This isolates the true marginal contribution of the camera modality.
4. **Candidate Exclusion vs. Ranking Diagnosis:** Proving that 84.7% of severe misses in a 20-beam candidate pool are caused by candidate exclusion rather than ranking error provides actionable guidance for candidate generator design.

---

## 3. Major Concerns (Must be Addressed for Acceptance)

1. **Lack of Physical Link-Budget and Latency Analysis:**  
   The paper repeatedly refers to a "five-beam shortlist" without providing the physical communication justification for $K=5$. What is the channel coherence time at the relative vehicular velocities observed in Scenarios 36–39 ($v_{\text{rel}} \approx 15\text{--}25\text{ m/s}$)? How many microseconds does probing each beam take under the 60 GHz transceiver architecture? The authors must demonstrate analytically that sweeping 5 beams fits within the coherence time whereas sweeping 256 beams does not. Without this, the choice of $K=5$ appears completely arbitrary.
2. **Superficial Treatment of Transceiver Array Geometry:**  
   The transceiver in DeepSense-V2V comprises four 64-beam uniform planar arrays (UPA) facing $0^\circ, 90^\circ, 180^\circ,$ and $270^\circ$. A 3 dB power success means the receiver aligned within the half-power beamwidth ($\approx 4.5^\circ$) of the transmitting vehicle. The manuscript treats the 256 beams as an abstract discrete set $b \in \{1, \dots, 256\}$. How does the array panel switching boundary affect alternative-beam success? When a vehicle crosses panel boundaries (e.g., from Panel 1 to Panel 2), are alternative beams available on both panels, or is there an RF blind spot? The physical antenna geometry must be discussed.
3. **The Analytic Geometry Baseline Dominance:**  
   The authors report that simple analytic geometry achieves 78.04% Top-5 3 dB power success on excluded scenarios, outperforming the full-codebook multimodal deep learning model (48.71%) by nearly 30 percentage points! This is a massive indictment of the deep learning approach. The authors must explicitly discuss why an unconstrained neural network fails so dramatically against basic trigonometry. If analytic geometry is so superior, why would a vehicular systems engineer ever deploy a 17-million-parameter deep neural network with two cameras?

---

## 4. Minor Concerns

1. **Sign Convention of APL:** In Section 3, ensure the sign convention of power loss $L(A) = \max_b P(b) - \max_{b \in A} P(b)$ is clearly stated as a positive loss in decibels (where lower is better), contrasting it with the negative log-ratio reported on the 2023 challenge leaderboard.
2. **Doppler and Vehicle Suspension Pitch:** When vehicles brake or accelerate, suspension pitch can misalign the camera axis relative to the roof-mounted mmWave horn array by $2^\circ\text{--}4^\circ$. Could this physical misalignment explain why cameras introduce noise into the beam forecast? This should be noted in the Discussion.

---

## 5. Missing Experiments

1. **Candidate-Width Pareto Trade-off Curve:** Provide a curve showing the 3 dB power success rate and Average Power Loss as a function of shortlist width $K$ for $K \in \{1, 2, 3, 5, 10, 15, 20, 32, 64\}$. This will show engineers the exact trade-off between probing latency and link reliability.
2. **RF Coherence Timing Budget Table:** Add a brief table comparing the time required for exhaustive sweep (256 beams), candidate sweep (5 beams), and channel coherence time across urban (15 m/s) and highway (30 m/s) regimes.

---

## 6. Novelty Assessment

The general concept of multimodal beam forecasting using GPS and cameras is already established (Alrabeiah 2020, Charan 2022, Fabiani 2024). The novelty here lies specifically in **uncovering and characterizing the Alternative-Beam Paradox in multi-beam shortlists**. This is a genuinely valuable insight for vehicular communications, provided the physical wireless context is strengthened.

---

## 7. Reviewer Recommendation and Concluding Remarks

**Recommendation:** **MAJOR REVISION.**  
The paper contains a compelling physical message that is highly relevant to IEEE TVT readers, but it is currently written like an exploratory machine learning diagnostic rather than a definitive vehicular communications study. If the authors anchor the 5-beam shortlist in physical coherence-time equations, analyze the 4-panel array geometry, and confront the 30% deep learning deficit honestly, this will make a strong TVT regular paper.
