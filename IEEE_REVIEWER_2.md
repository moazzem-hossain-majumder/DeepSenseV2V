# IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY: PEER REVIEW REPORT

**Reviewer Profile:** Reviewer 2 (Senior Researcher in Machine Learning, Multimodal Deep Learning, and Computer Vision)  
**Manuscript Title:** *Does Camera Input Improve the Power of a Five-Beam Shortlist? A Paired Study of Vehicle-to-Vehicle Beam Forecasting*  
**Tracking Number:** TVT-2026-AUDIT-R2  
**Recommendation:** **MAJOR REVISION**  
**Overall Score:** **6.0 / 10**

---

## 1. Summary of the Contribution

The manuscript presents an empirical investigation into multimodal fusion for mmWave beam forecasting. Using an adaptation of the shifted-window transformer model of Fabiani et al. (Asilomar 2024), the authors pair GPS telemetry with front and rear 224x224 RGB cameras across four vehicular scenarios from DeepSense 6G. 

The paper reports a counter-intuitive finding: on excluded driving scenarios, adding RGB features increases Top-5 exact-beam classification accuracy (+5.42 pp) but severely decreases the Top-5 3 dB power success rate ($-19.85$ pp). Through an alternative-beam decomposition, the authors show that this collapse is driven by a massive loss of alternative-beam recovery ($-25.28$ pp). For a separate geometry-constrained candidate reranker, the authors show that 84.7% of severe power outages are caused by candidate exclusion.

---

## 2. Strengths

1. **Rigorous Counter-Narrative to Multimodal Hype:** In machine learning literature, papers routinely claim that adding more modalities (especially vision) monotonically improves performance. This paper provides honest, reproducible evidence that adding visual features can actually degrade performance in safety-critical physical metrics. This level of intellectual honesty is rare and valuable.
2. **Identification of Cross-Entropy Over-Sharpening:** The authors correctly observe that Cross-Entropy loss on discrete beam labels creates an artificial Dirac-like distribution that penalizes physically viable neighboring beams, explaining the divergence between exact classification and physical power coverage.
3. **Pristine Protocol Isolation:** The authors went to great lengths to ensure zero temporal leakage across trajectory blocks and used strictly matched transformer architectures (zeroed visual tokens) for their GPS comparator.

---

## 3. Major Concerns (Must be Addressed for Acceptance)

1. **Failure to Experimentally Prove the Objective Hypothesis:**  
   In Section 6 (Discussion), the authors state:  
   *"The alternative-beam decomposition identifies an observable change in shortlist composition. It does not identify why the network made that change. Cross-entropy rewards the labeled maximum... Establishing that the objective caused the observed pattern would require a controlled intervention using the same architecture and optimization budget."*  
   As an ML reviewer, this is a major gap. The authors hypothesize that Cross-Entropy over-sharpening caused the alternative-beam collapse, but they stop short of proving it! Why did the authors not evaluate a power-aware loss function (e.g., soft Cross-Entropy with temperature proportional to measured dB loss, or a pairwise Huber ranking loss) on the exact same shifted-window architecture? Proving that a power-aware loss restores the alternative beams would elevate this paper from an inconclusive observational report into a landmark paper on loss function design for physical systems.
2. **Severe Visual Modality Degradation (Why Did Vision Fail?):**  
   Why do the cameras hurt? Is it because:
   - The visual backbone (EfficientNet-B3) was frozen and not fine-tuned on vehicular scenes?
   - The spatial feature pooling to a $2 \times 2$ grid discarded essential vehicle bounding-box coordinates?
   - Camera vibration and roadway pitch during braking introduced high-frequency noise?
   - Day-to-night lighting shifts between scenarios caused severe out-of-distribution visual embeddings?  
   The authors observe the degradation but offer very little diagnostic explanation. The paper needs at least a qualitative feature attribution or visual attention map analysis showing what the transformer was focusing on when it failed.
3. **The Geometry-Constrained Reranker is Under-Specified:**  
   In Section 4, the geometry-constrained reranker is described in just three sentences. The authors state that time-averaged features from a fixed ResNet-18 were joined with GPS history. Why was ResNet-18 used here, while EfficientNet-B3 was used in the full-codebook model? This architectural asymmetry makes it difficult to compare the two models cleanly.

---

## 4. Minor Concerns

1. **Tie-Breaking Sensitivity:** In Section 5, the authors mention that reversing the tie-breaking order of equal-valued beam scores resulted in a $-19.75$ pp difference vs. the primary $-19.85$ pp. Why were there equal-valued beam scores in a continuous softmax output? Does the model predict identical zeros or saturated logits for inactive beams? Please clarify.
2. **Temporal Windowing Notation:** In Section 3, the row offsets are listed as $[-8, -6, -4, -2, 0]$ predicting $+5$. Please state explicitly the effective time duration in seconds (i.e. $0.8\text{ s}$ of history predicting $0.5\text{ s}$ ahead).

---

## 5. Missing Experiments

1. **Loss-Function Intervention:** Train or evaluate a matched version of the full-codebook model using a soft label distribution $p_i \propto \exp(-L(b_i)/\tau)$ where $L(b_i)$ is the decibel power loss and $\tau = 3.0\text{ dB}$. If the alternative-beam success recovers, the hypothesis is proven.
2. **Visual Feature Distribution Shift Analysis:** Plot a t-SNE or PCA of the visual embeddings across the four scenarios to show whether Scenario 38/39 (intersections and night driving) suffer from severe feature distribution shift compared to Scenarios 36/37.

---

## 6. Novelty Assessment

The novelty does not reside in the model architecture (which is an adaptation of Fabiani et al. and standard shifted-window transformers). Rather, the novelty lies in **characterizing the fundamental failure mode of categorical cross-entropy when applied to continuous spatial beamforming codebooks**. This is a solid, publishable contribution to applied machine learning if the loss-mechanism discussion is deepened.

---

## 7. Reviewer Recommendation and Concluding Remarks

**Recommendation:** **MAJOR REVISION.**  
The paper makes an important empirical discovery that challenges prevailing assumptions about multimodal fusion in wireless systems. However, leaving the cause of the failure as an open question weakens the paper significantly. By deepening the analysis of cross-entropy over-sharpening and providing visual feature diagnostics, the authors can turn this into an impactful, widely cited publication.
