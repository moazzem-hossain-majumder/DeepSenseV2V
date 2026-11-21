# Ablation Results Provenance and Audit Notice

This notice accompanies `ablation_results.csv` in accordance with the Phase 3 Audit Findings documented in [`AUDIT/PAR_NET_KMACC_VERIFICATION.md`](AUDIT/PAR_NET_KMACC_VERIFICATION.md).

## Summary of Row Types in `ablation_results.csv`

1. **Rows M0–M6 (Verified Baselines & Benchmark Models):**
   - **M0 (Analytic Geometry Prior):** Non-parametric LOS kinematic trajectory projection.
   - **M1 (Learned GPS BiGRU):** Recurrent temporal GPS model.
   - **M2 (Learned RGB ResNet-18):** Visual single-frame / temporal baseline.
   - **M3 (Multimodal PreLN Fusion):** Early multimodal fusion model.
   - **M4 (Profile Regression P3):** 256-MSE power profile regression.
   - **M5 (Power-Aware Soft-Label):** Huber / soft cross-entropy variant.
   - **M6 (Shifted-Window Attention):** Reproduction of published vision+GPS temporal attention architecture (Fabiani et al., 2024 / Charan et al., 2022).
   - *Status:* Empirically evaluated on the 16,643 validation windows across Scenarios 36–39.

2. **Row M7 (PAR-Net Heuristic):**
   - *Status:* Exploratory heuristic perturbation defined in `src/unified_benchmark.py` (line 311) adding a synthetic cosine boost over top-20 candidate logits. Not a trained neural network checkpoint.

3. **Rows A1–A4 (PAR-Net Ablation Rows):**
   - *Status:* Hardcoded conceptual illustrative values (defined at `src/unified_benchmark.py:354-358`). No trained model weights or checkpoints exist on disk.

For peer-reviewed publication-grade empirical results based strictly on frozen trained model checkpoints across three random seeds and four cross-scenario folds, consult:
- **[`results/widecheck_v1/stage267/`](results/widecheck_v1/stage267/)**
- **[`MANUSCRIPT/FINAL_IEEE_MANUSCRIPT.md`](MANUSCRIPT/FINAL_IEEE_MANUSCRIPT.md)**
- **[`AUDIT/PAR_NET_KMACC_VERIFICATION.md`](AUDIT/PAR_NET_KMACC_VERIFICATION.md)**
