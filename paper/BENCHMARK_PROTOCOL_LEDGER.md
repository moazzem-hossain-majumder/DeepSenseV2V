# Benchmark and claim ledger — verified 2026-11-09

This is a source/protocol audit, not a claim of reproducing published scores.

- Official challenge baseline: https://github.com/DeepSense6G/Multi-Modal-V2V-Beam-Prediction-Challenge-2023-Baseline . The README identifies average power loss as the updated challenge score. The source uses five historical positions, nominal 0.2 s input spacing, 0.5 s extrapolation, and four 64-beam arrays. It expects an external challenge split CSV that is not present here. Its signed mean log-power ratio is the negative of our positive mean power regret. The local blocked split is not the official challenge split.
- Mollah, Wang and Fang, arXiv:2509.11112v1 (GLOBECOM 2025 accepted version): https://arxiv.org/html/2509.11112v1 . Section IV describes random within-scenario 60/20/20 partitions and a 64-beam codebook. The visual encoder is MaxViT. Equation 8 averages noise-adjusted linear ratios before converting to dB, unlike our mean of per-window dB losses. These differences prevent direct score ranking against our 256-beam trajectory-blocked forecasting protocol. An exact implementation reproduction remains outstanding.
- Gao et al., CLBP, IEEE TMC 2026, DOI 10.1109/TMC.2026.3674213: https://research.polyu.edu.hk/en/publications/clbp-a-cross-modal-loss-tolerant-beam-prediction-framework-for-v2/ . The institutional abstract describes RGB + LiDAR and missing-modality experiments. LiDAR lies outside this project's GPS + RGB scope. This is related work, not a modality-matched score target; full protocol/code reproduction has not been established.

## Our fixed local protocol

Five inputs at t-8,t-6,t-4,t-2,t; target t+5. Report actual time horizons. Four scenarios, four arrays/256 classes, front RGB5 and GPS only. Power is supervision/evaluation ground truth, never a predictor input. Original sequences contain windows; larger trajectory blocks separate train/validation/calibration/test. Fit preprocessing on training only.

## Predeclared next experiment

Same architecture families and training budget across three seeds. Shared regularization: dropout 0.25, AdamW weight decay 0.001, fixed pretrained BatchNorm statistics, mild clip-consistent brightness variation. Compare GPS, RGB, fusion CE, existing profile loss, and power-aware profile training. Eight epochs each, cosine LR 1e-4 to 1e-5; select by macro-scenario validation success within 3 dB with five beams. Report exact-beam accuracy, power regret, and shortlist metrics as secondary outcomes without hiding negative results.

The power-aware objective uses soft labels proportional to exp(-power_regret/3 dB), a relative-profile Huber auxiliary loss and expected power-regret penalty. This is an experiment, not an established novel contribution. All hard-label baselines share the new regularization; the existing profile objective isolates the objective change within the same network.

Cross-scenario checks exclude one scenario from BOTH training and checkpoint-selection validation. Evaluate its validation blocks only after checkpoint selection. These are development transfer checks; all test blocks remain reserved.

## Automatic freeze gate (fixed before new training)

Proceed to calibration and locked test only if: the power-aware method exceeds every matched baseline by at least 1 percentage point in three-seed mean macro validation success within 3 dB at k=5; the paired scenario-stratified block-bootstrap 95% lower bound against the strongest baseline is positive; and it exceeds fusion CE on at least three of four unseen-scenario validation checks. If these conditions fail, produce a negative-results/development report and leave calibration/test unused. The gate is a development decision rule, not a statistical guarantee or acceptance criterion.

Calibration fixes a 90% near-optimal-beam target for both fixed-k and adaptive probability-prefix candidate sets, with no test tuning. Temporal dependence and distribution shift prevent claiming exchangeable conformal guarantees. Record empirical per-scenario coverage, candidate count, power loss and trajectory-bootstrap uncertainty on the locked test if the gate passes. Report full-sweep (256-beam) reference cost; do not claim end-to-end online savings without a feedback/latency experiment.

## Remaining publication decisions

Novelty, venue, authorship, exact strongest-baseline reproduction and editorial review require substantive review of the results. Completing this compute queue does not automatically make the paper publishable. Never invent missing experiments, comparisons, statistical significance or citations.

## Three-seed transfer benchmark audit, 11 November

Stage 267 compares front/rear RGB plus GPS, matched GPS and analytic geometry on the same excluded-scenario validation windows. Across four equally weighted scenarios and three seeds, their top-five success rates within 3 dB were 78.030%, 78.018% and 78.037%, respectively. Their mean power losses were 4.964, 4.973 and 5.017 dB. Geometry led on the primary metric among these three controls and models. The paired RGB-minus-GPS loss interval included zero. Report both controls and both endpoints when describing transfer performance. All figures derive from the 24 arm/fold/seed records in results/widecheck_v1/stage267/evidence.json.

The official challenge uses average power loss. Reporting that secondary measure here supports a discussion of metric choice. A leaderboard comparison would also require the official evaluation windows, the same number of selected beams and identical scoring and aggregation. The local blocked transfer results support a development comparison only. [Official challenge baseline README](https://github.com/DeepSense6G/Multi-Modal-V2V-Beam-Prediction-Challenge-2023-Baseline)

## Candidate contribution and closest same-task comparator

Fabiani et al. already combine GPS with front and rear cameras to forecast V2V beams 500 ms ahead on DeepSense. Their comparison with a linear position baseline motivates a closer question: how much of the gain survives comparison against a matched learned GPS model when an entire driving scenario is excluded? An additional diagnostic attributes severe failures to candidate exclusion versus ranking inside the shortlist. These are candidate empirical contributions. Priority remains open pending a focused citation review. [Fabiani et al.](https://repository.kaust.edu.sa/server/api/core/bitstreams/712d9981-ea7a-44e5-870b-c36e6129acb2/content) [Earlier beam prediction distribution-shift study](https://alloulah.github.io/docs/Arnold22_BeamBench.pdf)

Before submission, prioritize one comparison: evaluate the closest reproducible published fusion system and its learned GPS ablation on identical trajectory-blocked validation windows, with the candidate budget, forecast horizon, seed set and metrics frozen in advance. Distinguish a faithful reproduction from an adaptation. Assess both exact-best-beam top-five accuracy and top-five success within 3 dB. The explored scenarios supply development evidence; independent confirmation requires separate data and a frozen protocol.

The focused literature check also identified Nazar et al., who study modality combinations for infrastructure-to-vehicle prediction, and the ITU 2022 challenge team's released multimodal transformer on Scenarios 31–34. These establish related prior work on modality ablation and environment change. The narrower question here concerns the incremental camera contribution in V2V forecasting on identical windows, together with attribution of failures to candidate exclusion. Priority remains unresolved. [Nazar et al.](https://repository.kaust.edu.sa/bitstreams/d571512f-2bcb-4010-a154-2425fac05605/download) [2022 challenge implementation](https://github.com/ITU-AI-ML-in-5G-Challenge/DeepSense6G_TII)
