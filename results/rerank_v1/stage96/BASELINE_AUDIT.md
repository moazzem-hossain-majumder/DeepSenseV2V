# Baseline protocol audit — 10 November 2026

Inspected primary sources: [official baseline](https://github.com/DeepSense6G/Multi-Modal-V2V-Beam-Prediction-Challenge-2023-Baseline) and [script](https://raw.githubusercontent.com/DeepSense6G/Multi-Modal-V2V-Beam-Prediction-Challenge-2023-Baseline/main/competition_benchmark.py).

The script uses five GPS inputs and nominal 0.2/0.5-second input/forecast timing. Its orientation routine indexes the previous sample (`i-1`), its angular ranking uses unwrapped absolute distances, and its final index correction adds/subtracts 255. It estimates an offset from beam-index differences. Its signed average log-power ratio differs in sign from our positive power regret. These details prevent presenting our adapted geometry as an exact reproduction.

Our baseline uses within-window orientation, circular angle distances, modulo 256 indexing and a training-power-fitted offset. It uses local trajectory-separated splits. The official challenge split file is not available in this workspace. Exact challenge reproduction and any state-of-the-art comparison remain outstanding; a local adapted-baseline win is insufficient. Do not silently replace the source implementation and call the result an official benchmark score.

The candidate coverage experiment uses the existing audited local geometry predictions. No official score or reproduction result is fabricated by this audit. Future reproduction should preserve a source snapshot and distinguish literal implementation from independently corrected adaptations under explicit split rules.
