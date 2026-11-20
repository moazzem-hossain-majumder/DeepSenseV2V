# Stages 72–80: timing provenance and RGB learning pathway

Stage 72 records primary-source timing documentation, training-sequence filename offsets and sampled image EXIF timestamps. The dataset paper describes synchronized CSV rows but the inspected sources do not explain the observed filename differences. No remapping is performed and physical synchronization is not declared verified.

Stages 73–79 probe all seven frozen multimodal checkpoints, using up to 32 evenly spaced training windows per training scenario. Held-out scenarios and all validation/calibration/test rows are excluded. Each probe uses identical weights and inputs under FP32 and BF16, recording per-module and image-input gradient norms, feature variability, raw-head saturation, numerically zero tanh derivatives and gray-image correction sensitivity. Dropout is disabled and the native GRU implementation permits eval-mode backward. No optimizer steps or checkpoint updates occur. These are local mechanism checks rather than generalization measurements.

Stage 80 produces a concrete next experiment recommendation based on saturation evidence. The next controlled pilot must validate its learning pathway before expensive multi-seed training. Timestamp uncertainty separately blocks shifted-image experiments and confident physical timing claims.

The same progress window tracks the finite queue, supervised every 15 seconds with bounded retry and sign-in recovery. Each completed stage is preserved; an interrupted probe restarts that small stage. Outputs: `results/pathway_v1/`; final report: `stage80/REPORT.md`. Initial runtime estimate: 5–15 minutes, subject to GPU and disk speed. Root `pause.flag` pauses at stage boundaries. Calibration/test remain reserved.
