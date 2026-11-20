# RGB learning pathway audit

Training-only, fixed-checkpoint diagnostic probes. No optimizer steps, validation selection, or calibration/test evaluation.
Timestamp conversion remains unresolved; the distributed CSV mapping is unchanged. See Stage 72/PROVENANCE.md.

- Checkpoint 49: saturation FP32 58.07%, BF16 58.06%; RGB parameter gradient norm 0.02391/0.02401; RGB feature variability 609.3; mean absolute original-versus-gray correction change 4.074e-06.
- Checkpoint 50: saturation FP32 61.68%, BF16 61.69%; RGB parameter gradient norm 0.02158/0.02113; RGB feature variability 0.4405; mean absolute original-versus-gray correction change 0.0007674.
- Checkpoint 51: saturation FP32 64.32%, BF16 64.33%; RGB parameter gradient norm 0.008395/0.008319; RGB feature variability 684.5; mean absolute original-versus-gray correction change 2.025e-06.
- Checkpoint 53: saturation FP32 61.25%, BF16 61.26%; RGB parameter gradient norm 0.03315/0.04118; RGB feature variability 9816; mean absolute original-versus-gray correction change 0.02807.
- Checkpoint 55: saturation FP32 64.79%, BF16 64.76%; RGB parameter gradient norm 0.3983/0.3994; RGB feature variability 9398; mean absolute original-versus-gray correction change 0.09819.
- Checkpoint 57: saturation FP32 87.61%, BF16 87.63%; RGB parameter gradient norm 0.02078/0.02081; RGB feature variability 59.49; mean absolute original-versus-gray correction change 0.001887.
- Checkpoint 59: saturation FP32 65.78%, BF16 65.77%; RGB parameter gradient norm 0.009379/0.009369; RGB feature variability 197.6; mean absolute original-versus-gray correction change 6.794e-06.

Run a controlled training-only pilot comparing the current tanh correction against a smooth non-saturating residual penalty, holding inputs and splits fixed. Verify finite nonzero RGB gradients before any full multi-seed queue.

These probes identify mechanisms to investigate; they do not prove which change will improve generalization. Gradient norms are local diagnostics and depend on parameter count and loss scaling. Zero output-head initialization intentionally blocks upstream gradients on the first step; persistent zero gradients in trained checkpoints require separate explanation.

Before any shifted-image experiment, obtain timing-conversion provenance. Keep final evaluation reserved and predeclare the next comparison.