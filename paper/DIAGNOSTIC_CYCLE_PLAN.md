# Stages 62–71: identify the failure before more training

This cycle follows the inconclusive 79.00% versus 78.56% validation comparison and the unseen-Scenario-38 regression. It is a fixed-checkpoint diagnostic cycle, not another hyperparameter search.

- Stage 62 checks historical input order, sequence membership, recorded forecast horizons, front/rear RGB and GPS filename timestamp differences, and duplicate image references across the training/validation input rows of all four scenarios. Timestamp offsets are observations, not proof of a sensor synchronization error. No frames are shifted.
- Stage 63 reports help/harm, power regret and success by scenario, trajectory block and low-speed heading group for the seven multimodal checkpoints.
- Stages 64–70 evaluate each of the three pooled seeds and four held-scenario checkpoints with original, uniform-gray and deterministically shuffled within-scenario validation clips. GPS, labels and geometry remain unchanged. Original predictions must reproduce saved logits within 0.0001 where reference raw predictions exist. Fallback is disabled to expose raw correction behavior, including for Scenario 39; this is not a new deployment selection.
- Stage 71 consolidates results and writes the next experiment specification. A substantial filename timestamp offset requires resolving clock conventions before any shifted-frame training. Gray/shuffle effects are sensitivity measurements under distribution shift, not equivalent to a trained GPS-only ablation.

No calibration/test evaluation or new model training is included. Repeated validation inspection remains exploratory. External published-method reproduction, physical mounting/camera coverage and independent final evaluation remain outstanding.

The existing progress window shows each stage and inference batches. Supervisor checks and transient retries occur every 15 seconds, with bounded failure retries and completion notifications. Completed intervention modes resume from saved predictions. Sign-in startup restores the service. `pause.flag` pauses at stage boundaries. Estimated runtime is 10–30 minutes, subject to measured inference speed and disk access.

Outputs: `results/diagnostic_v1/`; final decision: `stage71/REPORT.md` and `decision.json`. Earlier code and experiment sources remain frozen.
