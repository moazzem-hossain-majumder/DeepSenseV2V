# Stages 44–61: conservative corrections to geometry

Approved for execution 10 November 2026. This development cycle investigates why learned corrections damaged a strong analytic baseline. It does not establish publication readiness or authorize test-set tuning.

- 44: Compare the existing latitude/longitude construction with local east/north coordinates, velocity fitted from five observed timestamps, and a fixed 0.5-second forecast. Fit beam offsets on training only; choose conventions on selection validation.
- 45: Break down the previous models' helpful and harmful changes by scenario, slow motion and beam-panel classification. Physical mounting and semantic camera visibility remain unverified.
- 46–48: Compact GPS correction, seeds 42, 7 and 13.
- 49–51: GPS plus front-RGB correction, the same three seeds.
- 52–59: Both methods with each of Scenarios 36–39 held out in turn. Exclude that scenario from offset fitting, coordinate selection, normalization, training and fallback selection; evaluate its validation partition afterward.
- 60: Compare raw and fallback-selected performance, paired macro-scenario trajectory-bootstrap intervals, and unseen-scenario results.
- 61: Generate the evidence report, figure and next-action decision.

Each training run receives eight epochs. Corrections start at zero and are bounded to ±0.25 logits, with an L2 penalty. The compact model uses historical GPS and geometry context; the multimodal model adds front-camera imagery. A learned correction is enabled only when its selected validation macro success within 3 dB at five beams exceeds the matching analytic baseline by at least 0.25 percentage points. Otherwise inference uses geometry. Validation-selected fallback is optimistic on selection validation, so both raw results and unseen-scenario results must be reported.

The follow-up review screen requires a mean gain above one percentage point, a positive lower bound on the paired macro interval, and gains on at least three of four unseen scenarios. Passing this screen requests scientific review; it does not automatically open calibration/test or establish novelty. Failure produces a diagnostic recommendation rather than indefinite parameter searching.

Outputs: `results/conservative_v1/`. Final report: `stage61/REPORT.md`. Progress uses the existing desktop **DeepSense Training Progress** window and `results/active_training.json`. The queue automatically advances after verified successful completion. Completed epochs resume after interruption. The supervisor checks every 15 seconds and retries transient failures; repeated failures stop with an attention status instead of silently looping forever. All training data and pretrained weights are local, so normal training needs no internet. Sign-in startup restores the supervisor after a reboot; it cannot work while the PC is off or asleep.

Initial runtime estimate: roughly 1–3 hours, to be revised using measured epoch times. This is a planning estimate, not a completion guarantee. `pause.flag` in the project root pauses at an epoch/stage boundary. Source snapshots and hashes are frozen at Stage 44; the earlier scientific implementation remains unchanged.
