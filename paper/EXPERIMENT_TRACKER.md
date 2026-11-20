# Multi-scenario research tracker

Updated: 2026-11-10. Workspace: D:\DeepSenseV2V.

## Active five-hour autonomous research window

User authorized research decisions and execution every 30 minutes until 22:43:29 Bangladesh time on 10 November. [Autonomous tracker](AUTONOMOUS_5H_TRACKER.md) is authoritative for this window. The initial controlled objective/constraint queue is Stages 124–133 (`results/factorial_v1/`). Recurring agent reviews may create subsequent justified versions, preserving earlier snapshots. Calibration/test remain reserved. No new experiments after the cutoff; already-running work may finish safely.

## Current: Stages 117–123, focused mechanism audit

Completion review: all seven stages finished with zero failures at 16:02:49 Bangladesh time on 10 November. On sixteen selected training failures, final near-optimal top-five memorization was 93.75% for the existing loss, 81.25% for bounded cross-entropy, and 100% for both unbounded shared-head cross-entropy and the per-example positive control. The unbounded shared head achieved 75% exact best-candidate top-1; the control achieved 100%. These are not validation improvements. The pipeline can learn on this sample; a blanket broken-learning diagnosis is unsupported. Visual review of all sixteen front/rear examples is saved in `results/mechanism_v1/stage117/visual_review.md`: vehicle detail is substantially reduced at 96 pixels, especially at night; transmitter identities remain unverified. Synthetic official-source checks confirmed previous-sample dependence and non-permutation 255-wrap behavior; this is not a benchmark reproduction. No additional stages are running. Recommended next step: a small, predeclared comparison isolating objective and correction constraints within the same hard shortlist, with matched GPS/visual controls; evaluate resolution separately afterward rather than changing several factors at once.

User authorized the bounded audit after reranking failed to improve performance. [Plan](MECHANISM_AUDIT_PLAN.md): sixteen deliberately selected training failures, four memorization/control arms, visual inspection and synthetic checks of a snapshotted official baseline. Same progress monitor and automatic queue; estimated 3–10 minutes of compute plus visual review. No full-data training or held-out evaluation. Outputs: `results/mechanism_v1/`.

## Current: Stages 96–116, constrained reranking

Completion review: queue finished at 15:50:18 Bangladesh time on 10 November with zero failures. Seven stages executed; fourteen expansion stages (102–115) were skipped because neither visual pilot qualified. Top-10 GPS 78.5617%, visual 78.6137%; top-20 GPS and visual both 78.5617%. Best gain is only 0.0520 percentage points. Training losses decreased across all four pilots, but the selection metric was nearly unchanged. Source snapshot hashes remain intact; no final-test lock was created. The earlier 79.00% eight-epoch three-seed result remains inconclusive and is not directly budget-matched to these three-epoch pilots. No new stages are queued. Recommended next step is a bounded research/implementation audit: verify shortlist ranking sensitivity and loss-to-metric behavior with a tiny training-only memorization test, inspect actual transmitter visibility and timing provenance, and resolve published-baseline protocol comparability before another large training cycle. More automatic parameter search is not justified by the current evidence alone.

Launch verified: Stage 96 completed and Stage 97 started automatically. Three targeted tests passed, including hard shortlist containment, matched-model selection rules, and GPU training/checkpoint resumption for both K values and both GPS/visual variants. Native progress window and sign-in recovery are active. Training coverage admitted the top-10 pilot; complete coverage results are recorded in `results/rerank_v1/stage96/coverage.json`.

User authorized the next plan. [Execution plan](RERANK_CYCLE_PLAN.md): coverage feasibility, four matched pilots (GPS control/visual at K=10/20), and conditional paired three-seed/held-scenario expansion. Frozen normalized visual features and shortlist masking replace unconstrained 256-beam corrections. [Baseline audit](RERANK_BASELINE_AUDIT.md) records why the adapted geometry is not an exact official challenge reproduction. Same progress window; outputs `results/rerank_v1/`. Calibration/test remain reserved. Initial estimate 30–90 minutes through pilots, 2–5 hours if expansion qualifies.

## Current: Stages 81–95, controlled correction/camera experiments

Completion review: the queue finished at 13:50:35 Bangladesh time on 10 November, with zero failures. Eight stages executed; seven expansion stages (88–94) were marked complete as skipped because no pilot qualified. Three-epoch macro validation success: front BF16 tanh 78.43%, front FP32 tanh 78.56%, front linear 67.72%, rear linear 68.21%, both linear 73.19%; geometry 78.56%. No demonstrated improvement; the earlier eight-epoch three-seed 79.00% result remains the best recorded mean and is still inconclusive. Training probes show the linear variant responds to images but can produce much larger corrections, suggesting that removing saturation alone did not preserve geometric ranking quality. Both-camera performance exceeds the single-camera linear pilots, but one seed/three epochs does not establish a robust camera benefit. No calibration/test evaluation occurred. No stages beyond 95 are queued. Recommended next direction: quantify geometry shortlist coverage and failure cases before considering a constrained candidate-reranking pilot, while completing a matched published-baseline protocol audit. This is a proposed experiment, not an established fix.

Launch verified at 12:42 Bangladesh time: Stage 81 is decoding rear images and has checkpointed more than 5,000 Scenario 36 frames. Four targeted tests passed, including cache build/verified resume, camera ordering, conditional gating, and GPU training/checkpoint resume for all five pilot settings plus the learning-health check. Native progress window is open; camera supervisor and sign-in recovery are installed.

User authorized the next training cycle. [Execution plan](CAMERA_CORRECTION_PLAN.md): cache already-downloaded rear RGB, run five matched three-epoch pilots separating correction arithmetic, saturation and camera coverage, then automatically expand a qualifying pilot to three seeds and four held-scenario runs. If none qualifies, skip longer training and produce the report. Same native progress monitor; output `results/camera_v1/`. Calibration/test stay reserved. Preliminary estimate: 1–2 hours through pilots, 3–6 hours total if expansion runs. GPU integration checks passed training/checkpoint-resume for all five pilot configurations.

## Current: Stages 72–80, RGB learning pathway

Launch validation: two targeted tests passed, plus an eight-window real-checkpoint GPU preflight under FP32 and BF16. The small preflight showed finite nonzero RGB gradients with substantial correction saturation; this is preliminary mechanism evidence, not a generalization result. The queue is launched with sign-in recovery and the existing progress window. Its larger seven-checkpoint probe will test consistency of the finding.

User authorized the next investigation. See [execution plan](PATHWAY_CYCLE_PLAN.md) and [timing provenance](SENSOR_TIMING_PROVENANCE.md). This queue examines training-only gradient flow, feature variation and correction saturation under FP32/BF16 across seven saved models. Timing conversions remain unresolved; no image remapping or optimizer updates are included. Same progress monitor, automatic stage advancement and 15-second supervisor. Preliminary runtime estimate 5–15 minutes. Final recommendation: `results/pathway_v1/stage80/REPORT.md`.

## Next cycle: Stages 62–71

Completion review: all ten diagnostic stages completed with zero recorded failures at 12:18:56 Bangladesh time on 10 November. All six available raw-prediction references reproduced exactly (maximum logit error 0); the seventh previously used fallback and has no saved raw reference. Pooled original/gray/shuffled success is essentially unchanged for two seeds; seed 7 drops about 0.12 percentage points with gray images. Scenario 38 remains about 66.2% under all three interventions. These findings suggest weak useful image dependence in the current checkpoints, not proof that images are inherently unhelpful. Scenario 36 shows a larger shuffle effect (86.00% to 85.29%). Scenario 39's raw diagnostic correction reaches 68.51%, but its previously selected deployment policy remains geometry; switching based on held-scenario results would contaminate that evaluation. The next priority is verifying timestamp conventions and auditing the RGB learning pathway before new training. Calibration/test remain reserved; no additional stages were launched by this review. See `results/diagnostic_v1/stage71/REPORT.md`.

Launch confirmed: Stages 62–63 completed and Stage 64 started automatically. Three targeted diagnostic tests passed. The native progress window is open and the diagnostic service has sign-in recovery. The filename-clock audit found median row-minus-front-RGB offsets near 0.00, -0.90, +1.50 and -1.10 seconds for Scenarios 36–39 respectively. These are not yet established physical synchronization errors; sensor-clock conventions must be verified, especially before any causal real-time prediction claims or timestamp-based remapping.

User authorized proceeding after the Stage 61 review. [Diagnostic execution plan](DIAGNOSTIC_CYCLE_PLAN.md): historical input/timestamp audit, scenario/trajectory failure analysis, and original/gray/shuffled-image inference for seven frozen multimodal checkpoints. No additional training or test evaluation is included until these findings justify the next change. The same progress window tracks `results/diagnostic_v1/queue.json`; preliminary estimate 10–30 minutes. The final stage produces a decision rather than starting speculative retraining.

## Current: automatic Stages 44–61

Completion review: all 18 stages finished without recorded queue failures at 12:04:52 Bangladesh time on 10 November 2026. Frozen correction-source hashes still match. Geometry macro success is 78.56%; compact GPS raw mean 78.41% (all pooled runs fall back to geometry); bounded multimodal mean 79.00%, a 0.44 percentage-point gain. Its paired macro trajectory-bootstrap interval is -2.56 to +2.27 percentage points, and it wins only one of four unseen-scenario comparisons. In unseen Scenario 38 it falls to 66.16% versus geometry's 70.71%. The development screen failed; publication readiness is not established. Calibration/test remain reserved. Next priority is diagnosing scenario-specific camera/geometry failure and checking matched published baselines, rather than extending epochs indiscriminately. Multimodal pooled best checkpoints were epochs 3, 1 and 1. Final evidence: `results/conservative_v1/stage61/REPORT.md`. No further stages are queued by this review.

Launch verified at approximately 10:21 Bangladesh time: Stages 44 and 45 completed without failure; Stage 46 started automatically. All 43 unit/integration checks passed, including GPU training and checkpoint resumption for both correction models. The native progress window is open and startup recovery is installed.

User authorized the next cycle with the existing progress bar. See [the execution plan](CONSERVATIVE_CORRECTION_PLAN.md). Eighteen sequential stages audit coordinates and failure patterns, train compact GPS and bounded multimodal corrections (three pooled seeds and four held-out scenarios each), and generate uncertainty comparisons and a final report. Outputs and live queue: `results/conservative_v1/`. Initial estimate: 1–3 hours, pending measured epoch times. Calibration/test remain reserved. The earlier completed-stage notes below are historical snapshots.

## Armed: automatic conditional follow-up after Stage 36

### Completed through Stage 43 — 10 November

- Follow-up queue finished at 09:59 Bangladesh time. Geometry-assisted P3 selected validation macro success within 3 dB at five beams: seed 42 74.47%, seed 7 77.49%, seed 13 72.18%; mean 74.71%.
- This improves on the earlier power-aware mean of 49.21%, but falls 3.85 percentage points below analytic geometry (78.56%). The learned GPS baseline was 73.00%. The geometry-assisted network has not demonstrated an advantage over the strongest baseline.
- Pooled paired trajectory-bootstrap interval versus analytic geometry: -12.55 to +1.50 percentage points. This is a pooled, selection-conditioned validation interval; it is not an interval for the macro-scenario mean or a test significance claim.
- Calibration/test remain unopened. At the Stage 43 review, no follow-up stages were queued; the subsequently authorized Stages 44–61 are documented above. The results are not publication-ready for a claimed performance improvement.
- Recommended next cycle, not yet launched: audit panel/beam coordinate conventions and vehicle heading; analyze where learned residuals help/hurt the analytic prior; evaluate conservative residual correction/fallback and compact geometry/GPS alternatives; test front/rear-view coverage through controlled ablations if warranted; then repeat unseen-scenario checks and uncertainty analysis before freezing a model for held-out evaluation. Avoid adding epochs or a large sweep without a specific hypothesis.

### Decision and recovery — 10 November

- Stages 8–36 completed. The development gate failed all three checks. Three-seed mean macro-scenario validation success within 3 dB at five candidates: GPS 73.00%, RGB 46.68%, fusion CE 51.33%, original profile objective 48.54%, power-aware 49.21%. These are not Top-1 accuracies.
- Calibration and test stages recorded explicit skips. The service selected `improve_validation`; Stage 37 completed and follow-up Stages 38–43 were queued automatically.
- Stage 38 stopped on an installed-Matplotlib incompatibility (`boxplot(labels=...)`). Fixed axis labeling to use separate tick setters, verified real plot rendering plus five routing/model-adapter tests, and restarted the queue.
- Recorded the chart-only source amendment and old/new hashes in `results/aftercare_v1/plot_compatibility_recovery_20261110/`. Original Stage 8 scientific source hashes and all training configurations/results remain unchanged. Preserved the Stage 38 failure history before resetting its retry allowance.

- User requested productive automatic follow-up while sleeping. See `paper/AFTER_STAGE36_PLAN.md` for the fixed two-branch plan, Stages 37–43.
- A separate service waits for successful completion of both the existing queue and Stage 36, checking every 15 seconds. It does not modify or interrupt the running experiment.
- Failed development gate + unopened test: sampled image-quality EDA, validation failure analysis, three fresh geometry-assisted residual P3 runs, uncertainty comparisons and a next-cycle report. Calibration/test remain reserved.
- Any final-test exposure: image-quality EDA, validation diagnostics, checkpoint inference reproducibility, locked-test evidence audit, working manuscript and review package. No automatic test-driven retraining, including when test results disappoint.
- Existing progress window automatically follows the selected branch. Service state is `results/data_acquisition/aftercare_supervisor.json`; follow-up queue/results live in `results/aftercare_v1/`.
- Five new CPU tests passed for routing, test-exposure protection, bounded plans, geometry-wrapper gradients and shuffled-batch alignment. The real-data geometry prior reproduces 99.988% of validation top-1 choices from the analytic baseline; calibration/test prior rows stay untouched. Small endpoint/tie differences are retained as numerical provenance.

## Legacy cleanup and EDA index — 9 November

- User selected an archive, not permanent deletion, for their previous Scenario 36-only artifacts. Moved 56 files to `archive/scenario36_legacy_2026-11-09/`, preserving relative paths and verifying every SHA-256 hash against its pre-move value. `MANIFEST.json` is the restoration inventory.
- Replaced the root README with the current multi-scenario entry points; the original README is preserved in the archive. Added `paper/EDA_COVERAGE.md` to distinguish completed four-scenario core EDA from remaining image-quality, model-error and timing-sensitivity analyses.
- Raw Scenario 36 data, all multi-scenario experiment directories, the current cache, active training code and shared dependencies remain in place. No declared scientific code was changed by this cleanup.

## Active: publication-development pipeline, Stages 8–36

### Operational recovery — 9 November, evening

- Stages 8–18 completed. Stage 19 stopped after three transient Windows access-denied errors replacing `status.json`; the last saved checkpoint contains epoch 3. This was a progress-file IO failure, not a numerical training failure.
- Corrected monitor reads to share delete/rename access. JSON writes now use unique temporary files and retry transient permission locks for up to 10 seconds while preserving the previous valid JSON. Persistent permission failures still surface.
- Monitor now explicitly labels a stopped queue instead of leaving the last training phase as its headline.
- Preserved prior failure history and recorded the operational-only amendment in `results/publication_v1/operational_recovery_20261109/amendment.json`. All existing declared experiment source hashes, configurations and checkpoints remain unchanged. Only status IO and the display changed.
- All 30 tests passed, including real Windows sharing-lock recovery. Restarted Stage 19 from its epoch-3 checkpoint; completed stages are not rerun.

- User authorized an automatic sequence aimed at publication-quality evidence. This is a finite 29-stage programme; completion does not itself establish novelty, SOTA or acceptance.
- Authoritative plan: `src/publication_plan.py`; scientific protocol and source audit: `paper/BENCHMARK_PROTOCOL_LEDGER.md`. Queue state: `results/publication_v1/queue.json`. Same desktop progress window automatically follows each stage and displays the primary power-oriented validation metric.
- Stages 8–9: fingerprinted 96×96 front-image cache; strict raw-input split-disjointness audit; training-only scaling; source/environment snapshots; local geometric and train-frequency baselines; scenario-level label support checks. Original raw data and earlier checkpoints remain intact.
- Stages 10–24: GPS-only, RGB-only, fusion CE, original profile objective, and power-aware objective; seeds 42,7,13; eight epochs each. Shared dropout 0.25, AdamW weight decay 0.001, fixed BatchNorm running statistics, clip-consistent brightness variation, cosine LR 1e-4 to 1e-5. All start fresh; no inherited training state.
- Primary checkpoint selection: macro-scenario validation fraction with a beam within 3 dB of optimum among five candidates. Secondary metrics retain Top-1/3/5/10/15/20/32/64 and mean power regret. This is a deliberate new experiment objective, not a retroactive change to previous reports.
- Stages 25–32: four unseen-scenario development checks for fusion CE and power-aware P3. Exclude the held-out scenario from training AND checkpoint-selection validation, then evaluate that scenario's validation blocks.
- Stage 33: predeclared evidence gate. Require at least +1 percentage point over every matched baseline on three-seed mean primary metric, positive paired trajectory-bootstrap lower bound against the strongest baseline, and improvement in at least 3/4 unseen-scenario checks. Freeze checkpoint hashes, code, data and evaluation rules if passed.
- Stages 34–35: only if the gate passes, calibrate fixed-k and adaptive probability-prefix candidate sets to a 90% near-optimal-beam target and run locked test evaluation. Temporal dependence means no exchangeable conformal guarantee is claimed. Report empirical coverage, candidate counts, full-sweep reference, per-scenario outcomes and trajectory-bootstrap intervals. If the gate fails, these stages record a skip without opening held-out data.
- Stage 36: actual-results report, scientific comparison figure and explicit readiness assessment at `results/publication_v1/stage36/REPORT.md`. It must disclose missing exact published-method reproductions, novelty review, end-to-end online cost evaluation and manuscript/author/venue decisions. It must not mark the paper publication-ready automatically.
- Publication comparison audit found mismatched 64-vs-256 beam protocols, random-vs-trajectory splits and different power-loss formulas. See the ledger; no direct headline-score SOTA ranking is justified.
- Preserve 15-second restart checks and atomic checkpoints. Resume at the last complete epoch with per-epoch RNG seeds. Three failures flag needs-attention; no dependent stage advances without successful completion. Sign-in recovery installed for this queue.
- Runtime: multi-day budget; precise estimate awaits the first full cached-image epoch. Original decoding-heavy timing is not a reliable estimate after caching. Cache requires roughly 3 GB additional space; model/optimizer artifacts add several GB. More than 150 GB was available before launch.
- Verification: 27 tests passed, including GPU optimizer/checkpoint/resume integration, trajectory exclusion, train-only scaling, finite and offset-invariant power-aware loss, fixed BatchNorm, calibration boundaries, queue failure gates and prevention of held-out reads after a failed evidence gate.

## Completed: unattended queue, Stages 4–7

- Verified all four queue stages completed with zero recorded stage failures. Stage 7 completed on 2026-11-09; both fresh-seed best checkpoints and comparison report are present.
- Six-epoch, validation-selected P3 runs: seed 42 top-1 15.64%, seed 7 13.36%, seed 13 14.13%. Mean 14.38%; sample standard deviation 1.16 percentage points. These are validation results, not final test results or a published-benchmark comparison.
- The planned unattended queue is finished. Further ablations, literature/protocol verification, calibration and locked test evaluation remain outstanding.

- User authorized sequential unattended work while away for approximately 10 hours. Queue plan is fixed in `src/unattended_stages.py`; live durable state: `results/unattended_queue.json`. The existing progress window reloads the active stage automatically and lists queued/running/completed stages.
- Stage 4: 2,000 scenario-stratified trajectory-block bootstrap resamples of the saved validation predictions. Paired P3-vs-B1/P1 top-1 differences and percentile intervals. Conditional development uncertainty, not independent test confidence.
- Stage 5: fresh P3 initialization, seed 7, six full train/validation epochs.
- Stage 6: fresh P3 initialization, seed 13, six full train/validation epochs.
- Both fresh runs use the same split, preprocessing, architecture, loss and six-epoch LR schedule as seed 42: 1e-4 for epochs 1–3, 3e-5 for 4–6. Seed 42 historically resumed between stages; new seeds run continuously, so execution histories are not identical. No weights are copied between seeds.
- Stage 7: select the best validation-top-1 checkpoint within the first six epochs of each seed, then report all three runs, their mean and sample standard deviation. Final report: `results/stage7_comparison/REPORT.md`.
- Estimated duration: about 10–12 hours at the observed throughput; not guaranteed to finish within the absence. Queue continues until Stage 7 completes, even if that exceeds 10 hours, then stops. No unbounded experiment generation.
- Dependencies: a stage advances only after the worker succeeds and writes complete output. Failed stages restart after 15 seconds, preserving atomic completed-epoch checkpoints. Three failures stop the queue with needs-attention status. Surviving stage workers are adopted by PID plus creation-time checks to avoid duplicate GPU jobs after queue restart.
- Windows sign-in resumes the queue. A root `pause.flag` pauses between stages or at the next training epoch boundary. Local data means internet is unnecessary for these tasks. Training failures cannot generally be fixed just by checking internet connectivity.
- Calibration/test remain reserved; no final-test selection, publication, or external messages. All 19 tests passed, including automatic advancement and failure-gating tests; progress-window syntax checked.

## Completed: Stage 3

- All five frozen validation passes finished. P3 top-1 15.63%, top-5 43.03%; zero-GPS top-1 10.02%, zero-RGB 2.58%. Small differences from training logs can arise from inference batching/ties; these diagnostics use saved predictions.
- Top-5 shortlist within 3 dB: B1 59.91%, P3 57.17%, P1 42.91%. P3's higher exact-beam accuracy does not establish superiority on every power-oriented metric. This is a key reason to retain multiple metrics and uncertainty analysis.

## Stage 3 protocol (completed)

- Same desktop progress window, now tracking five evaluation passes: B1, P1, P3, P3 with normalized GPS zeroed, P3 with normalized RGB zeroed. Local completion alerts and the 15-second supervisor remain enabled; sign-in recovery installed.
- Evaluate Stage 2 best validation-top-1 checkpoints, not final epochs. Verify metadata, split manifest, and scaler match training exactly before inference.
- Record pooled and per-scenario Top-k, mean shortlist power loss, and fraction with a candidate within 3 dB of the measured best, for k=1,3,5,10,20,32,64.
- Save validation predictions and checkpoint provenance for reproducible analysis. Full report will appear at `results/validation_diagnostics_v1/REPORT.md` on completion; live status is in that directory's `status.json`.
- Input-zeroing is a sensitivity diagnostic with potentially out-of-distribution inputs, not a retrained modality ablation. Offline shortlist success is not online probing savings. Calibration/test remain reserved.
- Estimated 30–60 minutes pending observed inference timing. Completed passes are retained; a failed pass restarts. All 17 automated tests passed, including a metric test separating exact-beam accuracy from near-optimal-power success.

## Completed: Stage 2, lower-learning-rate continuation

- Finished 2026-11-09 at 02:22 Bangladesh time. B1 reached epoch 12; P1 stopped at 6; P3 stopped at 10 (validation early stopping).
- Selected best validation top-1: B1 10.36% at epoch 10; P1 11.34% at epoch 1 (pilot checkpoint retained); P3 15.64% at epoch 6. These are development validation results; no SOTA claim or test result.

- User authorized the next stage and the same progress window. The desktop shortcut now reads `results/active_training.json`, selecting `results/multiscenario_extended_v1` and the extended supervisor.
- Resume all three models from pilot epoch 3 with optimizer moments preserved; reduce learning rate from 1e-4 to 3e-5. Maximum cumulative epochs: 12/model (9 additional). Keep all pilot files unchanged and retain their best checkpoints in the new stage.
- Stop a model after at least 6 cumulative epochs if validation top-1 has not improved for 4 epochs. Best validation checkpoint remains available even if later training degrades. This is exploratory continuation, not a fresh independent run or a completed ablation.
- Motivation: P1 training top-1 rose to 51.86% while validation fell to 7.37%; P3 also shows a substantial gap. Lower LR is a bounded diagnostic intervention, not a guaranteed fix for generalization. Further training is capped with early stopping.
- Same data, split, seed, resolution and losses as the pilot. Check source metadata hashes and exact split-manifest hash before training. No calibration/test predictions.
- Estimated additional wall time: roughly 5–15 hours from measured 48–50 minute image-model epochs; actual stop point depends on validation.
- Read live status/history and `MODEL_finished.json` in the new stage folder. Epochs 1–3 in its histories are inherited, explicitly recorded in `continuation_provenance.json` with source checkpoint hashes.
- Same 15-second watchdog, atomic epoch checkpoints, pause flag, sign-in recovery and local completion notifications. Monitor labels now use the selected stage and epoch limit, including early-stopping outcomes.
- Validation before launch: 16 automated tests passed; Python compilation and PowerShell parsing passed.

## Completed

- Part 1 acquisition and extraction for scenarios 36–39. Verified archives deleted.
- Integrity audit: 892,474 files checked, zero reported errors; 112,196 CSV samples.
- EDA, sequence/time-gap checks, and trajectory-block split audit.
- 26 GPU execution checks passed. These are not performance experiments.
- Existing automated checks: 15 tests passed on 2026-11-08; the additional pilot-supervisor routing test also passed (three supervisor tests rerun).

## Completed: pooled pilot v1

- Finished 2026-11-08 at 17:43 Bangladesh time. All three models completed three epochs each (9/9); supervisor exited successfully, first attempt. No training job remains active in this queue.
- Image-model epochs took approximately 48–50 minutes each; the full queue took about five hours including preparation.
- Best validation top-1 per model: B1 7.49% (epoch 3), P1 11.34% (epoch 1), P3 14.28% (epoch 3). These are local blocked-protocol pilot results, not official benchmark or test results.
- Next: inspect the validation degradation in P1 and generalization across scenarios before expanding training. Calibration and final test evaluation remain pending.
- Progress monitor array handling corrected: shows 9/9 completed epochs and numeric percentages rather than nested-array text. Completed jobs now show the finish time instead of blank epoch/batch fields.

- Local progress UI: desktop shortcut **DeepSense Training Progress**, backed by `Training Progress.ps1`. Refreshes every 15 seconds, shows saved validation metrics, current phase progress and heartbeat age. Closing hides it to the tray; right-click the tray icon to exit.
- Completion alerts: local Windows notification, sound, and reopened progress window for the entire B1/P1/P3 pilot. Failures/stale supervisor heartbeat also trigger a notification. Windows notification settings may suppress balloons; the progress window remains available. No email or phone notifications are configured.
- Monitor starts in the tray at Windows sign-in. It is read-only and independent of the training worker; exiting the monitor does not stop training. Completion does not mean the full publication study or final test evaluation is finished.

- Live state: `results/multiscenario_pilot_v1/status.json`.
- Watchdog: `results/data_acquisition/pilot_supervisor.json`.
- Worker output: `results/data_acquisition/pilot_worker.log`.
- Queue: B1 GPS-only → P1 GPS+RGB classification → P3 GPS+RGB classification/profile.
- Three full training/validation epochs each, seed 42. These are pilot runs, not converged paper results.
- Four scenarios, five spaced historical inputs and a target five samples after the last input. Actual time horizons are recorded in the split manifest.
- Trajectory blocks cannot cross splits; windows cannot cross original sequences.
- All eligible training/validation windows are used. Calibration/test are reserved, with no prediction-based evaluation or model selection on them.
- Front camera RGB5 at 96×96; full model width 256; three GRU/fusion layers; eight attention heads; dropout 0.12; AdamW LR 1e-4, weight decay 3e-4; BF16; batch 64 GPS / 16 multimodal; gradient clipping 1.
- P3 profile weight 0.1, ranking weight 0.05, smoothness weight 0 (avoid smoothing across array-panel seams). GPS velocity fields retain the existing displacement-proxy definition.
- Train-only GPS scaling and majority baseline. Validation top-1/3/5 and positive power loss in dB; lower power loss is better.
- Atomic latest/best checkpoints after each epoch. Unexpected worker exits are retried after 15 seconds. Interrupted epochs restart from the last completed epoch. Windows sign-in restarts the supervisor; the machine must be powered on and awake.
- Training uses local data and cached weights, so internet outages do not interrupt it. `pause.flag` requests a pause at an epoch boundary.

## Time budget (provisional, not a completion promise)

- Preparation completed in about 75 seconds. First GPS training+validation epoch completed in 34.72 seconds, including initial loader startup. Atomic latest/best checkpoints were written successfully.
- Actual pilot population: 54,604 training windows and 16,643 validation windows. First-epoch validation top-1: 3.31%, top-5: 35.44%. This is an early baseline, not a final result.
- Three-model, three-epoch pilot: budget 2–8 hours until real epoch measurements replace this range. Raw image decoding can dominate runtime.
- Initial full training and validation comparisons: approximately 1–3 days of continuous GPU availability, to be revised after the pilot.
- Broad multi-seed, ablation, held-out-scenario and calibration study: potentially 1–3 weeks on one RTX 4060, depending on selected experiments and convergence. Exhaustively testing every parameter combination would take much longer and is not planned.
- Paper preparation and external review have no reliable fixed ETA yet.
- Every completed epoch writes its measured wall time to `MODEL_history.json`; multiply by remaining epochs for a same-model estimate. Extrapolating GPS runtime to RGB is invalid.
- Per-scenario accuracy and power loss are sample-specific. The per-scenario `loss` field is a sample-weighted batch-loss diagnostic; mixed training batches prevent interpreting it as exact scenario-specific objective loss.

## Next gates

1. Inspect actual pilot learning curves, memory, timing, and scenario imbalance; fix pipeline issues before scaling up.
2. Verify primary published papers and official benchmark splits; record exact datasets, modalities, horizons, metric signs, and source versions. Current local split is not an official challenge reproduction.
3. Reproduce geometry and strong learned baselines under a matched protocol. Establish a verified comparison target before claiming state of the art.
4. Select model/settings using validation only; extend training with convergence/early stopping. Compare resolutions, learning rates and loss weights in controlled stages.
5. Multi-seed runs (planned 42, 7, 13, 21, 101), ablations, per-scenario/pooled/leave-one-scenario-out comparisons.
6. Calibrate on reserved calibration data; final locked test evaluation and trajectory-level uncertainty intervals.
7. Audit realistic feedback availability and full-sweep measurement cost before claims about beam-probing savings. Existing oracle feedback is insufficient for those claims.
8. Rewrite paper around supported novelty, robust results, limitations, and reproducible artifacts. No promise of beating a published result or acceptance.

## Decisions still open

- Venue, author list, and exact published comparison target.
- Full experiment budget after measured pilot timing.
- Physical-time velocity features, stronger temporal baselines, and RGB resolution are explicit future ablations, not silently changed inputs.

Existing design: `BLUEPRINT_V5_MULTISCENARIO.md`. This tracker records execution status; machine-generated status/history files take precedence over this dated snapshot.

Autonomous review 10 Nov 2026: factorial stages 124-133 complete, no validation gain. Launching matched twelve-epoch residual-penalty study stages 134-139; see AUTONOMOUS_5H_TRACKER.md.

Stages 134-139 complete: no-penalty visual 79.1642% vs GPS/geometry 78.5617%, interval includes zero. Stages 140-145 launch matched seed-7/13 replication; no publication claim or test access.

Stages 140-145 complete: visual mean 78.9842%, GPS 78.6226%, uncertainty crosses zero. Stages 146-150 audit original/gray/shuffled image dependence across all three saved visual checkpoints.

Stages 146-150 complete: original-minus-shuffled mean +0.535 pp, uncertainty includes zero. Stages 151-154 compare native-derived 96/192-pixel frozen features under identical training.

Five-hour autonomous window closed after Stage 154; all queues complete, no active training. Best single-seed 192-pixel validation 79.5434%, matched 96-pixel 79.2348%; confidence interval includes zero. Next: matched resolution seed replication and GPS control. See AUTONOMOUS_5H_TRACKER.md for full results and operational limitations.

## Matched resolution replication: stages 155-163

Authorized by the user after the five-hour window, 2026-11-10T17:03:06.771609+00:00. This is a finite new queue, not a renewal of recurring reviews. Seven 12-epoch runs: GPS controls for seeds 42/7/13 and 96/192-pixel visual models for seeds 7/13. Reuse seed-42 visual runs 152/153 after verifying identical visual forward behavior. Same head initialization, input pipeline, optimizer, schedule, zero residual penalty, expected-regret coefficient 0.05 and no brightness augmentation. GPS controls zero visual features before the scoring head; images cannot affect their predictions. Existing frozen files preserved.

Validation: exact visual-forward equivalence to resolution_v1, GPS image independence with nonzero output weights, full stage sequence, and CUDA two-epoch training/checkpoint-resume for all three arms passed. Estimated runtime 25-35 minutes. Stage 163 reports three-seed means, paired resolution/GPS/geometry contrasts and per-scenario results. Selected validation checkpoints and prior seed-42 screening remain exploratory; no guarantee of publication readiness. Calibration/test remain reserved. Native progress window, 15-second supervisor and sign-in recovery maintained.

Stage 163 completed 2026-11-10T17:27:26.179750+00:00; see results/rescheck_v1/stage163/REPORT.md.

## Stages 164-173: prespecified transfer study

Authorized 2026-11-10T17:34:13.447821+00:00. Eight sequential 12-epoch seed-42 runs, GPS and 192px visual for each excluded scenario 36/37/38/39. Fixed prior protocol, no brightness augmentation, K20, unbounded hard-objective corrections, expected-regret 0.05, residual penalty zero. GPS/context scaling and geometry offsets use training scenarios only; checkpoint selection excludes the held scenario. Held scenario validation evaluated only after best checkpoint selection, without fallback or post-hoc gating. All calibration/test remain reserved. These scenarios were previously explored, so this is development transfer evidence, not an untouched external test.

Targeted CUDA checks passed for both arms: held-scenario training rows never loaded, checkpoint-selection IDs exclude held scenario, held validation prediction occurs afterward, and resume preserves results. Existing scientific files remain frozen; new queue/protocol/report frozen at stage164. Estimated 25-40 minutes. Native progress display, sequential supervisor and sign-in recovery retained. Report at results/transfer_v1/stage173/REPORT.md.

Stage 173 completed 2026-11-10T17:59:23.382556+00:00; see results/transfer_v1/stage173/REPORT.md.

## Stages 174-179: transfer failure audit

Authorized 2026-11-10T18:06:37.947848+00:00. CPU-only input distribution and correction audit, no fitting or test/calibration predictions. For each excluded scenario compare GPS and normalized cached visual feature distributions against statistics fitted only on sampled training rows from the other scenarios. Use up to 512 deterministic windows per scenario/split, selected without labels. Compare seen vs held shift and count geometry-to-model top-five successes helped/harmed on full saved validation predictions. Do not treat correlated windows as independent observations or distribution differences as proof of causality. Unit checks passed for correction signs/counts, shift identity, split sampling and stage sequence. Existing source hashes verified. Separate src/experiments/runner.py process observed and left untouched; no new GPU worker launched. Stage179 report proposes a training-only inner-split robustness study; does not start it or tune thresholds on excluded labels.
- Audit 174-179 completed without failures. Visual feature shift is larger on excluded scenarios than seen validation in every fold. S38 corrections help 20/harm 97 windows; S39 help 38/harm 191. This suggests testing visual robustness, not proof of causal distribution-shift failure. Report: results/failure_v1/stage179/REPORT.md. No new model trained; separate experiment runner left untouched.

## Stages 180-189: fixed visual robustness comparison

Authorized 2026-11-10T18:10:16.734971+00:00. No active Python process observed before launch. Two prespecified variants across each excluded scenario: whole-clip visual feature dropout probability 0.5; the same dropout plus symmetric prediction KL consistency coefficient 0.1 between clean-image and zero-image candidate scores. Consistency gradients flow through both branches; inference uses original features without dropout. All other settings match the 12-epoch seed-42 192px transfer protocol. Compare both against frozen original/GPS/geometry results; no held-label selection between variants, thresholds or scenario-specific fallback. This finite fixed-setting comparison avoids an expensive hyperparameter search; any subsequent search must use inner splits of training scenarios. Epoch selection uses only non-excluded scenario validation. Previously explored outer scenarios remain development evidence, not independent confirmation.

Checks passed: full-dropout image independence with nonzero scoring weights, finite consistency gradients, clean evaluation behavior, both CUDA training/checkpoint/resume arms and correct held-scenario prediction separation. Scientific files frozen separately. Eight sequential jobs, estimated 30-45 minutes, native progress window and recovery maintained. Final report results/robust_v1/stage189/REPORT.md. Calibration/test remain reserved.

Stage 189 complete 2026-11-10T18:42:30.285795+00:00; see results/robust_v1/stage189/REPORT.md.

New 17-hour autonomous window authorized, hourly reviews until 11 Nov 2026 17:42:50 Bangladesh. Initial stages 190-210 replicate robustness and evaluate pooled/transfer tradeoffs. Details: AUTONOMOUS_17H_TRACKER.md.
Three-seed robustness replication completed at Stage 210. Stages 211–217 evaluate whether reserving three geometry beam choices preserves transfer performance. See AUTONOMOUS_17H_TRACKER.md.
Stages 218-220 complete. Scenario 39 accounts for most of the observed pooled-transfer contrast. See results/variation_v1/stage220/REPORT.md.

Completed Scenario 39 contribution analysis on frozen validation predictions. Sequences 50 and 51 account for 77.5% of the signed pooled-minus-transfer contrast. Full results in results/scenario39_contributions_v1/REPORT.md.

Train/validation metadata audit complete; no shared frame rows, image paths or sequence IDs within scenarios. Stages 221–227 test temporal endpoint differences against matched mean-feature models.
Stages 221–227 completed. The seed-42 pooled temporal gain over the mean was 0.359 pp with a block interval crossing zero. Stages 228–239 replicate that comparison on two additional seeds with matched controls.

Stage 239 and the paired power-loss audit completed. See results/power_loss_review_v1/REPORT.md for dB outcomes and the 17-hour tracker for the next candidate-coverage diagnosis.
Top-20 coverage audit completed. Stages 240-249 test fixed K40 against K20 using matched GPS and 192px visual models across four transfer folds.
Stages 240–249 and the saved-prediction attribution audit complete. Stages 250–267 replicate the fixed K40 comparison for two additional seeds.

Stages 250–267 complete, 11 November 2026. Three-seed transfer replication of 40 versus 20 beam candidates is recorded in results/widecheck_v1/stage267/evidence.json. At K40 the paired visual gain over GPS on primary success was +0.011 percentage points; its block interval included zero. Closed the width sweep and updated paper/POOLED_AND_TRANSFER_RESULTS_DRAFT.md. Calibration and final test remain reserved.

11 November 2026, 15:02 Bangladesh. Started the separately requested publication comparison. Stage 268 froze the protocol and source and began extracting separate front/rear 224-pixel EfficientNet-B3 spatial features. Observed extracting_S36, 2736/14625, on supervisor attempt 1. Stages 269–298 run thirty matched fits (two arms, five partitions, three seeds). Stage 299 computes paired success, exact-beam accuracy, power loss and intervals. Archive the comparison under the name paper-based adaptation. The author implementation has not been verified. Existing 15-second supervisor and progress monitor are active. Full training ETA pending measured epoch durations. See paper/SWIN_COMPARISON_PROTOCOL.md.

Stage 299 complete at 17:04 Bangladesh time, 11 November. Verified twelve recorded epochs for all thirty fits and independently recomputed top-five exact and within-3-dB rates from validation predictions. The primary pooled-minus-transfer contrast was inconclusive. The architecture-specific reversal between exact and near-optimal metrics warrants reporting alongside the strong geometry baseline. Results and limitations incorporated into the manuscript. Next task is to integrate the benchmark scope, methods and limitations into the submission draft. Calibration and final test remain reserved.
Completed the integrated development manuscript in paper/V2V_BEAM_FORECASTING_MANUSCRIPT.md and its transfer outcome figure. Checked manuscript contrasts against Stage 299, local artifact links, validation horizon and counts, and the unique strongest-power label on all 16,643 development windows. Author list, venue selection and release packaging await editorial decisions.
Reviewer audit: reproduced 20 full-codebook estimates and intervals from recording-block summaries in a freshly extracted offline review package. Alternative-only transfer success fell in all four scenarios. Reverse-order score ties preserved the primary direction. Updated the manuscript and built a portable HTML reading copy and a ZIP of the evidence. No external submission.
TVT audit complete. Prepared and visually reviewed a three-page IEEEtran working PDF, verified nine references and documented cross-environment prior art. TVT rule 9 requires human-authored article content. A fresh author template and review notes are in paper/TVT. Authorship, affiliation and approval remain pending.
