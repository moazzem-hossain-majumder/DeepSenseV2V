# Automatic follow-up after Stage 36

Authorized on 2026-11-10. This is a separate, bounded follow-up cycle; the running Stages 8–36 and their declared sources stay unchanged.

## Handoff

`src/aftercare_service.py` checks every 15 seconds for both the parent queue and Stage 36 to complete successfully. It does not take over the progress window or launch GPU work early. It then saves a durable branch decision, starts Stages 37–43, and supervises retries. The same progress window switches automatically. Windows sign-in restarts the service; completed stages and epoch checkpoints are retained.

## Branch A — weak development results, test unopened

The development gate did not pass and there is no record of final-test access.

- Stage 37: record the branch, source snapshots and parent provenance.
- Stage 38: image brightness, contrast, saturation and resized-image sharpness proxies from up to 2,000 train/validation input frames per scenario. Export a scientific figure; do not infer semantic day/night labels from brightness alone.
- Stage 39: validation failure analysis by scenario, trajectory and distance; include the analytic geometry baseline, which the parent's learned-model gate did not include.
- Stages 40–42: fresh geometry-assisted P3 models, seeds 42/7/13, eight epochs each. Retain the parent regularization and power-aware objective. Add a train-fitted analytic geometry prior from historical positions, with fixed 0.35-radian width and learned positive gain. The model learns residual corrections. No calibration/test data is used for fitting, selection or evaluation.
- Stage 43: compare all three runs with analytic geometry and parent learned baselines; compute trajectory-bootstrap uncertainty and report the next scientific checks. Do not automatically open calibration/test after a new, exploratory hypothesis.

This is motivated by observed geometric-baseline strength, not a promise of improved results or novelty. Original parent training files are reused through an isolated process-local adapter; their on-disk source hashes remain unchanged. The new wrapper and configuration are separately snapshotted.

## Branch B — final test has been evaluated

Test exposure always selects the evidence branch, even if test scores disappoint. Test performance never triggers an automatic retraining/tuning loop on that same test set.

- Stages 37–39: provenance, sampled image-quality EDA and validation failure analysis as above.
- Stage 40: reload the selected power-aware checkpoint and reproduce its validation predictions; record numerical differences and prediction agreement. This is inference reproducibility, not independent retraining.
- Stage 41: audit existing locked-test coverage, candidate counts and worst scenario/seed coverage. A descriptive check flags mean coverage below 90% or worst scenario/seed coverage below 85%; it does not change any selected model or calibration parameter.
- Stage 42: generate a working manuscript from actual saved evidence, including limitations and the verified source ledger. It does not invent comparisons, claims, references or missing results.
- Stage 43: produce a review package. If results are weak, explain what requires a fresh evaluation design; if promising, identify remaining novelty, exact published-method comparison, realistic online-cost, authorship and venue review.

## Inconsistent or failed state

If the parent gate passed but a supposedly completed queue contains no test record, stop with needs-attention rather than guess a branch. A parent failure does not launch the follow-up. Failed follow-up workers retry after 15 seconds; the queue stops after three failures in a stage. An overall service retry limit also prevents endless unproductive failures. `pause.flag` pauses at the next supported stage/epoch boundary.

## Outputs and stopping rule

Live service: `results/data_acquisition/aftercare_supervisor.json`.
Branch: `results/aftercare_v1/route.json` (created only after the handoff).
Queue: `results/aftercare_v1/queue.json`.
Final report: `results/aftercare_v1/stage43/REPORT.md`.
Evidence-branch draft: `results/aftercare_v1/stage42/MANUSCRIPT_WORKING.md`.

Both branches stop after Stage 43 and leave an evidence-based recommendation. Strong scores do not automatically establish publication readiness. No submission or external messaging is configured.
