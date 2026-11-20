# DeepSense V2V — GPS + RGB research across Scenarios 36–39

This workspace studies future beam selection across four DeepSense scenarios, using GPS and front-camera RGB inputs with measured power as supervision. The active work uses trajectory-separated training, validation, calibration and test partitions. Publication claims depend on the completed evidence and protocol review.

## Current work

- [Five-hour autonomous research tracker](paper/AUTONOMOUS_5H_TRACKER.md): scheduled reviews, current experiment decisions and cutoff.
- [Focused mechanism audit, Stages 117–123](paper/MECHANISM_AUDIT_PLAN.md): small-sample memorization, visual inspection and baseline implementation checks.
- [Constrained reranking, Stages 96–116](paper/RERANK_CYCLE_PLAN.md): shortlist feasibility, matched GPS/visual pilots and conditional expansion.
- [Correction/camera experiments, Stages 81–95](paper/CAMERA_CORRECTION_PLAN.md): controlled pilots and conditional multi-seed/transfer training.
- [RGB learning pathway, Stages 72–80](paper/PATHWAY_CYCLE_PLAN.md): timing provenance, gradients, feature variation and saturation checks.
- [Current diagnostic cycle, Stages 62–71](paper/DIAGNOSTIC_CYCLE_PLAN.md): timing checks, failure analysis and image-dependence interventions.
- [Active correction cycle, Stages 44–61](paper/CONSERVATIVE_CORRECTION_PLAN.md): automatic diagnostics, bounded corrections, unseen-scenario checks and evidence report.
- [Experiment tracker](paper/EXPERIMENT_TRACKER.md): current queue, completed stages, results, operational recovery and open work.
- [Research protocol and source ledger](paper/BENCHMARK_PROTOCOL_LEDGER.md): fixed comparison rules, metrics, related-work differences and held-out evaluation gates.
- [Multi-scenario blueprint](BLUEPRINT_V5_MULTISCENARIO.md).
- [EDA coverage](paper/EDA_COVERAGE.md): completed four-scenario EDA and remaining analysis.
- [Automatic follow-up after Stage 36](paper/AFTER_STAGE36_PLAN.md): conditional improvement or evidence-review queue, Stages 37–43.

Open **DeepSense Training Progress** from the desktop to follow the automatic queue. Its active stage is recorded in `results/active_training.json`; the publication-development queue is recorded in `results/publication_v1/queue.json`.

## Data and current outputs

- Raw Part 1 data: `data/raw/scenario36/` through `data/raw/scenario39/`.
- Four-scenario integrity and descriptive EDA: [readiness report](results/readiness/REPORT.md), [numerical EDA](results/readiness/eda.json), [overview figure](results/readiness/scenario_overview.png).
- Current cached model inputs: `data/processed/publication_v1/`.
- Current training and evaluation outputs: `results/publication_v1/`.
- Earlier multi-scenario pilot, continuation, diagnostics and seed runs remain in their named `results/` directories for provenance.

## Legacy Scenario 36 work

The previous single-scenario README, notebooks, draft, Word report, figures, result tables, split files and standalone legacy runners have been moved to [the legacy archive](archive/scenario36_legacy_2026-11-09/ARCHIVE.md). The archive manifest records original paths and verified SHA-256 hashes for all 56 files. Nothing in that archive is presented as a current multi-scenario result.

Raw Scenario 36 data stays in the active dataset because it is one of the four scenarios. Shared source modules, the `train.py` compatibility entry point, `config.yaml`, and the older broader V4 design document remain available; the active research queue uses `src/publication_queue.py` and `src/publication_worker.py`.

## Execution

The installed hidden supervisor starts `src.supervise_work --job publication` with the configured Python 3.11 interpreter. It resumes completed epochs and advances stages after successful completion. A root `pause.flag` pauses at a training epoch or queue-stage boundary. Sign-in shortcuts restore the supervisor and progress monitor.

Calibration and locked test evaluation are gated by the predeclared development criteria. The eventual [evidence report](results/publication_v1/stage36/REPORT.md) is generated only after the queue reaches its final stage; it may not exist yet. Completing the queue does not automatically establish publication readiness.
