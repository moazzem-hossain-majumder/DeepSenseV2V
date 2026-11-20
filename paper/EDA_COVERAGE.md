# EDA coverage — Scenarios 36–39

Checked against the actual saved outputs on 2026-11-09.

**Core descriptive EDA and data integrity are complete for all four scenarios. The broader publication analysis is not entirely complete.**

## Completed

- Full Part 1 file/reference audit: 112,196 CSV rows across four scenarios; 892,474 files scanned with zero file errors. Checks include full JPEG decoding, numeric GPS/PWR parsing, source metadata agreement and local fingerprints. See [integrity summary](../results/readiness/integrity_summary.json).
- Per-scenario row counts, sequence counts, trajectory-block counts, duplicate IDs, missing metadata and timestamp intervals/gaps.
- GPS receiver-speed distributions, transmitter–receiver separation and HDOP summaries.
- Beam-class and panel distributions, class entropy, active beams and descriptive majority fractions.
- Best measured power distributions and near-optimal beam-set sizes at multiple dB tolerances.
- Qualified window/split counts and actual forecasting horizons; source rows and trajectory blocks do not cross the declared partitions.
- Training-vs-validation class support and unseen-label fractions for each scenario in [Stage 9](../results/publication_v1/stage9/report.json).

## Where to inspect it

- [Four-scenario EDA report](../results/readiness/REPORT.md).
- [Detailed numerical descriptors](../results/readiness/eda.json), keyed by scenario 36, 37, 38 and 39.
- [Scenario overview plot](../results/readiness/scenario_overview.png).
- [Window and split audit](../results/readiness/window_and_split_audit.json).
- Current verified data/split cache: `data/processed/publication_v1/receipt.json` and `manifest.jsonl`.

## Remaining publication analysis

- Quantitative image-quality and visual-domain analysis, such as brightness, blur and scenario differences. Successful JPEG decoding is an integrity check, not an image-quality or day/night analysis.
- Detailed error analysis versus distance, speed, panel, class support and trajectory, using the final selected models.
- Dedicated timing-outlier/horizon-sensitivity experiments; outliers are described and retained, but a sensitivity result is not yet established.
- Completed model-dependent robustness, calibrated reliability and locked-test analysis where the declared gates permit them. These are evaluation stages rather than raw descriptive EDA.

Descriptive all-data statistics must not become test-tuned preprocessing or selection criteria. The readiness report contains historical training-readiness notes; current execution status belongs to [the experiment tracker](EXPERIMENT_TRACKER.md).
