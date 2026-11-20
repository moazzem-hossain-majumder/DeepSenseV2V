# Research evidence package

This is a generated development report, not a submission-ready manuscript or a state-of-the-art claim.

## Validation comparisons

- gps: mean macro success within 3 dB at k=5 73.00%; sample SD 3.90%; three seeds.
- rgb: mean macro success within 3 dB at k=5 46.68%; sample SD 5.89%; three seeds.
- fusion_ce: mean macro success within 3 dB at k=5 51.33%; sample SD 4.50%; three seeds.
- profile_original: mean macro success within 3 dB at k=5 48.54%; sample SD 2.01%; three seeds.
- power_aware: mean macro success within 3 dB at k=5 49.21%; sample SD 1.21%; three seeds.

Automatic development gate: not passed.
Strongest matched baseline: gps. Paired validation block-bootstrap interval: [-0.2895083149886749, -0.20970264337301942].

## Held-out evaluation

Calibration and test were deliberately skipped because the predeclared development gate failed. Further hypotheses are required before opening the test set.

## Claims and outstanding work

- This is the local 256-beam forecasting protocol. Published headline scores on different splits, codebooks or modalities are not directly comparable.
- Exact reproduction of a stronger published architecture remains outstanding; local fusion baselines are not substitutes for claiming published SOTA.
- Candidate-set power metrics are offline. End-to-end measurement, feedback latency and temporal coverage guarantees have not been established.
- Novelty assessment, failure analysis, authorship, venue selection and manuscript review remain necessary.
- No acceptance or publication-readiness claim follows automatically from passing the development gate.

Protocol: paper/BENCHMARK_PROTOCOL_LEDGER.md. Source snapshots and environment: stage8/.