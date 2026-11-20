# Conservative-correction research results

Exploratory selection-validation and unseen-scenario validation. Fallback creates optimistic selection-validation scores; transfer evidence is required.

Selected analytic convention: legacy_latlon; selection-validation macro success 78.56%.

- compact_gps: raw learned mean 78.41%; after validation-selected fallback 78.56%; corrections enabled in 0/3 pooled runs; unseen-scenario wins 1/4.
  Paired macro trajectory-bootstrap interval: [0.0, 0.0].
- bounded_multimodal: raw learned mean 79.00%; after validation-selected fallback 79.00%; corrections enabled in 3/3 pooled runs; unseen-scenario wins 1/4.
  Paired macro trajectory-bootstrap interval: [-0.025602233951692158, 0.022696152950094348].

Do not open test. Investigate geometry/camera observability and whether simpler analytic methods are sufficient.

No calibration/test data was used in this cycle. Publication readiness, novelty, exact published-method comparisons and physical mounting/camera coverage remain separate review requirements.