# Camera and correction study

Exploratory validation selection; gain threshold 0.25 percentage points, finite positive RGB gradients and gray sensitivity above 1e-6. No publication claim.

- Pilot 82: front / tanh_bf16: 78.43%; expansion eligible: False.
- Pilot 83: front / tanh_fp32: 78.56%; expansion eligible: False.
- Pilot 84: front / linear: 67.72%; expansion eligible: False.
- Pilot 85: rear / linear: 68.21%; expansion eligible: False.
- Pilot 86: both / linear: 73.19%; expansion eligible: False.

No pilot passed the predeclared expansion screen; longer runs were skipped.

All results are development evidence. Camera/correction choice used pooled validation; subsequent held-scenario checks exclude their scenario during fitting and checkpoint selection but are not independent of architecture selection. Prior repeated validation inspection further limits confirmation claims.
Physical timestamp conversion remains unresolved. No time shifts or calibration/test evaluation were performed. Novelty and exact published-baseline reproduction remain outstanding.