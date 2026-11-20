# Four-scenario transfer comparison

Seed-42 leave-one-scenario-out development evaluation. Held-scenario validation labels excluded from fitting and checkpoint selection. These scenarios were previously explored in pooled and earlier transfer studies; this is not an untouched external test.

- Excluded S36: visual 83.94%, GPS 84.59%, geometry 84.05%.
- Excluded S37: visual 95.23%, GPS 95.23%, geometry 95.23%.
- Excluded S38: visual 68.42%, GPS 69.99%, geometry 70.71%.
- Excluded S39: visual 57.35%, GPS 62.16%, geometry 62.16%.

Macro means: {'visual': 0.7623274697236635, 'gps': 0.7799193549626076, 'geometry': 0.7803649877611816}
Review transfer consistency and failures before additional seeds or final protocol decisions. Do not tune on these held-scenario results and present them as independent confirmation.
Calibration and final test remain reserved.