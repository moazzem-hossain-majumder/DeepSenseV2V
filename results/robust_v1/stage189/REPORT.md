# Prespecified visual robustness comparison

Fixed settings, seed-42 exploratory transfer development; excluded scenarios previously examined. No independent confirmation claim.

- S36: geometry 84.05%, gps 84.59%, original 83.94%, dropout 83.86%, consistency 84.15%
- S37: geometry 95.23%, gps 95.23%, original 95.23%, dropout 95.23%, consistency 95.23%
- S38: geometry 70.71%, gps 69.99%, original 68.42%, dropout 70.68%, consistency 70.65%
- S39: geometry 62.16%, gps 62.16%, original 57.35%, dropout 62.16%, consistency 62.16%

Report both prespecified variants. Assess whether transfer harm decreased without selecting per-scenario variants or fallback from held labels. Further hyperparameter search requires inner splits restricted to training scenarios.
Calibration and final test remain reserved.