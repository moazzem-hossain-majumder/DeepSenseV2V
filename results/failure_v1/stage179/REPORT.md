# Transfer failure audit

Descriptive findings from previously explored validation data; no final-test evaluation or model fitting.

- S36: visual corrections help 12 windows and harm 18; top-five set changes in 4.87% of windows. Visual standardized mean shift: seen 0.180, held 0.601. GPS shift: seen 0.142, held 0.486.
- S37: visual corrections help 0 windows and harm 0; top-five set changes in 0.00% of windows. Visual standardized mean shift: seen 0.177, held 0.598. GPS shift: seen 0.137, held 0.600.
- S38: visual corrections help 20 windows and harm 97; top-five set changes in 20.53% of windows. Visual standardized mean shift: seen 0.180, held 0.728. GPS shift: seen 0.255, held 0.198.
- S39: visual corrections help 38 windows and harm 191; top-five set changes in 49.36% of windows. Visual standardized mean shift: seen 0.196, held 0.576. GPS shift: seen 0.273, held 0.260.

Next candidate: compare image-feature dropout/consistency regularization against the unchanged model using inner scenario splits within each training fold. Choose settings without excluded-scenario labels. Then freeze choices and run outer transfer evaluation. Because these scenarios have already been explored, retain the development-evidence limitation. No post-hoc per-scenario fallback from held labels.

Input shifts are descriptive, not causal attribution. Overlapping windows are correlated; counts are not independent statistical trials. Final calibration/test remain reserved.