# Diagnostic cycle: Stages 62–71

Front-image filename offsets (row minus filename, seconds): {'36': -0.0002029999959631823, '37': -0.9001560000033351, '38': 1.4997179999991204, '39': -1.0996179999929154}.
Filename offsets alone do not prove an alignment bug. No image rows were shifted.

Scenario 38 raw-checkpoint intervention with highest exploratory success: gray.
Gray and shuffled inputs are distribution shifts; their results diagnose sensitivity, not the performance of a separately trained GPS-only model.

- Checkpoint 49, raw corrections (fallback disabled for diagnosis): original: 79.01%, gray: 79.01%, shuffled: 79.01%.
- Checkpoint 50, raw corrections (fallback disabled for diagnosis): original: 78.98%, gray: 78.86%, shuffled: 78.98%.
- Checkpoint 51, raw corrections (fallback disabled for diagnosis): original: 79.02%, gray: 79.02%, shuffled: 79.02%.
- Checkpoint 53, raw corrections (fallback disabled for diagnosis): original: 86.00%, gray: 85.98%, shuffled: 85.29%.
- Checkpoint 55, raw corrections (fallback disabled for diagnosis): original: 94.26%, gray: 93.99%, shuffled: 94.24%.
- Checkpoint 57, raw corrections (fallback disabled for diagnosis): original: 66.16%, gray: 66.25%, shuffled: 66.22%.
- Checkpoint 59, raw corrections (fallback disabled for diagnosis): original: 68.51%, gray: 68.51%, shuffled: 68.51%.

Resolve image filename clock conventions and verify physical synchronization before retraining with any shifted images.

No calibration/test evaluation, publication claim, or automatic timestamp correction is authorized by these diagnostic results. Further training requires a justified, recorded protocol; repeatedly inspecting validation does not provide independent confirmation.