# Penalty and convergence study

Geometry: 78.56%. Twelve-epoch, seed-42 exploratory validation.

- Stage 135: penalty=0.1, images=False: 78.81%, best epoch 3.
- Stage 136: penalty=0.1, images=True: 78.55%, best epoch 1.
- Stage 137: penalty=0.0, images=False: 78.56%, best epoch 1.
- Stage 138: penalty=0.0, images=True: 79.16%, best epoch 5.

Replicate the qualifying matched pair across seeds before drawing conclusions.

GPS and image pairs share the same budget, regularization and objective. Test/calibration remain reserved. Repeated validation comparisons are exploratory, not confirmatory evidence.