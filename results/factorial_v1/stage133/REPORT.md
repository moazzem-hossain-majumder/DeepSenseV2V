# Controlled objective/constraint factorial

Geometry: 78.56%. Single-seed, three-epoch exploratory validation.

- Stage 125: bounded=True, objective=soft, images=False: 78.56%, best epoch 3.
- Stage 126: bounded=True, objective=soft, images=True: 78.56%, best epoch 3.
- Stage 127: bounded=True, objective=hard, images=False: 78.56%, best epoch 3.
- Stage 128: bounded=True, objective=hard, images=True: 78.56%, best epoch 3.
- Stage 129: bounded=False, objective=soft, images=False: 78.56%, best epoch 3.
- Stage 130: bounded=False, objective=soft, images=True: 78.55%, best epoch 1.
- Stage 131: bounded=False, objective=hard, images=False: 78.56%, best epoch 3.
- Stage 132: bounded=False, objective=hard, images=True: 78.55%, best epoch 1.

No pair meets the preliminary gain screen. Review training curves and per-scenario effects; consider a separately controlled resolution/feature test, not automatic test-set evaluation.

Identical regularization, shortlist and features within all pairs. Head arithmetic is FP32. Metrics above K=20 are not shortlist-contained claims. All test/calibration data remain reserved.