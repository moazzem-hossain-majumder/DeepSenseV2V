# Frozen-checkpoint validation diagnostics

Validation data only. Test/calibration reserved. Checkpoint selection used validation top-1.
Shortlist results use offline measured powers; they are not an online probing-cost claim.

- B1 (checkpoint epoch 10): Top-1 10.32%, Top-5 28.60%, mean top-1 power loss 14.18 dB; within 3 dB with 5 candidates: 59.91%.
- P1 (checkpoint epoch 1): Top-1 11.36%, Top-5 31.09%, mean top-1 power loss 13.72 dB; within 3 dB with 5 candidates: 42.91%.
- P3 (checkpoint epoch 6): Top-1 15.63%, Top-5 43.03%, mean top-1 power loss 11.29 dB; within 3 dB with 5 candidates: 57.17%.
- P3_noGPS (checkpoint epoch 6): Top-1 10.02%, Top-5 26.56%, mean top-1 power loss 14.79 dB; within 3 dB with 5 candidates: 38.86%.
- P3_noRGB (checkpoint epoch 6): Top-1 2.58%, Top-5 9.14%, mean top-1 power loss 13.87 dB; within 3 dB with 5 candidates: 27.05%.

Zeroed normalized images correspond to ImageNet mean pixels. Zeroed scaled GPS is a sensitivity intervention, not a physical missing-GPS model. These distribution shifts do not replace retrained ablations.