# Native-image resolution comparison

Review paired resolution effect before seed replication or transfer. Both arms omit brightness augmentation and use cached frozen features, so compare them directly, not as an isolated resolution change from older training.

- 96 pixels per view: 79.23%, selected epoch 5.
- 192 pixels per view: 79.54%, selected epoch 5.
192 minus 96: +0.309 percentage points; paired block interval [-0.002889861178631232, 0.007762661616166083] (fraction units).

Single-seed exploratory comparison; no publication or SOTA claim.