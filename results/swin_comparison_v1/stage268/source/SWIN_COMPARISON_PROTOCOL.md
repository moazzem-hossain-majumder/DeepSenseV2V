# Paired shifted-window camera comparison

## Research question

Does the incremental contribution of front and rear RGB over a matched GPS network change between represented and excluded driving scenarios when the model can select any of 256 beams? This is an exploratory development comparison on previously examined scenarios. The study freezes architecture, runs and analysis before observation of its validation outputs. All 30 training fits belong to this single finite study.

## Closest system and adaptation

Fabiani et al. use five historical observations, GPS and two 224-pixel camera views, EfficientNet-B3, multiscale residual fusion, four attention heads, eight shifted-window layers and a full-codebook classifier. Their reported gains use a linear GPS comparator. Their reported exact-beam top-five accuracy and average power loss measure different outcomes from near-optimal top-five success. The public paper and repository searches did not identify a verified author implementation. [Paper and architecture](https://repository.kaust.edu.sa/server/api/core/bitstreams/712d9981-ea7a-44e5-870b-c36e6129acb2/content)

Our adaptation uses fixed ImageNet EfficientNet-B3 features at two depths with 2-by-2 spatial pooling per view. Five timestamps form one axis; eight spatial tokens and one GPS token form the other. Two stages each have four 32-dimensional blocks, four heads and alternating unshifted and shifted windows. This replaces the published multiscale feedback into the image encoder and its 25-by-20 fusion representation. We use the project's nine GPS features, train-only scaling, cosine-decayed AdamW, no EMA and no image augmentation. No published score reproduction is implied. Both arms have the same trainable modules and initial seed. The GPS arm receives zero image contribution at both stages. Analytic geometry uses the existing fold-specific fit.

## Frozen design and interpretation

Use seeds 42, 7 and 13. For each seed fit a GPS and a GPS-plus-RGB model on pooled data and each of the four leave-one-scenario-out partitions. Each fit uses batch size 96, twelve epochs, cross entropy over 256 beams and a starting learning rate of 0.0004. Select the highest macro top-five within-3-dB success on included-scenario validation, using the latest epoch in a tie. Evaluate identical validation windows in the two protocols. Prior calibration and final test reservations continue.

Primary endpoint is the equal-scenario-weighted difference in top-five within-3-dB success between RGB and matched GPS. Primary contrast is its change from pooled to excluded-scenario validation. Report both arms versus each other, RGB versus geometry, scenario-specific differences and scenario-stratified paired block percentile intervals with 4,000 draws after averaging seeds within each window. Secondary endpoints are exact-beam top-five accuracy, best-of-five power loss and top-one power loss. A sign change alone does not establish a general effect. Discuss uncertainty, scenario heterogeneity and overlap with earlier development. The established development gate still applies. Supervisor, retries, epoch checkpoints and the existing progress display use the project's sequential queue.
