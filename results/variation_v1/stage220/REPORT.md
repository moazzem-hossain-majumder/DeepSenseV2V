# How scenario exclusion changes the visual contribution

All 16,643 validation windows were paired by ID across both training protocols and arms. We averaged the three seed-specific success differences per window and weighted the four scenarios equally.

The visual advantage over matched GPS was +0.431 points in pooled validation and -0.286 points in transfer. The paired difference was +0.717 percentage points.
Scenario-stratified 95% percentile intervals from 4,000 paired draws were [-0.10468833998245762, 1.2080134763880566] points for recorded blocks and [0.06765137496106356, 1.5396992873203947] points for recorded sequence IDs.

- S36 contributed +0.344 points within that scenario. Omitting S36 gave +0.841 points across the other three.
- S37 contributed -0.075 points within that scenario. Omitting S37 gave +0.981 points across the other three.
- S38 contributed -0.149 points within that scenario. Omitting S38 gave +1.005 points across the other three.
- S39 contributed +2.747 points within that scenario. Omitting S39 gave +0.040 points across the other three.

Intervals condition on the four observed scenarios and selected checkpoints. Dependence between recorded sequences, limited cluster counts and prior model selection restrict inference to this development comparison.