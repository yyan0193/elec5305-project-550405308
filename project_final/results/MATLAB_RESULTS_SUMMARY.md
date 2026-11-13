# MATLAB downstream results

Generated: 20261113T155828. MATLAB: 26.1.0.3312084 (R2026a) Update 4.

MERT embeddings are extracted in Python; classical features, fitting, evaluation and plots are MATLAB.

Classifier: identical one-vs-rest L2 logistic models for both representations; this differs from the Stage 2 multinomial classifier. Positive binary scores are row-normalised for aggregation, not calibrated confidence.

|Sources/class|Classical segment F1|MERT segment F1|Classical recording F1|MERT recording F1|
|---:|---:|---:|---:|---:|
|1|0.8627 +/- 0.0534|0.9305 +/- 0.0309|0.9636 +/- 0.0570|0.9879 +/- 0.0373|
|2|0.9102 +/- 0.0465|0.9692 +/- 0.0198|0.9960 +/- 0.0221|1.0000 +/- 0.0000|
|3|0.9366 +/- 0.0436|0.9835 +/- 0.0147|1.0000 +/- 0.0000|1.0000 +/- 0.0000|
|4|0.9532 +/- 0.0398|0.9862 +/- 0.0122|1.0000 +/- 0.0000|1.0000 +/- 0.0000|

SD describes overlapping configurations, not independent confidence intervals. No significance claim is made.

Budget 1: MERT minus classical mean segment macro-F1 = +0.0678; 17 wins, 0 ties, 3 losses.

Budget 2: MERT minus classical mean segment macro-F1 = +0.0590; 29 wins, 0 ties, 1 losses.

Budget 3: MERT minus classical mean segment macro-F1 = +0.0469; 19 wins, 0 ties, 1 losses.

Budget 4: MERT minus classical mean segment macro-F1 = +0.0330; 4 wins, 0 ties, 1 losses.

Label efficiency must be judged from these curves, not assumed from Stage 2. A better mean is not guaranteed savings on future data.

Only 55 source recordings; no claim of performer/session independence, cultural-bias causality, verified pretraining nonoverlap, mixture recognition or open-set recognition.
