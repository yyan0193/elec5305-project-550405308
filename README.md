# Label-Efficient Recognition of Traditional Chinese Instruments

**Student:** Yuxuan Yang

**SID:** 550405308

**Course:** ELEC5305

## Research question

How does the amount of labelled source audio affect traditional Chinese instrument recognition, and does a frozen music representation (MERT-v1-95M) require fewer labelled recordings than a compact, interpretable classical timbre representation?

The project covers all 11 instrument classes in ChMusic and treats each original recording—not each five-second excerpt—as an independent source. This prevents excerpts from the same recording appearing in both training and testing.

## Stage 2 deliverables

- Reproducible ChMusic data manifest: 55 source recordings and 988 non-overlapping five-second segments.
- Reproduction of the published MFCC + KNN reference split.
- A 64-dimensional classical representation combining MFCC statistics with interpretable timbral features.
- A frozen 768-dimensional MERT-v1-95M representation using temporal mean pooling of the final hidden layer.
- Source-aware five-fold evaluation with label budgets of 1, 2, 3, and 4 original recordings per instrument.
- Recording-level and segment-level accuracy, macro-F1, confusion matrices, timing, and targeted error analysis.

## Current results

| Experiment | Recording macro-F1 | Segment macro-F1 |
|---|---:|---:|
| Published-split MFCC + KNN reproduction | 1.0000 | 0.9379 |
| Classical representation, four labelled sources/class | 1.0000 | 0.9547 |
| MERT-v1-95M, four labelled sources/class | 1.0000 | 0.9822 |

The MFCC + KNN segment accuracy is **94.152%**, matching the published reference value of **94.15%** at two decimal places. At the smallest label budget, recording-level macro-F1 is **0.9636 ± 0.0692** for the classical representation and **0.9939 ± 0.0271** for MERT. These are interim results, not the final project conclusion.

Feature extraction on the current CPU environment took approximately 44.2 seconds for the classical representation and 353.6 seconds for MERT across 988 segments. MERT is 12 times larger in feature dimension and about 8.0 times slower to extract.

## Repository structure

```text
config/       Experiment configuration
docs/         Stage 2 progress report and upload notes
figures/      Generated learning curves and confusion matrices
metadata/     Source-aware recording audit and segment manifest
results/      Metrics, predictions, confusion matrices, and timings
src/          Reproducible Python pipeline
tests/        Leakage and pipeline checks
```

Raw ChMusic audio, generated feature caches, and downloaded model weights are deliberately excluded from Git.

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the official ChMusic dataset and place the extracted files at:

```text
data/raw/ChMusic/Musics/<instrument_id>/*.wav
```

The expected structure contains folders `1` through `11`, with five WAV recordings in each folder.

## Reproduce the pipeline

Run commands from the repository root:

```bash
python -m src.prepare_dataset
python -m src.reproduce_mfcc_knn
python -m src.extract_classical_features
python -m src.extract_mert_embeddings
python -m src.run_representation_experiments
python -m src.summarise_computational_cost
python -m src.analyze_errors
python -m pytest -q
```

The MERT step downloads `m-a-p/MERT-v1-95M` from Hugging Face on first use. Feature caches are created under `data/cache/` and reused by later commands.

## Evaluation design

For each of five folds, recording number 1–5 is held out for every instrument. Training uses combinations of the remaining recordings at each label budget:

| Budget | Combinations per fold | Runs over five folds |
|---:|---:|---:|
| 1 | 4 | 20 |
| 2 | 6 | 30 |
| 3 | 4 | 20 |
| 4 | 1 | 5 |

Both representations use the same standardised logistic-regression classifier. The comparison therefore focuses on representation quality rather than classifier choice. All segments belonging to an original recording remain together.

## Project documents

- [Stage 2 progress report](docs/feedback2_progress.md)
- [Project proposal](ELEC5305_Project_Proposal_Yuxuan_Yang_550405308.pdf)
- [GitHub Pages site](https://yyan0193.github.io/elec5305-project-550405308/)

## Data and model attribution

- ChMusic dataset and published MFCC-KNN baseline: [HaoranWeiUTD/ChMusic](https://github.com/HaoranWeiUTD/ChMusic)
- MERT-v1-95M pretrained model: [m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)
