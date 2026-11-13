# Label-efficient recognition of Chinese instruments

ELEC5305 final phase | Yuxuan Yang | SID 550405308

## Completed experiment

This folder contains the completed MATLAB-led experiment, source code, saved features,
held-out classifiers, result tables and figures. The final experiment was run on Windows
with MATLAB R2026a Update 4 and Python 3.11.9. All nine MATLAB unit tests passed on that
installation. Both feature branches completed 150 MATLAB classifier fits in total.
The saved results are from that run, not old Python predictions.

The question is: how does the number of labelled source recordings affect recognition,
and can frozen MERT features reach comparable performance with fewer labelled recordings
than classical timbre features?

This is a MATLAB-led hybrid, not an entirely MATLAB implementation. MATLAB extracts
classical features, trains both classifiers, evaluates results and draws the figures.
Python is used only for the frozen MERT encoder. Earlier Stage 2 work remains a separate
experiment. This folder does not replace that history.

## Main results

Values below are mean segment-level macro-F1 across the tested configurations.

| Labelled recordings per instrument | Classical features | Frozen MERT |
|---:|---:|---:|
| 1 | 86.27% | 93.05% |
| 2 | 91.02% | 96.92% |
| 3 | 93.66% | 98.35% |
| 4 | 95.32% | 98.62% |

In this experiment, MERT with two recordings per instrument exceeded the classical method
with four. This does not guarantee the same saving on new data. Classical extraction took
47.38 seconds for all 988 segments. MERT extraction took 655.97 seconds, about 13.8 times
longer. MERT model loading is excluded from this batch comparison. These are measured
CPU times on the original machine, not a hardware-independent speed claim.

At the full budget, both methods correctly classified all 55 recordings after segment
aggregation. This ceiling hides segment errors. Standard deviations and paired counts
are in [the results summary](results/MATLAB_RESULTS_SUMMARY.md). The configurations
overlap, so SD is descriptive, not an independent confidence interval.

Main figure: [learning curves](figures/matlab_learning_curves.png).
Numerical summaries: [summary CSV](results/matlab_summary.csv).

## Included files

| Location | Contents |
|---|---|
| Root .m files and +ti/ | MATLAB pipeline and helper functions |
| python/ | Frozen MERT bridge, requirements and transport tests |
| tests/ | Nine MATLAB unit tests |
| metadata/data_manifest.csv | Fixed manifest of 988 five-second segments |
| data/ | Computed 64-D classical and 768-D MERT feature caches |
| models/heldout_models.mat | MATLAB classifiers for both branches and five held-out folds |
| results/ | Run metrics, predictions, audits and acoustic analysis |
| figures/ | Learning curves, confusion matrices and acoustic examples |

The classifiers are small downstream models, not the pretrained MERT weights.
Raw WAVs, Python environments, encoder weights and local machine settings are excluded.
The report and demonstration video are separate submission items. The optional MATLAB
MFCC-KNN adaptation is supplied as code; its output is not claimed as part of the
completed 150-fit comparison.

## Requirements

- Verified environment: MATLAB R2026a Update 4 and Python 3.11.9.
- Audio Toolbox, Signal Processing Toolbox, and Statistics and Machine Learning Toolbox.
- Python packages in python/requirements.txt; the run used Transformers 4.46.3.
- The official 55 ChMusic WAV files for audio checking or demonstration.
- Internet access for the first MERT download when running the encoder.

Other MATLAB releases have not been verified. Obtain audio from the
[official ChMusic repository](https://github.com/HaoranWeiUTD/ChMusic).
Flat filenames 1.1.wav through 11.5.wav are supported, as are instrument-number
subfolders. Raw recordings are not redistributed here.

The bridge loads upstream model code with trust_remote_code=True at revision
12af15fef9d0ac838c3f475bfbbf26d2060dd4f5. Review that code if required by your
security policy. Several package requirements use version ranges; they are not a
complete environment lockfile.

## Windows setup

1. Extract this folder. In MATLAB, set Current Folder to matlab_final.
2. In Command Prompt opened in the same folder, create a dedicated Python environment:

```bat
py -3.11 -m venv .venv_mert
.venv_mert\Scripts\python.exe -m pip install -r python\requirements.txt
```

3. In MATLAB, replace the audio path with the folder containing the 55 WAV files:

```matlab
pythonExe = fullfile(pwd,'.venv_mert','Scripts','python.exe');
setup_project('D:\ChMusic\Musics',pythonExe);
check_inputs;
testResults = runtests('tests');
disp(testResults);
assert(all([testResults.Passed]),'Tests failed.');
```

setup_project creates local_settings.mat for this machine. Do not upload it.
check_inputs checks 55 sources and all 988 declared windows. It does not append extra
tails, preserving the Stage 2 evaluation set.

If mfcc reports too many input arguments, run `which mfcc -all`. An older course
toolbox may shadow MATLAB's Audio Toolbox function. Remove only that conflicting
folder from the current search path using rmpath, run `clear mfcc`, and repeat
the tests. Do not delete the course toolbox or reset all MATLAB paths.

## Inspect results without rerunning extraction

Open the CSV and PNG files directly. To regenerate tables and figures from the saved
outputs/features, run:

```matlab
addpath(pwd);
make_results;
write_result_summary;
```

This is not a fresh end-to-end run. It reuses the saved outputs. No rerun is needed
just to view the supplied results.
Regenerating acoustic example figures with analyze_acoustics also requires the WAVs.

## Reproduce the full experiment

After setup and successful tests, execute:

```matlab
check_inputs;
extract_classical;
prepare_mert;
run_comparison;
make_results;
analyze_acoustics;
write_result_summary;
```

Alternatively, run_all runs the tests and this full sequence. It also writes a local
log and completion marker. These two extra files are not required for the step-by-step
workflow used for the supplied results. A rerun writes result files and models in this
folder, so keep a copy of the submitted results first. CPU MERT extraction can take
several minutes; first-time download and loading add time.

The experiment uses 11 classes and five recording-wise folds. Each test fold holds out
one source per instrument. For budgets 1-4, all aligned subsets of remaining recording
numbers are tested: 20, 30, 20 and 5 configurations per branch. Segments from a held-out
recording do not enter fitting or standardisation.

Both branches use the same MATLAB one-vs-rest L2 logistic classifier (C=1),
training-only standardisation and source-equal weighting. Classical features contain
20 MFCCs and 12 other frame descriptors, summarised by mean and population SD (64 values).
MERT uses its final hidden layer and temporal mean pooling (768 values). The seed is
550405308. See +ti/config.m, +ti/features.m and +ti/train.m for full definitions.

MATLAB MFCCs, the explicit spectral descriptors and the one-vs-rest classifier differ
from the earlier Python implementation. Final-stage numbers are not an exact numerical
translation of Stage 2. No old prediction CSVs were used for this MATLAB comparison.

## Short audio demonstration

After setup, run:

```matlab
play_segment('dizi_r1_s004');
demo_instrument('dizi_r1_s004','live');
```

The demo reads the real five-second input and recomputes both representations.
Both MATLAB classifiers exclude source fold 1 from training. In the verified run,
the classical method predicted Yangqin and MERT predicted the correct class, Dizi.
The figure shows waveform, spectrogram and classifier scores. These scores are not
calibrated confidence. play_segment plays audio only; it does not open a player.

For an alternative that does not run the encoder, use:

```matlab
demo_instrument('dizi_r1_s004','cached');
```

Cached mode computes a fresh classical vector and both MATLAB predictions, but loads
the saved MERT embedding. It does not rerun the encoder and is visibly labelled cached.
The demo accepts manifest segment IDs, not arbitrary audio uploads.

## Optional benchmark adaptation

run_mfcc_knn supplies a separate MATLAB adaptation of the published MFCC-KNN setup:
20 frame MFCCs, K=15, recordings 1-4 for training, and recording 5 for testing.
It is not called by run_all and can be slow. No MATLAB MFCC-KNN score is supplied
or assumed to reproduce the published result. Retain the original Stage 2 benchmark
and report in the existing repository as the historical reproduction.

## Scope and limits

There are 55 source recordings, not 988 independent performances. Recording-wise
separation does not establish performer, session or device independence. This is
closed-set solo-instrument classification, not mixture or open-set recognition.
The experiment does not establish whether ChMusic appeared in pretraining and does
not attribute performance to cultural bias. Acoustic examples describe observed
sounds and errors; they do not prove MERT's internal causal mechanism.

## Submission and upload

Add this complete folder as matlab_final/ in the existing project repository.
Preserve earlier Stage 2 files and reports. Do not replace the repository root README
with this subfolder README; add a link to this file instead. Upload the extracted
folder, not just the ZIP, so the code is browsable. Retain .gitignore and +ti/.

Submit the final report and video (or permitted links) separately according to the
assignment page. Link the report to the actual project repository. No final edited
report or video is included in this code-and-results package.

## References

- Gong et al. (2021), [ChMusic](https://arxiv.org/abs/2108.08470).
- Li et al. (2024), [MERT](https://arxiv.org/abs/2306.00107).
- [Pinned MERT model](https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5).
- MathWorks, [MFCC documentation](https://www.mathworks.com/help/audio/ref/mfcc.html).
- MathWorks, [linear classifier documentation](https://www.mathworks.com/help/stats/fitclinear.html).
