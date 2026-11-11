# Full synthetic reproduction

Run these commands from the repository root.
Use a fresh output directory for each experiment; the examples below use `outputs/full/`.
These are the full experiment commands, not tasks automatically launched by installation or tests.

Python 3.11 and dependency versions are pinned.
Preparation, analysis and plotting use `pyproject.toml` and `uv.lock`; model scripts use separate locked environments to preserve their TensorFlow, Kriging and TabPFN dependencies.
Numerical workers use one CPU thread.

Keeping all data and outputs from this guide requires approximately 210 GB, estimated from array sizes: about 21 GB for downloaded and prepared data, 41 GB for baseline runs, 139 GB for CPT-count runs and 12 GB for fine-tuning and its evaluation.
The estimate excludes installed environments, package caches and temporary files; compressed validation outputs and training states can change the total.
Use new output paths for preparation, analysis and plotting; completed prediction runs can be reused when their inputs, settings and output checksums match.

## 1. Original sections and exact input placements

The data are from [SchemaGAN's Zenodo deposit](https://doi.org/10.5281/zenodo.13143431).
The archive is approximately 5.19 GB and the 24,000 extracted CSVs occupy approximately 12.33 GB.
All raw-file SHA-256 checksums are listed in `configs/synthetic_cpt/raw_sections.csv`.

```bash
uv run --locked scripts/download_data.py
uv run --locked scripts/prepare_synthetic_cpt.py --config configs/synthetic_cpt/baseline6_test.json
uv run --locked scripts/prepare_synthetic_cpt.py --config configs/synthetic_cpt/baseline6_train.json
uv run --locked scripts/prepare_synthetic_cpt.py --config configs/synthetic_cpt/baseline6_vali.json
uv run --locked scripts/prepare_layouts.py counts --output data/processed/counts
uv run --locked scripts/prepare_layouts.py fresh --output data/processed/fresh
```

An existing original archive can be supplied with `download_data.py --archive /path/to/data.zip`.
The archive and extracted files are verified before use.
To run only test-set inference, extract `--split test` and omit the training/validation preparation commands.

The baseline sampler resets NumPy's MT19937 `RandomState` to 20232023 at the start of each split and visits files in lexicographic relative-path order.
It consumes depth draws for every grid column, including unselected columns.
Replacing this with `default_rng`, changing the path order, or reseeding per section changes the inputs.
The first 4,000 training placements and the validation/test placements consequently coincide at corresponding positions, although their target sections differ.
The preparation scripts regenerate the placements from these settings; the full placement lists are not bundled.

The fresh test inputs use seed 20261022 and the same sampler, with no placement repeated from training, validation or baseline test inputs.
`prepare_layouts.py fresh` verifies this independently of the training targets.

For CPT counts, the original six columns are randomly ordered, then unused columns are appended to make a nested sequence of 48 columns.
The evaluation mask is the complement of that section's 48-column mask.
Each test section has one such sequence.

## 2. Checkpoints

```bash
uv run --locked scripts/download_models.py schemagan
uv run --locked scripts/download_models.py tabpfn
```

Both commands check the exact file checksum used in the manuscript.
TabPFN's default checkpoint, pinned download revision, software versions and inference settings are in `configs/tabpfn.json`.

## 3. Baseline predictions and prediction intervals

```bash
uv run --locked scripts/run_experiment.py baseline --output outputs/full
uv run --locked scripts/analyse_runs.py --baseline outputs/full/baseline --distance --output outputs/full/analysis
```

The first command runs all four methods and the 100 SchemaGAN seeds sequentially.
It saves TabPFN medians and means, Kriging means, nearest-neighbour predictions, and the individual SchemaGAN draws.
It also computes TabPFN and Kriging intervals, and aggregates SchemaGAN's draws into predictive means, medians and normal-approximation intervals.
`--methods tabpfn` or another subset can be used to run individual methods.

SchemaGAN is loaded into a fresh process/model for each seed and called once per section in the original order.
Batching sections or starting at a later section without replaying previous calls changes its stochastic sequence.
For the baseline table, errors are calculated per draw and per section, then averaged over seeds within each section.
The statistical sample remains 4,000 sections.

The baseline point errors use all pixels.
The runs also save unobserved-pixel errors for the additional checks.
The interval table uses unobserved pixels and gives every section equal weight.
SchemaGAN intervals use the sample standard deviation of the 100 draws, without division by the square root of 100.
Kriging intervals include the fitted WhiteKernel variance.
No method's outputs are clipped or overwritten at input pixels.

The distance analysis uses Euclidean distance in pixel coordinates to the nearest observed pixel, with a separate distance-zero bin and positive bins `(0,4]`, `(4,8]`, etc.
It averages errors within each section and bin before averaging contributing sections.
The plotted curve includes bins with at least 200 contributing sections.
Its confidence intervals describe variation across sections, not pixels or draws.

Expected baseline means, rounded to manuscript precision:

| Method | MAE | RMSE |
|---|---:|---:|
| TabPFN | 0.123 | 0.264 |
| SchemaGAN | 0.128 | 0.260 |
| Kriging | 0.191 | 0.299 |
| Nearest neighbour | 0.170 | 0.365 |

The analysis command calculates unrounded values and paired comparisons from your run and writes them to `table1.csv`.

## 4. CPT count

```bash
uv run --locked scripts/run_experiment.py counts --output outputs/full/counts
uv run --locked scripts/analyse_runs.py --baseline outputs/full/baseline --counts outputs/full/counts --output outputs/full/count_analysis
```

Only TabPFN and SchemaGAN are included in this experiment.
The resulting `n_3` through `n_48` directories contain the per-section results and individual SchemaGAN draws.
The manuscript mean MAEs are approximately `0.208, 0.125, 0.088, 0.059, 0.036` for TabPFN and `0.377, 0.130, 0.231, 0.368, 0.489` for SchemaGAN.

## 5. Fine-tuning and checkpoint selection

```bash
uv run --locked scripts/train_finetune.py --config configs/finetune_seed0.json --output-dir outputs/full/training_seed0
uv run --locked scripts/train_finetune.py --config configs/finetune_seed1.json --output-dir outputs/full/training_seed1
```

Before training, the command automatically checks all four preprocessing families, isolation of query targets, checkpoint serialization and resumption.
The full run uses all 16,000 training sections for five epochs and all 4,000 validation sections after each epoch.
The selected checkpoint is written to `selection.json`.
Only validation MAE is used for selection; the test targets are not loaded by the training command.
Both original runs selected epoch 5.

Training settings follow Appendix A of the manuscript and are recorded in `configs/finetune_seed0.json` and `configs/finetune_seed1.json`.
The implementation in `finetune_core.py` preserves the target-transform alignment and per-step random-seed rules.
Validation uses the eight-member inference settings and predictive medians.

The recorded seed-1 run, including validation, took 18.1 hours on one NVIDIA RTX 6000 Ada Generation GPU and one CPU thread, with peak allocated GPU memory of 3.21 GiB.
This is a recorded measurement, not a runtime guarantee for another machine.

To resume an interrupted training run, use the same command with `--resume`; its settings, code and saved training state must match.

## 6. Test accuracy and fresh placements after fine-tuning

The following commands assume the reproduced selection is epoch 5; otherwise use the checkpoint named by each run's `selection.json`.

```bash
uv run --locked scripts/predict_tabpfn.py --output outputs/full/finetuning/baseline/base
uv run --locked scripts/predict_tabpfn.py --checkpoint outputs/full/training_seed0/checkpoints/epoch_5.ckpt --output outputs/full/finetuning/baseline/ft0
uv run --locked scripts/predict_tabpfn.py --checkpoint outputs/full/training_seed1/checkpoints/epoch_5.ckpt --output outputs/full/finetuning/baseline/ft1
uv run --locked scripts/predict_tabpfn.py --data data/processed/fresh --output outputs/full/finetuning/fresh/base
uv run --locked scripts/predict_tabpfn.py --data data/processed/fresh --checkpoint outputs/full/training_seed0/checkpoints/epoch_5.ckpt --output outputs/full/finetuning/fresh/ft0
uv run --locked scripts/predict_tabpfn.py --data data/processed/fresh --checkpoint outputs/full/training_seed1/checkpoints/epoch_5.ckpt --output outputs/full/finetuning/fresh/ft1
uv run --locked scripts/compare_finetuning.py --baseline-data data/processed/test --fresh-data data/processed/fresh --baseline-runs outputs/full/finetuning/baseline --fresh-runs outputs/full/finetuning/fresh --output outputs/full/finetuning/comparison
```

The comparison includes the pixels unobserved under both baseline and fresh placements, keeping evaluation pixels fixed when comparing the placements.
For optimisation seed 0, the reported all-pixel MAE/RMSE reductions are 22.1%/16.0% for baseline inputs and 22.6%/16.5% for fresh inputs.

## 7. Plot a computed section

```bash
uv run --locked scripts/plot_prediction.py --run outputs/full/baseline/tabpfn --index 0 --output outputs/full/tabpfn_section0.png
```

This plots the inputs, target and prediction from a completed inference run.
Use `--index` to select a section in lexicographic file-path order.
The same command accepts the Kriging and nearest-neighbour run directories, individual SchemaGAN seed directories, or the fine-tuned prediction runs above.
For a fresh-input run, also pass `--data data/processed/fresh`.
The small plotter uses baseline or fresh inputs; it does not draw the CPT-count curves or the full manuscript figures.

## 8. Spatial variation and output-summary comparison

Use the completed baseline runs from Section 3, including all 100 SchemaGAN draws and their aggregated `medians.npy`:

```bash
uv run --locked scripts/analyse_spatial_variation.py --baseline outputs/full/baseline --output outputs/full/spatial_variation
```

This writes the 96 values of Table B.1 to `table_b1.csv` and a rounded table to `table_b1.md`.
It also compares individually scored SchemaGAN reconstructions with their pixelwise median, as reported in Section 4.4 of the manuscript.
The 24 comparisons are in `output_summary_comparison.csv`, and `findings.json` reports the largest changes in the medians across sections.
`per_section.npz` contains the target descriptors and section-level ratios; `run.json` records checked input hashes, section and draw counts, code hashes and calculation settings.

The command uses one CPU thread and processes 16 sections per block by default; no inference or training is performed.
It verifies prediction-file checksums and the common baseline inputs, including the identities of the draws used to construct the cached pixelwise median.
The cached median is also reconstructed from individual draws at up to four section indices.
For a small check, add `--limit 4`; this reads and verifies the same files but scores only their first four sections.
A full-size analysis requires 4,000 test sections and seeds 0--99; smaller runs are marked `full_analysis: false` and their results are not the manuscript table.
Choose a new output directory for each analysis.

The [spatial-variation notes](spatial_variation.md) describe the aggregation order, manuscript reference values and source provenance.

## Reproducibility limits

Small baseline runs were compared with the original manuscript predictions for all four methods, and CPT-count runs for TabPFN and SchemaGAN at all five counts.
The protocol tests and fine-tuning smoke check were run, and a fresh clone was checked from preparation through prediction and plotting.
The full training and inference campaigns were not rerun when preparing this repository.
GPU architecture and numerical libraries can affect floating-point results; use the pinned environments and checkpoint checksums.
