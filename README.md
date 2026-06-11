# Minimal Public Release for Scenario 4

This package provides a **minimal public-release implementation** of the core personalised multi-horizon prediction experiment from the manuscript.

## Scope

This release is intentionally limited to the core **Scenario 4** workflow:

- single-subject glucose prediction
- leave-one-day-out cross-validation (LODOCV)
- 15 / 30 / 45 / 60 minute prediction horizons
- optimisation of model parameters
- RMSE-based summary outputs
- Table 4-style aggregation from precomputed subject results

This package is **not** a full reproduction of every figure, table, or internal analysis used during manuscript development. The demo files are intended to provide a portable minimal reproduction of the Scenario 4 workflow and the manuscript-style Table 4 summary.

## Files

- `scenario4_min_public.py` — main minimal reproducible Scenario 4 script
- `demo.py` — Table 4-style demo runner; can either re-run models or aggregate saved results from `data/`
- `summarize_parameters.py` — extracts fold-specific fitted parameters from saved detail files
- `run_all_subjects.py` — optional batch runner for multiple subject CSV files
- `requirements.txt` — Python dependencies
- `data/` — optional precomputed subject-level or fold-level result files, if included in the release

## Expected Input Format for Re-running Models

Each subject CSV file must contain the following columns:

- `Timestamp`
- `Libre GL`
- `Meal Type`
- `GI`
- `Carbs`

Example row:

```text
Timestamp,Libre GL,Meal Type,GI,Carbs
2025-01-01 08:00:00,118.4,Breakfast,55,45
```

## Public Dataset

The study uses the **CGMacros Dataset v1.0.0** from PhysioNet, which is publicly available.

Please note that this public code expects **subject-level CSV files already formatted** with the columns above. Raw PhysioNet files may require preprocessing before use.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Synthetic Demo

Run a small synthetic example:

```bash
python scenario4_min_public.py --demo --output_dir demo_outputs
```

## Reconstruct Table 4 from Saved Results

If the repository includes precomputed subject results in `data/`, Table 4 can be regenerated without re-optimising the models:

```bash
python demo.py \
  --precomputed_dir data \
  --output_dir table4_from_data
```

This produces:

- `table4_subject_level_results.csv`
- `table4_summary.csv`
- `table4_paper_ready.csv`

If fold-level detail files are available, it also produces:

- `table4_fold_level_results.csv`

## Summarise Fold-Specific Parameters

LODOCV re-estimates parameters within each held-out-day fold. Therefore, each subject has fold-specific optimal parameters rather than one unique parameter set. To extract these parameters from saved detail files:

```bash
python summarize_parameters.py \
  --precomputed_dir data \
  --output_dir table4_from_data
```

This produces:

- `parameter_fold_level.csv`
- `parameter_subject_summary.csv`

## Run on One Subject

```bash
python scenario4_min_public.py \
  --input_csv path/to/subject_049.csv \
  --output_dir results_subject_049
```

## Batch Run Across Multiple Subjects

```bash
python demo.py \
  --input_dir path/to/formatted_subject_csvs \
  --pattern "*.csv" \
  --output_dir table4_rerun_outputs
```

## Notes

- This release focuses on **clarity, portability, and minimal reproducibility**.
- Saved `data/` results are the recommended path for reproducing manuscript-style Table 4 values quickly.
- Re-running optimisation may produce numerical differences if preprocessing, subject formatting, optimiser settings, or dependency versions differ from the original internal pipeline.
- LODOCV parameters are fold-specific. Subject-level parameter summaries report the mean and variability across held-out-day folds.
