"""Generate Table 4-style Scenario 4 results.

This demo runs the minimal public Scenario 4 implementation across all formatted
subject CSV files and aggregates the output into paper-style cohort summaries.

Expected input CSV columns:
    Timestamp, Libre GL, Meal Type, GI, Carbs

Example:
    python demo.py \
        --input_dir path/to/formatted_subject_csvs \
        --output_dir table4_demo_outputs

Outputs:
    table4_fold_level_results.csv
    table4_subject_level_results.csv
    table4_summary.csv
    table4_paper_ready.csv

Notes:
    The public release is intentionally minimal. Numerical values may differ from
    the manuscript Table 4 if preprocessing, subject formatting, optimizer
    settings, or dependency versions differ from the original internal pipeline.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from scenario4_min_public import run_scenario4


RMSE_COLUMNS = ["RMSE_OriginalBMM", "RMSE_BaselineMeal", "RMSE_Extended"]
MODEL_LABELS = {
    "RMSE_OriginalBMM": "Baseline_OriginalBMM",
    "RMSE_BaselineMeal": "BaselineMeal_CarbsOnly",
    "RMSE_Extended": "Proposed_Extended",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all formatted subject CSVs and generate Table 4-style "
            "multi-horizon RMSE summaries."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing formatted subject-level CSV files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern used to find subject CSV files. Default: *.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="table4_demo_outputs",
        help="Directory for per-subject outputs and Table 4 summaries.",
    )
    parser.add_argument(
        "--prediction_horizons",
        type=int,
        nargs="+",
        default=[15, 30, 45, 60],
        help="Prediction horizons in minutes. Default: 15 30 45 60",
    )
    parser.add_argument(
        "--subject_limit",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "Reuse existing per-subject scenario4_detailed_results.csv files "
            "when available."
        ),
    )
    return parser.parse_args()


def subject_id_from_file(path: Path) -> str:
    stem = path.stem
    for prefix in ("CGMacros-", "subject_", "Subject_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def find_subject_files(input_dir: Path, pattern: str, subject_limit: int = None) -> List[Path]:
    files = sorted(input_dir.glob(pattern))
    if subject_limit is not None:
        files = files[:subject_limit]
    if not files:
        raise FileNotFoundError(f"No files matched pattern {pattern!r} in {input_dir}")
    return files


def run_or_load_subject(
    file_path: Path,
    subject_out: Path,
    prediction_horizons: Iterable[int],
    skip_existing: bool,
) -> pd.DataFrame:
    detail_path = subject_out / "scenario4_detailed_results.csv"
    if skip_existing and detail_path.exists():
        return pd.read_csv(detail_path)

    subject_out.mkdir(parents=True, exist_ok=True)
    return run_scenario4(
        input_csv=str(file_path),
        output_dir=str(subject_out),
        prediction_horizons=list(prediction_horizons),
    )


def build_subject_level_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    expected = {"Subject", "PH", "Test Day", *RMSE_COLUMNS}
    missing = expected.difference(fold_df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in fold-level results: {sorted(missing)}")

    aggregations = {col: ["mean", "std", "count"] for col in RMSE_COLUMNS}
    subject_level = fold_df.groupby(["Subject", "PH"], as_index=False).agg(aggregations)

    flattened_cols = []
    for col in subject_level.columns:
        if isinstance(col, tuple):
            base, stat = col
            if stat:
                flattened_cols.append(f"{MODEL_LABELS.get(base, base)}_{stat}")
            else:
                flattened_cols.append(base)
        else:
            flattened_cols.append(col)
    subject_level.columns = flattened_cols

    # The count is identical across model columns, so keep one explicit fold count.
    first_count_col = f"{MODEL_LABELS['RMSE_Extended']}_count"
    subject_level = subject_level.rename(columns={first_count_col: "N_lodo_folds"})
    redundant_count_cols = [c for c in subject_level.columns if c.endswith("_count")]
    subject_level = subject_level.drop(columns=redundant_count_cols)

    return subject_level


def build_table4_summary(subject_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ph, group in subject_level.groupby("PH", sort=True):
        row = {
            "PH": int(ph),
            "N_subjects": int(group["Subject"].nunique()),
            "N_lodo_folds_total": int(group["N_lodo_folds"].sum()),
        }

        for rmse_col, label in MODEL_LABELS.items():
            subj_mean_col = f"{label}_mean"
            if subj_mean_col not in group:
                continue
            row[f"{label}_RMSE_mean"] = float(group[subj_mean_col].mean())
            row[f"{label}_RMSE_std_across_subjects"] = float(group[subj_mean_col].std(ddof=1))
            row[f"{label}_RMSE_weighted_mean_by_folds"] = float(
                np.average(group[subj_mean_col], weights=group["N_lodo_folds"])
            )

        if "Baseline_OriginalBMM_RMSE_mean" in row and "Proposed_Extended_RMSE_mean" in row:
            abs_gain = row["Baseline_OriginalBMM_RMSE_mean"] - row["Proposed_Extended_RMSE_mean"]
            row["Proposed_vs_OriginalBMM_abs_RMSE_reduction"] = float(abs_gain)
            row["Proposed_vs_OriginalBMM_relative_reduction_percent"] = float(
                100.0 * abs_gain / row["Baseline_OriginalBMM_RMSE_mean"]
            )

        if "BaselineMeal_CarbsOnly_RMSE_mean" in row and "Proposed_Extended_RMSE_mean" in row:
            abs_gain = row["BaselineMeal_CarbsOnly_RMSE_mean"] - row["Proposed_Extended_RMSE_mean"]
            row["Proposed_vs_BaselineMeal_abs_RMSE_reduction"] = float(abs_gain)
            row["Proposed_vs_BaselineMeal_relative_reduction_percent"] = float(
                100.0 * abs_gain / row["BaselineMeal_CarbsOnly_RMSE_mean"]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def format_mean_std(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "NA"
    if pd.isna(std_value):
        return f"{mean_value:.2f}"
    return f"{mean_value:.2f} +/- {std_value:.2f}"


def build_paper_ready_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        out = {
            "PH": int(row["PH"]),
            "N_subjects": int(row["N_subjects"]),
            "N_lodo_folds_total": int(row["N_lodo_folds_total"]),
        }
        for label in MODEL_LABELS.values():
            mean_col = f"{label}_RMSE_mean"
            std_col = f"{label}_RMSE_std_across_subjects"
            if mean_col in row.index:
                out[label] = format_mean_std(row[mean_col], row.get(std_col, np.nan))
        if "Proposed_vs_OriginalBMM_relative_reduction_percent" in row.index:
            out["Reduction_vs_OriginalBMM"] = f"{row['Proposed_vs_OriginalBMM_relative_reduction_percent']:.1f}%"
        if "Proposed_vs_BaselineMeal_relative_reduction_percent" in row.index:
            out["Reduction_vs_BaselineMeal"] = f"{row['Proposed_vs_BaselineMeal_relative_reduction_percent']:.1f}%"
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    per_subject_dir = output_dir / "per_subject"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_subject_dir.mkdir(parents=True, exist_ok=True)

    subject_files = find_subject_files(input_dir, args.pattern, args.subject_limit)
    print(f"Found {len(subject_files)} subject file(s) in {input_dir}")

    all_fold_results = []
    for idx, file_path in enumerate(subject_files, start=1):
        subject = subject_id_from_file(file_path)
        subject_out = per_subject_dir / subject
        print(f"[{idx}/{len(subject_files)}] Running subject {subject}: {file_path}")
        subject_df = run_or_load_subject(
            file_path=file_path,
            subject_out=subject_out,
            prediction_horizons=args.prediction_horizons,
            skip_existing=args.skip_existing,
        ).copy()
        subject_df.insert(0, "Subject", subject)
        all_fold_results.append(subject_df)

    fold_df = pd.concat(all_fold_results, ignore_index=True)
    fold_path = output_dir / "table4_fold_level_results.csv"
    fold_df.to_csv(fold_path, index=False)

    subject_level = build_subject_level_table(fold_df)
    subject_path = output_dir / "table4_subject_level_results.csv"
    subject_level.to_csv(subject_path, index=False)

    summary = build_table4_summary(subject_level)
    summary_path = output_dir / "table4_summary.csv"
    summary.to_csv(summary_path, index=False)

    paper_ready = build_paper_ready_table(summary)
    paper_ready_path = output_dir / "table4_paper_ready.csv"
    paper_ready.to_csv(paper_ready_path, index=False)

    print("\nTable 4-style summary:")
    print(paper_ready.to_string(index=False))
    print("\nSaved outputs:")
    print(f"  {fold_path}")
    print(f"  {subject_path}")
    print(f"  {summary_path}")
    print(f"  {paper_ready_path}")


if __name__ == "__main__":
    main()
