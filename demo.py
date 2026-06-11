"""Generate Table 4-style Scenario 4 results.

This demo has two modes.

1. Re-run the minimal public Scenario 4 implementation from formatted subject CSVs:

    python demo.py \
        --input_dir path/to/formatted_subject_csvs \
        --output_dir table4_demo_outputs

2. Reconstruct Table 4 directly from precomputed subject result CSV files:

    python demo.py \
        --precomputed_dir data \
        --output_dir table4_from_data

The second mode is recommended for a minimal public reproduction of the manuscript
Table 4 when per-subject results and fitted parameters have already been saved.
It avoids re-optimising the model and simply aggregates the stored LODO results.

Expected formatted subject CSV columns for re-run mode:
    Timestamp, Libre GL, Meal Type, GI, Carbs

Recognised precomputed CSV formats:
    A. Fold-level files with columns such as
       Subject, Test Day, PH, RMSE_Extended, RMSE_BaselineMeal, RMSE_OriginalBMM
    B. Per-subject summary files with columns such as
       PH, RMSE_mean, RMSE_std, N_folds
       These are interpreted as Proposed_Extended by default.
    C. Already-labelled per-subject summary files with columns such as
       PH, Proposed_Extended_RMSE_mean, Baseline_OriginalBMM_RMSE_mean, ...

Outputs:
    table4_fold_level_results.csv      (when fold-level input is available)
    table4_subject_level_results.csv
    table4_summary.csv
    table4_paper_ready.csv
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from scenario4_min_public import run_scenario4


RMSE_COLUMNS = ["RMSE_OriginalBMM", "RMSE_BaselineMeal", "RMSE_Extended"]
MODEL_LABELS = {
    "RMSE_OriginalBMM": "Baseline_OriginalBMM",
    "RMSE_BaselineMeal": "BaselineMeal_CarbsOnly",
    "RMSE_Extended": "Proposed_Extended",
}
SUMMARY_MEAN_COLUMNS = [f"{label}_RMSE_mean" for label in MODEL_LABELS.values()]
GENERATED_OUTPUT_NAMES = {
    "table4_fold_level_results.csv",
    "table4_subject_level_results.csv",
    "table4_summary.csv",
    "table4_paper_ready.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Table 4-style multi-horizon RMSE summaries either by "
            "running the public demo pipeline or by aggregating precomputed results."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing formatted subject-level CSV files for re-run mode.",
    )
    parser.add_argument(
        "--precomputed_dir",
        type=str,
        default=None,
        help="Directory containing saved subject result CSV files for aggregation-only mode.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for subject CSVs in re-run mode. Default: *.csv",
    )
    parser.add_argument(
        "--precomputed_pattern",
        type=str,
        default="*.csv",
        help="Recursive glob pattern for saved result CSVs. Default: *.csv",
    )
    parser.add_argument(
        "--precomputed_model_label",
        type=str,
        default="Proposed_Extended",
        choices=list(MODEL_LABELS.values()),
        help=(
            "Model label used when reading per-subject summary files that only "
            "contain generic RMSE_mean/RMSE_std columns."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="table4_demo_outputs",
        help="Directory for Table 4 summaries.",
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
        help="Optional limit for quick smoke tests in re-run mode.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse existing per-subject scenario4_detailed_results.csv files when available.",
    )
    return parser.parse_args()


def subject_id_from_file(path: Path) -> str:
    stem = path.stem
    for prefix in ("CGMacros-", "subject_", "Subject_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    if stem in {"scenario4_detailed_results", "scenario4_summary_results", "lodo_internal80_20_summary"}:
        return path.parent.name
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
    expected = {"Subject", "PH", *RMSE_COLUMNS}
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

    first_count_col = f"{MODEL_LABELS['RMSE_Extended']}_count"
    subject_level = subject_level.rename(columns={first_count_col: "N_lodo_folds"})
    redundant_count_cols = [c for c in subject_level.columns if c.endswith("_count")]
    subject_level = subject_level.drop(columns=redundant_count_cols)
    return subject_level


def build_table4_summary(subject_level: pd.DataFrame) -> pd.DataFrame:
    if "N_lodo_folds" not in subject_level.columns:
        subject_level = subject_level.copy()
        subject_level["N_lodo_folds"] = 1

    rows = []
    for ph, group in subject_level.groupby("PH", sort=True):
        row = {
            "PH": int(ph),
            "N_subjects": int(group["Subject"].nunique()),
            "N_lodo_folds_total": int(group["N_lodo_folds"].sum()),
        }

        for label in MODEL_LABELS.values():
            subj_mean_col = f"{label}_mean"
            already_summary_col = f"{label}_RMSE_mean"
            if subj_mean_col in group:
                values = group[subj_mean_col]
            elif already_summary_col in group:
                values = group[already_summary_col]
            else:
                continue

            row[f"{label}_RMSE_mean"] = float(values.mean())
            row[f"{label}_RMSE_std_across_subjects"] = float(values.std(ddof=1))
            row[f"{label}_RMSE_weighted_mean_by_folds"] = float(
                np.average(values, weights=group["N_lodo_folds"])
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


def find_precomputed_files(precomputed_dir: Path, pattern: str) -> List[Path]:
    files = sorted(p for p in precomputed_dir.rglob(pattern) if p.name not in GENERATED_OUTPUT_NAMES)
    if not files:
        raise FileNotFoundError(f"No precomputed CSV files matched {pattern!r} in {precomputed_dir}")
    return files


def is_fold_level_result(df: pd.DataFrame) -> bool:
    return "PH" in df.columns and any(col in df.columns for col in RMSE_COLUMNS)


def is_generic_subject_summary(df: pd.DataFrame) -> bool:
    return "PH" in df.columns and "RMSE_mean" in df.columns


def is_labelled_subject_summary(df: pd.DataFrame) -> bool:
    return "PH" in df.columns and any(col in df.columns for col in SUMMARY_MEAN_COLUMNS)


def generic_summary_to_subject_level(df: pd.DataFrame, subject: str, model_label: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Subject"] = df["Subject"] if "Subject" in df.columns else subject
    out["PH"] = df["PH"]
    out["N_lodo_folds"] = df["N_folds"] if "N_folds" in df.columns else df.get("N_lodo_folds", 1)
    out[f"{model_label}_mean"] = df["RMSE_mean"]
    if "RMSE_std" in df.columns:
        out[f"{model_label}_std"] = df["RMSE_std"]
    return out


def labelled_summary_to_subject_level(df: pd.DataFrame, subject: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Subject"] = df["Subject"] if "Subject" in df.columns else subject
    out["PH"] = df["PH"]
    out["N_lodo_folds"] = df["N_lodo_folds"] if "N_lodo_folds" in df.columns else df.get("N_folds", 1)
    for label in MODEL_LABELS.values():
        mean_col = f"{label}_RMSE_mean"
        std_col = f"{label}_RMSE_std_across_subjects"
        if mean_col in df.columns:
            out[f"{label}_mean"] = df[mean_col]
        if std_col in df.columns:
            out[f"{label}_std"] = df[std_col]
    return out


def load_precomputed_results(
    precomputed_dir: Path,
    pattern: str,
    default_model_label: str,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    files = find_precomputed_files(precomputed_dir, pattern)
    fold_frames = []
    subject_frames = []
    ignored = []

    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            ignored.append(path)
            continue
        if "PH" not in df.columns:
            ignored.append(path)
            continue

        subject = subject_id_from_file(path)
        if is_fold_level_result(df):
            frame = df.copy()
            if "Subject" not in frame.columns:
                frame.insert(0, "Subject", subject)
            fold_frames.append(frame)
        elif is_generic_subject_summary(df):
            subject_frames.append(generic_summary_to_subject_level(df, subject, default_model_label))
        elif is_labelled_subject_summary(df):
            subject_frames.append(labelled_summary_to_subject_level(df, subject))
        else:
            ignored.append(path)

    if fold_frames:
        fold_df = pd.concat(fold_frames, ignore_index=True)
        subject_level = build_subject_level_table(fold_df)
        if subject_frames:
            subject_level = pd.concat([subject_level, *subject_frames], ignore_index=True, sort=False)
        return fold_df, subject_level

    if subject_frames:
        return None, pd.concat(subject_frames, ignore_index=True, sort=False)

    ignored_list = "\n".join(f"  - {p}" for p in ignored[:20])
    raise ValueError(
        "No recognised precomputed result files were found. Expected fold-level "
        "RMSE columns or per-subject PH/RMSE_mean summaries. Ignored examples:\n"
        f"{ignored_list}"
    )


def run_from_raw_inputs(args: argparse.Namespace, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.input_dir is None:
        raise ValueError("Provide --input_dir for re-run mode or --precomputed_dir for aggregation-only mode.")

    input_dir = Path(args.input_dir).expanduser().resolve()
    per_subject_dir = output_dir / "per_subject"
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
    subject_level = build_subject_level_table(fold_df)
    return fold_df, subject_level


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.precomputed_dir is not None:
        precomputed_dir = Path(args.precomputed_dir).expanduser().resolve()
        print(f"Reading precomputed result CSV files from {precomputed_dir}")
        fold_df, subject_level = load_precomputed_results(
            precomputed_dir=precomputed_dir,
            pattern=args.precomputed_pattern,
            default_model_label=args.precomputed_model_label,
        )
    else:
        fold_df, subject_level = run_from_raw_inputs(args, output_dir)

    if fold_df is not None:
        fold_path = output_dir / "table4_fold_level_results.csv"
        fold_df.to_csv(fold_path, index=False)
    else:
        fold_path = None

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
    if fold_path is not None:
        print(f"  {fold_path}")
    print(f"  {subject_path}")
    print(f"  {summary_path}")
    print(f"  {paper_ready_path}")


if __name__ == "__main__":
    main()
