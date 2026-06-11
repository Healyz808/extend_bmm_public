"""Summarise fold-specific fitted parameters from precomputed details.

This utility scans a directory such as ``data/`` for saved fold-level result CSV
files, extracts fitted parameter columns, and writes two audit tables:

    parameter_fold_level.csv
        One row per subject/fold/model with the fitted parameters used in that
        fold. If the input file repeats the same fold across multiple prediction
        horizons, duplicates are removed.

    parameter_subject_summary.csv
        Mean, standard deviation, median, minimum, and maximum of each parameter
        across LODO folds for each subject and model.

Example:
    python summarize_parameters.py --precomputed_dir data --output_dir table4_from_data
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


MODEL_PREFIXES: Dict[str, str] = {
    "Extended_": "Proposed_Extended",
    "BaselineMeal_": "BaselineMeal_CarbsOnly",
    "OriginalBMM_": "Baseline_OriginalBMM",
}
EXCLUDED_TOKENS = ("RMSE", "N_", "count", "mean", "std")
GENERATED_OUTPUT_NAMES = {
    "table4_fold_level_results.csv",
    "table4_subject_level_results.csv",
    "table4_summary.csv",
    "table4_paper_ready.csv",
    "parameter_fold_level.csv",
    "parameter_subject_summary.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise fold-level fitted parameters from saved result CSV files."
    )
    parser.add_argument(
        "--precomputed_dir",
        type=str,
        default="data",
        help="Directory containing saved result CSV files. Default: data",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Recursive glob pattern for result CSV files. Default: *.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="parameter_summary_outputs",
        help="Directory where parameter summaries will be saved.",
    )
    return parser.parse_args()


def subject_id_from_file(path: Path) -> str:
    stem = path.stem
    for prefix in ("CGMacros-", "subject_", "Subject_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    if stem in {"scenario4_detailed_results", "scenario4_summary_results", "lodo_internal80_20_all_results"}:
        return path.parent.name
    return path.parent.name if path.parent.name != "data" else stem


def find_result_files(root: Path, pattern: str) -> List[Path]:
    files = sorted(p for p in root.rglob(pattern) if p.name not in GENERATED_OUTPUT_NAMES)
    if not files:
        raise FileNotFoundError(f"No CSV files matched {pattern!r} in {root}")
    return files


def is_parameter_column(column: str) -> bool:
    if not any(column.startswith(prefix) for prefix in MODEL_PREFIXES):
        return False
    return not any(token in column for token in EXCLUDED_TOKENS)


def model_for_column(column: str) -> Optional[str]:
    for prefix, model in MODEL_PREFIXES.items():
        if column.startswith(prefix):
            return model
    return None


def parameter_name(column: str) -> str:
    for prefix in MODEL_PREFIXES:
        if column.startswith(prefix):
            return column[len(prefix):]
    return column


def fold_id_columns(df: pd.DataFrame) -> List[str]:
    candidates = ["Subject", "Test Day", "test_date", "lodo_fold", "fold", "PH"]
    return [col for col in candidates if col in df.columns]


def load_parameter_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    parameter_cols = [col for col in df.columns if is_parameter_column(col)]
    if not parameter_cols:
        return pd.DataFrame()

    subject = subject_id_from_file(path)
    working = df.copy()
    if "Subject" not in working.columns:
        working.insert(0, "Subject", subject)

    id_cols = fold_id_columns(working)
    # Parameters are commonly repeated once per PH within the same LODO fold.
    # Keep PH if no fold/test-day identifier exists; otherwise drop PH before
    # duplicate removal so each fold parameter set appears once.
    dedupe_cols = [col for col in id_cols if col != "PH"]
    if not dedupe_cols:
        dedupe_cols = id_cols
    dedupe_cols = dedupe_cols + parameter_cols
    working = working.drop_duplicates(subset=dedupe_cols)

    keep_cols = [col for col in id_cols if col in working.columns] + parameter_cols
    long_df = working[keep_cols].melt(
        id_vars=[col for col in id_cols if col in working.columns],
        value_vars=parameter_cols,
        var_name="ParameterColumn",
        value_name="Value",
    )
    long_df = long_df.dropna(subset=["Value"])
    long_df["Model"] = long_df["ParameterColumn"].map(model_for_column)
    long_df["Parameter"] = long_df["ParameterColumn"].map(parameter_name)
    long_df.insert(0, "SourceFile", str(path))
    return long_df


def build_fold_parameter_table(files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    ignored = []
    for path in files:
        try:
            frame = load_parameter_rows(path)
        except Exception:
            ignored.append(path)
            continue
        if frame.empty:
            ignored.append(path)
            continue
        frames.append(frame)

    if not frames:
        examples = "\n".join(f"  - {p}" for p in ignored[:20])
        raise ValueError(
            "No parameter columns were found. Expected columns beginning with "
            "Extended_, BaselineMeal_, or OriginalBMM_. Ignored examples:\n"
            f"{examples}"
        )
    return pd.concat(frames, ignore_index=True)


def build_subject_parameter_summary(fold_params: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["Subject", "Model", "Parameter"]
    summary = (
        fold_params.groupby(group_cols)["Value"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
        .rename(columns={"count": "N_folds"})
    )
    return summary


def main() -> None:
    args = parse_args()
    root = Path(args.precomputed_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_result_files(root, args.pattern)
    print(f"Scanning {len(files)} CSV file(s) under {root}")

    fold_params = build_fold_parameter_table(files)
    fold_path = output_dir / "parameter_fold_level.csv"
    fold_params.to_csv(fold_path, index=False)

    subject_summary = build_subject_parameter_summary(fold_params)
    summary_path = output_dir / "parameter_subject_summary.csv"
    subject_summary.to_csv(summary_path, index=False)

    print("\nDetected parameter summary preview:")
    print(subject_summary.head(20).to_string(index=False))
    print("\nSaved outputs:")
    print(f"  {fold_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
