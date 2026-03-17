import argparse
import os
import pandas as pd
from scenario4_min_public import run_scenario4


def parse_args():
    parser = argparse.ArgumentParser(description="Batch runner for Scenario 4 minimal public release.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing subject-level CSV files.")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for subject CSV files.")
    parser.add_argument("--output_dir", type=str, default="batch_results", help="Directory for per-subject and pooled outputs.")
    parser.add_argument("--prediction_horizons", type=int, nargs="+", default=[15, 30, 45, 60])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    input_dir = os.path.abspath(args.input_dir)
    files = sorted(pd.Series([str(p) for p in __import__('pathlib').Path(input_dir).glob(args.pattern)]))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched pattern {args.pattern!r} in {input_dir}")

    for file_path in files:
        subject_id = os.path.splitext(os.path.basename(file_path))[0]
        subject_out = os.path.join(args.output_dir, subject_id)
        results_df = run_scenario4(file_path, subject_out, args.prediction_horizons).copy()
        results_df.insert(0, "Subject", subject_id)
        all_results.append(results_df)
        print(f"Finished {subject_id}")

    pooled = pd.concat(all_results, ignore_index=True)
    pooled.to_csv(os.path.join(args.output_dir, "scenario4_all_subjects_detailed.csv"), index=False)

    summary = (
        pooled.groupby("PH")["RMSE"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "RMSE_mean", "std": "RMSE_std"})
    )
    summary.to_csv(os.path.join(args.output_dir, "scenario4_all_subjects_summary.csv"), index=False)
    print(summary)


if __name__ == "__main__":
    main()
