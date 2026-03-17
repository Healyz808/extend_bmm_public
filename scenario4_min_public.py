import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution


@dataclass
class Config:
    Gb: float = 81.0
    Ib: float = 14.0
    n: float = 5 / 54
    p2: float = 0.0287
    dt_minutes: int = 1
    day_minutes: int = 1440
    random_state: int = 42
    init_points: int = 5
    n_iter: int = 10


REQUIRED_COLUMNS = ["Timestamp", "Libre GL", "Meal Type", "GI", "Carbs"]


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Libre GL"]).copy()
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Date"] = df["Timestamp"].dt.date

    df = df.set_index("Timestamp")
    df["Libre GL"] = df["Libre GL"].interpolate(method="time")
    df = df.reset_index()
    return df


def add_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Timestamp").reset_index(drop=True)
    start_time = df["Timestamp"].min()
    df["t_min"] = (df["Timestamp"] - start_time).dt.total_seconds() / 60.0
    return df


def lodo_splits(df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, object]]:
    splits = []
    unique_dates = sorted(df["Date"].unique())
    for test_day in unique_dates:
        train_df = df[df["Date"] != test_day].copy()
        test_df = df[df["Date"] == test_day].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        splits.append((train_df, test_df, test_day))
    return splits


def chronological_train_valid_split(train_df: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.sort_values("Timestamp").reset_index(drop=True)
    split_idx = max(1, int(len(train_df) * ratio))
    calib_df = train_df.iloc[:split_idx].copy()
    valid_df = train_df.iloc[split_idx:].copy()
    if len(valid_df) == 0:
        valid_df = calib_df.tail(min(20, len(calib_df))).copy()
    return calib_df, valid_df


def phi_gi(gi_value: float) -> float:
    if gi_value > 70:
        return 1.2
    if gi_value < 30:
        return 0.8
    return 1.0


def phi_type(meal_type: str) -> float:
    meal_type = str(meal_type).strip().lower()
    if "breakfast" in meal_type:
        return 1.05
    if "lunch" in meal_type:
        return 1.00
    if "dinner" in meal_type:
        return 0.95
    if "snack" in meal_type:
        return 0.90
    return 1.00


def absorption_kernel(dt: float, beta_meal: float, gamma_meal: float, t_lag: float, peak_mult: float) -> float:
    shifted_dt = dt - t_lag
    if shifted_dt < 0:
        return 0.0
    return peak_mult * (1.0 - np.exp(-beta_meal * shifted_dt)) * np.exp(-gamma_meal * shifted_dt)


def meal_disturbance(
    t_val: float,
    meal_data: pd.DataFrame,
    current_start: pd.Timestamp,
    beta_meal: float,
    gamma_meal: float,
    t_lag: float,
    peak_mult: float,
) -> float:
    total = 0.0
    for _, row in meal_data.iterrows():
        meal_t = (row["Timestamp"] - current_start).total_seconds() / 60.0
        dt = t_val - meal_t
        fm = (
            float(row["Carbs"])
            * float(row["GI"]) / 100.0
            * phi_gi(float(row["GI"]))
            * phi_type(row["Meal Type"])
        )
        total += fm * absorption_kernel(dt, beta_meal, gamma_meal, t_lag, peak_mult)
    return total


def circadian_factor(t_val: float) -> float:
    hour_of_day = (t_val / 60.0) % 24.0
    return 1.0 + 0.1 * np.sin(2.0 * np.pi * (hour_of_day - 6.0) / 24.0)


def bergman_min_public(y, t_val: float, params: Dict[str, float], meal_data: pd.DataFrame, current_start: pd.Timestamp, cfg: Config):
    G, X, I = y
    d_meal = meal_disturbance(
        t_val=t_val,
        meal_data=meal_data,
        current_start=current_start,
        beta_meal=params["beta_meal"],
        gamma_meal=params["gamma_meal"],
        t_lag=params["t_lag"],
        peak_mult=params["peak_mult"],
    )
    c_t = circadian_factor(t_val)

    dGdt = -params["p1"] * (G - cfg.Gb) - G * X + d_meal * c_t
    dXdt = -cfg.p2 * X + params["p3"] * (I - cfg.Ib)
    dIdt = -cfg.n * (I - cfg.Ib)
    return [dGdt, dXdt, dIdt]


def solve_system(params: Dict[str, float], meal_data: pd.DataFrame, current_start: pd.Timestamp, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    t_grid = np.arange(0, cfg.day_minutes + cfg.dt_minutes, cfg.dt_minutes)
    y0 = [cfg.Gb, 0.0, cfg.Ib]

    def ode_func(y, t_val):
        return bergman_min_public(y, t_val, params, meal_data, current_start, cfg)

    sol = odeint(ode_func, y0, t_grid)
    glucose_pred = sol[:, 0]
    return t_grid, glucose_pred


def extract_meals(df: pd.DataFrame) -> pd.DataFrame:
    meal_df = df[["Timestamp", "Meal Type", "GI", "Carbs"]].copy()
    meal_df = meal_df[(meal_df["GI"] > 0) & (meal_df["Carbs"] > 0)].reset_index(drop=True)
    return meal_df


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def predict_last_step(model_params: Dict[str, float], test_df: pd.DataFrame, cfg: Config, prediction_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    test_df = add_relative_time(test_df)
    test_start = test_df["Timestamp"].min()
    meals = extract_meals(test_df)

    obs_t = test_df["t_min"].values
    obs_g = test_df["Libre GL"].values

    pred_times = []
    pred_values = []

    t_grid, g_pred = solve_system(model_params, meals, test_start, cfg)
    for i in range(len(obs_t)):
        target_t = obs_t[i] + prediction_horizon
        if target_t > cfg.day_minutes:
            continue
        target_pred = np.interp(target_t, t_grid, g_pred)
        pred_times.append(target_t)
        pred_values.append(target_pred)

    pred_times = np.array(pred_times, dtype=float)
    pred_values = np.array(pred_values, dtype=float)
    target_obs = np.array([np.interp(target_t, obs_t, obs_g) for target_t in pred_times], dtype=float)
    return target_obs, pred_values


def optimize_parameters(calib_df: pd.DataFrame, valid_df: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    calib_df = add_relative_time(calib_df)
    valid_df = add_relative_time(valid_df)

    calib_start = calib_df["Timestamp"].min()
    calib_meals = extract_meals(calib_df)
    valid_t = valid_df["t_min"].values
    valid_g = valid_df["Libre GL"].values

    bounds = [
        (0.001, 0.05),
        (1e-6, 1e-4),
        (0.005, 1.0),
        (0.005, 1.0),
        (0.0, 90.0),
        (1.0, 6.0),
    ]

    def objective(theta):
        p1, p3, beta_meal, gamma_meal, t_lag, peak_mult = theta
        params = {
            "p1": p1,
            "p3": p3,
            "beta_meal": beta_meal,
            "gamma_meal": gamma_meal,
            "t_lag": t_lag,
            "peak_mult": peak_mult,
        }
        t_grid, g_pred = solve_system(params, calib_meals, calib_start, cfg)
        g_valid_pred = np.interp(valid_t, t_grid, g_pred)
        return rmse(valid_g, g_valid_pred)

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=cfg.random_state,
        maxiter=cfg.n_iter,
        popsize=max(5, cfg.init_points),
        polish=False,
        updating="deferred",
    )

    p1, p3, beta_meal, gamma_meal, t_lag, peak_mult = result.x
    return {
        "p1": float(p1),
        "p3": float(p3),
        "beta_meal": float(beta_meal),
        "gamma_meal": float(gamma_meal),
        "t_lag": float(t_lag),
        "peak_mult": float(peak_mult),
    }


def run_scenario4(input_csv: str, output_dir: str, prediction_horizons: List[int]) -> pd.DataFrame:
    cfg = Config()
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(input_csv)
    splits = lodo_splits(df)

    all_results = []
    for train_df, test_df, test_day in splits:
        calib_df, valid_df = chronological_train_valid_split(train_df, ratio=0.8)
        best_params = optimize_parameters(calib_df, valid_df, cfg)

        for ph in prediction_horizons:
            y_true, y_pred = predict_last_step(best_params, test_df, cfg, prediction_horizon=ph)
            fold_rmse = rmse(y_true, y_pred)
            all_results.append({
                "Test Day": str(test_day),
                "PH": ph,
                "RMSE": fold_rmse,
                "p1": best_params["p1"],
                "p3": best_params["p3"],
                "beta_meal": best_params["beta_meal"],
                "gamma_meal": best_params["gamma_meal"],
                "t_lag": best_params["t_lag"],
                "peak_mult": best_params["peak_mult"],
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "scenario4_detailed_results.csv"), index=False)

    summary_df = (
        results_df.groupby("PH")["RMSE"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "RMSE_mean", "std": "RMSE_std"})
    )
    summary_df.to_csv(os.path.join(output_dir, "scenario4_summary_results.csv"), index=False)
    return results_df


def generate_demo_subject(output_csv: str, n_days: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01 00:00:00")
    rows = []
    for day in range(n_days):
        day_start = start + pd.Timedelta(days=day)
        times = pd.date_range(day_start, periods=288, freq="5min")
        base = 110 + 12 * np.sin(np.linspace(0, 2 * np.pi, len(times)))
        glucose = base + rng.normal(0, 5, len(times))
        meal_slots = {8 * 60: ("Breakfast", 55, 45), 13 * 60: ("Lunch", 70, 65), 19 * 60: ("Dinner", 60, 80)}
        for i, ts in enumerate(times):
            minute = ts.hour * 60 + ts.minute
            meal_type, carbs, gi = meal_slots.get(minute, (np.nan, 0, 0))
            rows.append({
                "Timestamp": ts,
                "Libre GL": float(glucose[i]),
                "Meal Type": meal_type,
                "GI": gi,
                "Carbs": carbs,
            })
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Scenario 4 minimal public-release implementation.")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to one subject CSV file.")
    parser.add_argument("--output_dir", type=str, default="scenario4_outputs", help="Output directory.")
    parser.add_argument("--prediction_horizons", type=int, nargs="+", default=[15, 30, 45, 60], help="Prediction horizons in minutes.")
    parser.add_argument("--demo", action="store_true", help="Generate and run a small synthetic demo dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    input_csv = args.input_csv
    if args.demo:
        input_csv = os.path.join(args.output_dir, "demo_subject.csv")
        generate_demo_subject(input_csv)

    if input_csv is None:
        raise ValueError("Provide --input_csv or use --demo.")

    results_df = run_scenario4(input_csv=input_csv, output_dir=args.output_dir, prediction_horizons=args.prediction_horizons)
    print("\nDetailed results preview:")
    print(results_df.head())
    print(f"\nSaved summary results to: {os.path.join(args.output_dir, 'scenario4_summary_results.csv')}")


if __name__ == "__main__":
    main()
