"""
Minimal open-source implementation of the extended Bergman model described in the paper.

What this script keeps
----------------------
1. Subject-specific extended Bergman ODE with
   - baseline glucose-insulin dynamics,
   - GI-driven meal disturbance,
   - circadian modulation.
2. Leave-One-Day-Out cross-validation (LODOCV).
3. Chronological 80/20 split of the training days inside each outer fold:
   - first 80%: parameter identification (Bayesian optimisation)
   - last 20%: internal validation
4. Optional rolling multi-horizon prediction on the held-out test day.

What this script intentionally removes
--------------------------------------
- internal 5-fold processing
- outlier detection
- engineered rolling statistics not used by the ODE model
- extra plotting/reporting logic not required for reproducibility

Expected input CSV columns
--------------------------
Required:
    Timestamp, Libre GL
Optional but recommended:
    Carbs, GI, Meal Type

Time unit: minutes
Glucose unit: mg/dL
Insulin unit: µU/mL
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from scipy.integrate import odeint
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================
# 1. Physiological constants
# ============================
GB = 81.0                 # basal glucose (mg/dL)
IB = 14.0                 # basal insulin (µU/mL)
N_CLEAR = 5.0 / 54.0      # insulin clearance (min^-1)
P2 = 0.0287               # remote insulin action decay (min^-1)
H_THRESHOLD = 110.0       # glucose threshold for endogenous secretion (mg/dL)
GAMMA_SEC = 5e-3          # fixed pancreatic secretion gain (kept fixed in minimal release)


# ============================
# 2. Data handling
# ============================
def load_subject_csv(file_path: str) -> pd.DataFrame:
    """Load one subject CSV and keep only columns needed by the paper model."""
    df = pd.read_csv(file_path)

    required = {"Timestamp", "Libre GL"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Model-required channels
    if "Carbs" not in df.columns:
        df["Carbs"] = 0.0
    if "GI" not in df.columns:
        df["GI"] = 0.0
    if "Meal Type" not in df.columns:
        df["Meal Type"] = "unknown"

    # Minimal preprocessing consistent with the manuscript text.
    df = df.dropna(subset=["Libre GL"]).copy()
    df["Libre GL"] = df["Libre GL"].astype(float)
    df["Carbs"] = pd.to_numeric(df["Carbs"], errors="coerce").fillna(0.0)
    df["GI"] = pd.to_numeric(df["GI"], errors="coerce").fillna(0.0)
    df["Meal Type"] = df["Meal Type"].fillna("unknown").astype(str)
    df["Date"] = df["Timestamp"].dt.date

    return df.reset_index(drop=True)


# ============================
# 3. Meal-effect model
# ============================
def phi_gi(gi_value: float) -> float:
    """Empirical GI modifier used in the original working code."""
    if gi_value > 70:
        return 1.2 + (gi_value - 70.0) * 0.005
    if gi_value < 30:
        return 0.8 - (30.0 - gi_value) * 0.005
    return 1.0 + (gi_value - 50.0) * 0.004


def phi_type(meal_type: str) -> float:
    """Empirical meal-type modifier used in the original working code."""
    name = str(meal_type).strip().lower()
    if "breakfast" in name:
        return 1.2
    if "dinner" in name:
        return 0.9
    return 1.0


@dataclass
class ModelParams:
    p1: float
    p3: float
    beta: float
    lambda_decay: float
    tau: float
    peak: float


def bi_exponential_kernel(dt_eff: float, beta: float, lambda_decay: float, peak: float) -> float:
    """Bi-exponential kernel in Eq. (2.9)."""
    if dt_eff < 0:
        return 0.0
    return peak * (1.0 - math.exp(-beta * dt_eff)) * math.exp(-lambda_decay * dt_eff)


def gi_disturbance(
    t_min: float,
    meals: pd.DataFrame,
    time_origin: pd.Timestamp,
    params: ModelParams,
) -> float:
    """GI-driven disturbance D_GI(t) in Eq. (2.7)."""
    total = 0.0
    if meals.empty:
        return total

    for row in meals.itertuples(index=False):
        meal_time = (row.Timestamp - time_origin).total_seconds() / 60.0
        dt_eff = t_min - meal_time - params.tau
        if dt_eff < 0:
            continue

        gi = max(0.0, float(row.GI))
        carbs = max(0.0, float(row.Carbs))
        fm = carbs * (gi / 100.0) * phi_gi(gi) * phi_type(row.MealType)
        total += fm * bi_exponential_kernel(dt_eff, params.beta, params.lambda_decay, params.peak)
    return total


# ============================
# 4. Circadian modulation
# ============================
def circadian_factor(hour_of_day: float, ac1: float = 0.15, ac2: float = 0.10, sigma1: float = 2.0, sigma2: float = 4.0) -> float:
    """Circadian modulation C(t) in Eq. (2.10)."""
    morning = ac1 * math.exp(-((hour_of_day - 6.0) ** 2) / (sigma1 ** 2))
    evening = ac2 * math.exp(-((hour_of_day - 18.0) ** 2) / (sigma2 ** 2))
    return 1.0 + morning - evening


# ============================
# 5. ODE system
# ============================
def extended_bmm_ode(
    y: Iterable[float],
    t_min: float,
    params: ModelParams,
    time_origin: pd.Timestamp,
    meals: pd.DataFrame,
) -> List[float]:
    """
    Extended Bergman model.

    States are represented as actual concentrations:
        G : glucose concentration (mg/dL)
        X : remote insulin action
        I : insulin concentration (µU/mL)
    """
    g, x, i = y

    disturbance = gi_disturbance(t_min, meals, time_origin, params)
    clock_hour = ((time_origin.hour * 60 + time_origin.minute + t_min) / 60.0) % 24.0
    c_t = circadian_factor(clock_hour)

    dgdt = -params.p1 * (g - GB) - g * x + disturbance * c_t
    dxdt = -P2 * x + params.p3 * (i - IB)
    didt = -N_CLEAR * (i - IB) + GAMMA_SEC * max(g - H_THRESHOLD, 0.0)
    return [dgdt, dxdt, didt]


# ============================
# 6. Simulation utilities
# ============================
def extract_meals(df: pd.DataFrame) -> pd.DataFrame:
    meals = df.loc[(df["Carbs"] > 0) & (df["GI"] > 0), ["Timestamp", "Meal Type", "GI", "Carbs"]].copy()
    meals = meals.rename(columns={"Meal Type": "MealType"})
    meals = meals.sort_values("Timestamp").reset_index(drop=True)
    return meals


def simulate_segment(df: pd.DataFrame, params: ModelParams) -> np.ndarray:
    """Simulate an entire contiguous segment and interpolate predictions to observed timestamps."""
    if df.empty:
        return np.array([], dtype=float)

    df = df.sort_values("Timestamp").reset_index(drop=True).copy()
    time_origin = df.loc[0, "Timestamp"]
    meals = extract_meals(df)
    df["t_min"] = (df["Timestamp"] - time_origin).dt.total_seconds() / 60.0

    t_max = float(df["t_min"].max())
    t_grid = np.arange(0.0, math.floor(t_max) + 1.0, 1.0)
    if len(t_grid) == 0 or t_grid[-1] < t_max:
        t_grid = np.append(t_grid, t_max)

    y0 = [float(df.loc[0, "Libre GL"]), 0.0, IB]
    solution = odeint(lambda y, t: extended_bmm_ode(y, t, params, time_origin, meals), y0, t_grid)
    g_pred_grid = solution[:, 0]
    g_pred_obs = np.interp(df["t_min"].to_numpy(), t_grid, g_pred_grid)
    return g_pred_obs


# ============================
# 7. Metrics
# ============================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "N": int(len(y_true)),
    }


# ============================
# 8. Parameter identification
# ============================
def optimise_parameters(calibration_df: pd.DataFrame) -> ModelParams:
    """Bayesian optimisation on the first 80% of training days."""
    if calibration_df.empty or len(calibration_df) < 10:
        raise ValueError("Calibration partition is too small for optimisation.")

    bounds = {
        "p1": (0.001, 0.05),
        "p3": (1e-6, 1e-4),
        "beta": (0.005, 1.0),
        "lambda_decay": (0.005, 1.0),
        "tau": (0.0, 90.0),
        "peak": (1.0, 6.0),
    }

    y_true = calibration_df["Libre GL"].to_numpy(dtype=float)

    def objective(p1: float, p3: float, beta: float, lambda_decay: float, tau: float, peak: float) -> float:
        params = ModelParams(p1=p1, p3=p3, beta=beta, lambda_decay=lambda_decay, tau=tau, peak=peak)
        try:
            y_pred = simulate_segment(calibration_df, params)
            return -rmse(y_true, y_pred)
        except Exception:
            return -1e6

    bo = BayesianOptimization(f=objective, pbounds=bounds, random_state=42, verbose=0)
    bo.maximize(init_points=8, n_iter=20)
    best = bo.max["params"]
    return ModelParams(**best)


# ============================
# 9. LODOCV with 80/20 split
# ============================
def chronological_80_20_day_split(train_dates: List) -> Tuple[List, List]:
    if len(train_dates) < 2:
        raise ValueError("Need at least two training days for chronological 80/20 split.")
    split_idx = max(1, int(math.floor(0.8 * len(train_dates))))
    if split_idx >= len(train_dates):
        split_idx = len(train_dates) - 1
    return train_dates[:split_idx], train_dates[split_idx:]


# ============================
# 10. Rolling multi-horizon prediction
# ============================
def rolling_multi_horizon_predict(
    test_df: pd.DataFrame,
    params: ModelParams,
    horizons: List[int],
) -> Dict[int, List[Dict[str, float]]]:
    """
    Generate rolling forecasts on the held-out day.
    For each decision time, integrate forward from the current glucose state.
    """
    results: Dict[int, List[Dict[str, float]]] = {h: [] for h in horizons}
    if test_df.empty or len(test_df) < 2:
        return results

    test_df = test_df.sort_values("Timestamp").reset_index(drop=True).copy()
    max_h = max(horizons)

    for idx in range(len(test_df) - 1):
        current_time = test_df.loc[idx, "Timestamp"]
        current_glucose = float(test_df.loc[idx, "Libre GL"])

        history = test_df.loc[:idx].copy()
        meals = extract_meals(history)
        time_origin = current_time
        y0 = [current_glucose, 0.0, IB]
        t_grid = np.arange(0.0, max_h + 1.0, 1.0)
        sol = odeint(lambda y, t: extended_bmm_ode(y, t, params, time_origin, meals), y0, t_grid)
        g_pred_grid = sol[:, 0]

        for h in horizons:
            target_time = current_time + timedelta(minutes=int(h))
            future = test_df[test_df["Timestamp"] >= target_time]
            if future.empty:
                continue
            actual_row = future.iloc[0]
            results[h].append({
                "current_time": str(current_time),
                "target_time": str(actual_row["Timestamp"]),
                "y_true": float(actual_row["Libre GL"]),
                "y_pred": float(g_pred_grid[int(h)]),
            })

    return results


# ============================
# 11. Main experiment runner
# ============================
def run_lodocv(file_path: str, horizons: List[int], output_dir: str) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    df = load_subject_csv(file_path)
    unique_dates = sorted(df["Date"].unique())

    if len(unique_dates) < 3:
        raise ValueError("At least 3 calendar days are required for LODOCV.")

    fold_summaries = []
    horizon_records = {h: [] for h in horizons}

    for fold_idx, test_date in enumerate(unique_dates, start=1):
        train_dates = [d for d in unique_dates if d != test_date]
        calib_dates, val_dates = chronological_80_20_day_split(train_dates)

        calibration_df = df[df["Date"].isin(calib_dates)].copy()
        validation_df = df[df["Date"].isin(val_dates)].copy()
        test_df = df[df["Date"] == test_date].copy()

        params = optimise_parameters(calibration_df)

        val_pred = simulate_segment(validation_df, params)
        test_pred = simulate_segment(test_df, params)

        val_metrics = regression_metrics(validation_df["Libre GL"].to_numpy(dtype=float), val_pred)
        test_metrics = regression_metrics(test_df["Libre GL"].to_numpy(dtype=float), test_pred)

        fold_summary = {
            "fold": fold_idx,
            "test_date": str(test_date),
            "n_train_days": len(train_dates),
            "n_calibration_days": len(calib_dates),
            "n_validation_days": len(val_dates),
            "validation_RMSE": val_metrics["RMSE"],
            "test_RMSE_24h": test_metrics["RMSE"],
            **asdict(params),
        }
        fold_summaries.append(fold_summary)

        rolling = rolling_multi_horizon_predict(test_df, params, horizons)
        for h in horizons:
            if not rolling[h]:
                continue
            y_true = np.array([r["y_true"] for r in rolling[h]], dtype=float)
            y_pred = np.array([r["y_pred"] for r in rolling[h]], dtype=float)
            metrics = regression_metrics(y_true, y_pred)
            horizon_records[h].append({
                "fold": fold_idx,
                "test_date": str(test_date),
                **metrics,
            })

    fold_df = pd.DataFrame(fold_summaries)
    fold_df.to_csv(os.path.join(output_dir, "lodocv_24h_results.csv"), index=False)

    horizon_summary_rows = []
    for h in horizons:
        h_df = pd.DataFrame(horizon_records[h])
        if h_df.empty:
            continue
        h_df.to_csv(os.path.join(output_dir, f"lodocv_PH{h}_fold_results.csv"), index=False)
        horizon_summary_rows.append({
            "PH": h,
            "RMSE_mean": float(h_df["RMSE"].mean()),
            "RMSE_std": float(h_df["RMSE"].std(ddof=1)) if len(h_df) > 1 else 0.0,
            "MAE_mean": float(h_df["MAE"].mean()),
            "MAE_std": float(h_df["MAE"].std(ddof=1)) if len(h_df) > 1 else 0.0,
            "n_folds": int(len(h_df)),
        })

    horizon_summary_df = pd.DataFrame(horizon_summary_rows)
    if not horizon_summary_df.empty:
        horizon_summary_df.to_csv(os.path.join(output_dir, "lodocv_multi_horizon_summary.csv"), index=False)

    summary = {
        "n_days": int(len(unique_dates)),
        "mean_test_RMSE_24h": float(fold_df["test_RMSE_24h"].mean()),
        "std_test_RMSE_24h": float(fold_df["test_RMSE_24h"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "fold_results_file": os.path.join(output_dir, "lodocv_24h_results.csv"),
        "multi_horizon_summary_file": os.path.join(output_dir, "lodocv_multi_horizon_summary.csv"),
    }

    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal extended Bergman model aligned with the paper method.")
    parser.add_argument("--input_csv", required=True, help="Path to one subject CSV file")
    parser.add_argument("--output_dir", default="results_extended_bmm_minimal", help="Output directory")
    parser.add_argument("--prediction_horizons", nargs="+", type=int, default=[15, 30, 45, 60], help="Prediction horizons in minutes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_lodocv(
        file_path=args.input_csv,
        horizons=args.prediction_horizons,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))
