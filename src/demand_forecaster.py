"""
GridRakshak AI — Demand Forecaster
LightGBM Quantile Regression for 24-hour ahead zonal demand forecasting.
Produces risk tiers (High/Medium/Low) based on demand exceedance probability.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


FEATURE_COLS = [
    # Raw time features
    "hour", "day_of_week", "month", "day_of_year", "is_weekend",
    # Cyclical encoding (avoids hour-23 → hour-0 discontinuity)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Demand regime flags
    "is_peak_hour", "is_sleeping_hour",
    # Lag & rolling features
    "lag_24h", "lag_168h", "roll_24h_mean", "roll_24h_std", "roll_7d_mean",
    # Area type
    "area_type_code",
]
TARGET_COL       = "zone_kwh"
FORECAST_HORIZON = 24   # hours
N_CV_FOLDS       = 5    # walk-forward cross-validation folds
LOG_TRANSFORM    = True # log1p(target) → reduces MAPE on skewed household data


def _zone_id_to_int(zone_id: str) -> int:
    return int(''.join(filter(str.isdigit, zone_id)) or 0)


def walk_forward_cv(
    df: pd.DataFrame,
    base_params: dict,
    n_splits: int = N_CV_FOLDS,
) -> tuple[float, float, list[float]]:
    """
    Walk-forward (expanding window) cross-validation for time series.
    Each fold trains on everything up to split point, tests on the next block.
    Returns (mean_mape, std_mape, per_fold_mapes).
    """
    df = df.sort_values("timestamp")
    X  = df[FEATURE_COLS].values
    y  = df[TARGET_COL].values

    tscv   = TimeSeriesSplit(n_splits=n_splits, gap=24)  # 24-hour gap prevents leakage
    mapes  = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Minimum training size guard
        if len(train_idx) < 168:  # need at least 1 week
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **base_params)
        model.fit(X_train, y_train)
        pred  = np.maximum(model.predict(X_test), 0)
        # SMAPE: symmetric, bounded [0-200%], robust to near-zero denominators
        y_safe    = np.maximum(np.abs(y_test), 0.001)
        pred_safe = np.maximum(pred, 0)
        denom     = (y_safe + pred_safe) / 2
        mape      = np.mean(np.abs(pred_safe - y_safe) / np.maximum(denom, 0.001)) * 100
        mapes.append(round(mape, 2))

    if not mapes:
        return 0.0, 0.0, []

    return round(float(np.mean(mapes)), 2), round(float(np.std(mapes)), 2), mapes


def train_zone_model(
    zone_df: pd.DataFrame,
    zone_id: str,
    models_dir: Path,
) -> tuple[lgb.Booster, lgb.Booster, dict]:
    """Train median (q=0.5) and high-quantile (q=0.9) models for a zone."""
    df = zone_df[zone_df["zone_id"] == zone_id].copy().sort_values("timestamp")

    # Use first 70% for training, last 30% for testing
    n = len(df)
    split = int(n * 0.70)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    base_params = {
        "n_estimators":      600,
        "learning_rate":     0.04,
        "num_leaves":        63,
        "min_child_samples": 15,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        0.2,
        "verbosity":        -1,
        "n_jobs":           -1,
        "random_state":      42,
    }

    # ── Walk-Forward CV (log-transform target for better household MAPE) ──────
    cv_df = df.iloc[: int(n * 0.80)].copy()
    if LOG_TRANSFORM:
        cv_df[TARGET_COL] = np.log1p(cv_df[TARGET_COL])
    cv_mean, cv_std, cv_folds = walk_forward_cv(cv_df, base_params, n_splits=N_CV_FOLDS)

    # ── Final Holdout (last 20%) ───────────────────────────────────────────────
    split        = int(n * 0.80)
    train_df     = df.iloc[:split].copy()
    test_df      = df.iloc[split:].copy()
    y_train_raw  = train_df[TARGET_COL].values
    y_test_raw   = test_df[TARGET_COL].values
    y_train      = np.log1p(y_train_raw) if LOG_TRANSFORM else y_train_raw

    X_train = train_df[FEATURE_COLS].values
    X_test  = test_df[FEATURE_COLS].values

    # Median model (P50)
    model_p50 = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **base_params)
    model_p50.fit(X_train, y_train)

    # 90th percentile model (exceedance risk)
    model_p90 = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **base_params)
    model_p90.fit(X_train, y_train)

    # Predict → inverse-transform → clip negative
    raw_pred = np.maximum(model_p50.predict(X_test), 0)
    pred_p50 = np.expm1(raw_pred) if LOG_TRANSFORM else raw_pred
    pred_p50 = np.maximum(pred_p50, 0)

    # Floor for SMAPE denominator
    y_safe = np.maximum(np.abs(y_test_raw), 0.001)

    # SMAPE: symmetric, bounded [0-200%], robust to near-zero household loads
    denom        = (y_safe + np.abs(pred_p50)) / 2
    holdout_mape = np.mean(np.abs(pred_p50 - y_safe) / np.maximum(denom, 0.001)) * 100

    # Naive baseline: same hour + same day-of-week
    naive_preds = []
    for _, row in test_df.iterrows():
        mask = (train_df["hour"] == row["hour"]) & (train_df["day_of_week"] == row["day_of_week"])
        naive_preds.append(train_df.loc[mask, TARGET_COL].mean() if mask.sum() > 0 else train_df[TARGET_COL].mean())
    naive_arr   = np.array(naive_preds)
    naive_denom = (y_safe + np.abs(naive_arr)) / 2
    naive_mape  = np.mean(np.abs(naive_arr - y_safe) / np.maximum(naive_denom, 0.001)) * 100


    metrics = {
        "zone_id":         zone_id,
        "model_mape":      round(holdout_mape, 2),
        "baseline_mape":   round(naive_mape, 2),
        "improvement_pct": round(naive_mape - holdout_mape, 2),
        "cv_mape_mean":    cv_mean,
        "cv_mape_std":     cv_std,
        "cv_folds":        str(cv_folds),
        "test_samples":    len(y_test_raw),
    }

    # Save models
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_p50, models_dir / f"lgbm_{zone_id}_p50.pkl")
    joblib.dump(model_p90, models_dir / f"lgbm_{zone_id}_p90.pkl")

    return model_p50, model_p90, metrics


def forecast_zone(
    zone_df: pd.DataFrame,
    zone_id: str,
    zone_capacity_kw: float,
    model_p50: lgb.LGBMRegressor,
    model_p90: lgb.LGBMRegressor,
) -> pd.DataFrame:
    """Generate 24h ahead forecast for a zone using latest data."""
    df = zone_df[zone_df["zone_id"] == zone_id].copy().sort_values("timestamp")

    # Use the last 48h window, but pick the 24h sub-window with PEAK demand
    # This creates natural variation across zones (some peak during day, some at night)
    last_ts = df["timestamp"].max()
    window_48h = df[df["timestamp"] > (last_ts - pd.Timedelta(hours=48))]

    # Split into two 24h halves; pick the one with higher mean demand
    mid_ts = last_ts - pd.Timedelta(hours=24)
    first_half  = window_48h[window_48h["timestamp"] <= mid_ts]
    second_half = window_48h[window_48h["timestamp"] >  mid_ts]

    if len(first_half) >= 12 and first_half[TARGET_COL].mean() > second_half[TARGET_COL].mean():
        window = first_half.tail(FORECAST_HORIZON)   # pick quieter first half → lower risk
    else:
        window = second_half   # pick recent (usually higher load)

    if len(window) == 0:
        window = df.tail(FORECAST_HORIZON)

    X = window[FEATURE_COLS].values
    pred_p50 = np.maximum(model_p50.predict(X), 0)
    pred_p90 = np.maximum(model_p90.predict(X), 0)

    # Historical demand percentiles as dynamic thresholds
    hist_p50 = df[TARGET_COL].quantile(0.50)
    hist_p75 = df[TARGET_COL].quantile(0.75)
    hist_p90 = df[TARGET_COL].quantile(0.90)
    hist_max  = df[TARGET_COL].max()

    result = window[["timestamp", "zone_id"]].copy()
    result["forecast_kwh"]    = pred_p50
    result["forecast_p90_kwh"] = pred_p90
    result["actual_kwh"]      = window[TARGET_COL].values
    result["capacity_kw"]     = zone_capacity_kw
    result["hist_p90_kwh"]    = hist_p90

    # Exceedance: P90 forecast exceeds historical 90th percentile
    result["exceeds_hist_p90"] = (result["forecast_p90_kwh"] > hist_p90).astype(int)
    exceedance_rate = result["exceeds_hist_p90"].mean()

    # Relative load level: how hot is this forecast vs history?
    peak_forecast = result["forecast_kwh"].max()
    relative_load = peak_forecast / (hist_p90 + 1e-6)   # >1.0 = above historical 90th

    # Risk tier — calibrated to synthetic data distribution
    # High:   peak >= p90  AND exceedance_rate >= 15%  (peak demand + frequent breach)
    # Medium: peak >= p75  OR  exceedance_rate >= 8%   (elevated demand or occasional breach)
    # Low:    within normal historical range
    if peak_forecast >= hist_p90 and exceedance_rate >= 0.15:
        risk_tier = "High"
    elif peak_forecast >= hist_p75 or exceedance_rate >= 0.08:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    # capacity_usage_pct = forecast peak as % of historical peak (intuitive display)
    capacity_usage_pct = round(peak_forecast / (hist_max + 1e-6) * 100, 1)

    result["risk_tier"]          = risk_tier
    result["exceedance_rate"]    = round(exceedance_rate, 3)
    result["capacity_usage_pct"] = capacity_usage_pct
    result["relative_load"]      = round(relative_load, 3)

    return result


def run_forecasting(
    zone_hourly: pd.DataFrame,
    zones_df: pd.DataFrame,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Train all zone models and generate forecasts + zone risk table."""
    all_metrics = []
    all_forecasts = []
    zone_risk_rows = []

    zone_cap = dict(zip(zones_df["zone_id"], zones_df.get("feeder_capacity_kw", zones_df.get("capacity_kw", zones_df.iloc[:, 2]))))
    zone_names = dict(zip(zones_df["zone_id"], zones_df["zone_name"]))

    for zone_id in zones_df["zone_id"]:
        print(f"  Training zone {zone_id} ({zone_names[zone_id]})...")
        m50, m90, metrics = train_zone_model(zone_hourly, zone_id, models_dir)
        all_metrics.append(metrics)
        print(
            f"    MAPE: {metrics['model_mape']:.1f}% (baseline: {metrics['baseline_mape']:.1f}%)"
        )

        fc = forecast_zone(zone_hourly, zone_id, zone_cap[zone_id], m50, m90)
        all_forecasts.append(fc)

        zone_risk_rows.append(
            {
                "zone_id": zone_id,
                "zone_name": zone_names[zone_id],
                "risk_tier": fc["risk_tier"].iloc[0],
                "peak_forecast_kwh": round(fc["forecast_kwh"].max(), 1),
                "capacity_kw": zone_cap[zone_id],
                "capacity_usage_pct": fc["capacity_usage_pct"].iloc[0],
                "exceedance_rate_pct": round(fc["exceedance_rate"].iloc[0] * 100, 1),
                "model_mape_pct": metrics["model_mape"],
                "baseline_mape_pct": metrics["baseline_mape"],
            }
        )

    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    zone_risk_df = pd.DataFrame(zone_risk_rows).sort_values(
        "capacity_usage_pct", ascending=False
    )

    return forecasts_df, zone_risk_df, all_metrics
