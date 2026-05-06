"""
GridRakshak AI — Data Preprocessor
Handles missing data imputation, outlier clipping, and feature engineering
for smart meter readings.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    readings = pd.read_csv(data_dir / "meter_readings.csv", parse_dates=["timestamp"])
    zones = pd.read_csv(data_dir / "zone_metadata.csv")
    meters = pd.read_csv(data_dir / "meters_metadata.csv")
    return readings, zones, meters


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill then backward-fill missing readings per meter.
    Remaining NaNs (at boundaries) are replaced with per-meter median.
    """
    df = df.copy()
    df = df.sort_values(["meter_id", "timestamp"])

    # Forward fill within each meter group
    df["kwh"] = df.groupby("meter_id")["kwh"].transform(
        lambda s: s.ffill().bfill()
    )

    # Any residual NaN → replace with meter median
    medians = df.groupby("meter_id")["kwh"].transform("median")
    df["kwh"] = df["kwh"].fillna(medians)

    df["is_missing"] = df["is_missing"].fillna(False)
    print(f"  Imputed missing values. Remaining NaN: {df['kwh'].isna().sum()}")
    return df


def clip_outliers(df: pd.DataFrame, iqr_factor: float = 4.0) -> pd.DataFrame:
    """IQR-based outlier clipping per meter (very conservative to keep genuine spikes)."""
    df = df.copy()

    def clip_series(s):
        q1, q3 = s.quantile(0.05), s.quantile(0.95)
        iqr = q3 - q1
        lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        return s.clip(lower=max(lo, 0), upper=hi)

    df["kwh"] = df.groupby("meter_id")["kwh"].transform(clip_series)
    return df


def engineer_features(df: pd.DataFrame, meters_meta: pd.DataFrame = None) -> pd.DataFrame:
    """Add time-based, lag, and area-type features for model training."""
    df = df.copy().sort_values(["meter_id", "timestamp"])

    # Encode area type as numeric (0=residential, 1=mixed, 2=commercial, 3=industrial)
    AREA_CODES = {"residential": 0, "mixed": 1, "commercial": 2, "industrial": 3}
    if meters_meta is not None and "area_type" in meters_meta.columns:
        area_map = dict(zip(meters_meta["meter_id"], meters_meta["area_type"].map(AREA_CODES).fillna(0)))
        df["area_type_code"] = df["meter_id"].map(area_map).fillna(0).astype(int)
    else:
        df["area_type_code"] = 0

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["hour_of_day"] = ts.dt.hour + ts.dt.minute / 60  # float hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding — prevents model treating hour 23 and 0 as far apart
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Peak / sleeping flags (crucial for household forecasting)
    df["is_peak_hour"]     = df["hour"].isin([7, 8, 9, 18, 19, 20, 21]).astype(int)
    df["is_sleeping_hour"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    # ── Lag features (per meter) ───────────────────────────────────────────────
    # 15-min slot index
    df["slot"] = df["hour"] * 4 + df["minute"] // 15

    grp = df.groupby("meter_id")["kwh"]
    df["lag_1h"] = grp.shift(4)       # 1 hour ago
    df["lag_24h"] = grp.shift(96)     # same time yesterday
    df["lag_168h"] = grp.shift(672)   # same time last week
    df["roll_24h_mean"] = grp.transform(lambda s: s.rolling(96, min_periods=1).mean())
    df["roll_24h_std"] = grp.transform(lambda s: s.rolling(96, min_periods=1).std().fillna(0))
    df["roll_7d_mean"] = grp.transform(lambda s: s.rolling(672, min_periods=1).mean())

    # Fill lag NaNs with rolling mean
    for col in ["lag_1h", "lag_24h", "lag_168h"]:
        df[col] = df[col].fillna(df["roll_24h_mean"])

    return df


def build_zone_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate meter-level data to zone-level hourly demand."""
    df_hourly = (
        df.set_index("timestamp")
        .groupby(["zone_id", pd.Grouper(freq="1h")])["kwh"]
        .sum()
        .reset_index()
        .rename(columns={"kwh": "zone_kwh"})
    )

    ts = df_hourly["timestamp"]
    df_hourly["hour"]        = ts.dt.hour
    df_hourly["day_of_week"] = ts.dt.dayofweek
    df_hourly["month"]       = ts.dt.month
    df_hourly["day_of_year"] = ts.dt.dayofyear
    df_hourly["is_weekend"]  = (df_hourly["day_of_week"] >= 5).astype(int)

    # Cyclical encoding
    df_hourly["hour_sin"] = np.sin(2 * np.pi * df_hourly["hour"] / 24)
    df_hourly["hour_cos"] = np.cos(2 * np.pi * df_hourly["hour"] / 24)
    df_hourly["dow_sin"]  = np.sin(2 * np.pi * df_hourly["day_of_week"] / 7)
    df_hourly["dow_cos"]  = np.cos(2 * np.pi * df_hourly["day_of_week"] / 7)

    # Peak / sleeping flags
    df_hourly["is_peak_hour"]     = df_hourly["hour"].isin([7, 8, 9, 18, 19, 20, 21]).astype(int)
    df_hourly["is_sleeping_hour"] = df_hourly["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    grp = df_hourly.groupby("zone_id")["zone_kwh"]
    df_hourly["lag_24h"]       = grp.shift(24)
    df_hourly["lag_168h"]      = grp.shift(168)
    df_hourly["roll_24h_mean"] = grp.transform(lambda s: s.rolling(24,  min_periods=1).mean())
    df_hourly["roll_24h_std"]  = grp.transform(lambda s: s.rolling(24,  min_periods=1).std().fillna(0))
    df_hourly["roll_7d_mean"]  = grp.transform(lambda s: s.rolling(168, min_periods=1).mean())

    for col in ["lag_24h", "lag_168h"]:
        df_hourly[col] = df_hourly[col].fillna(df_hourly["roll_24h_mean"])

    # Area type code per zone (dominant area type)
    zone_area = df.groupby("zone_id")["area_type_code"].agg(lambda x: x.mode()[0]).to_dict() \
        if "area_type_code" in df.columns else {}
    df_hourly["area_type_code"] = df_hourly["zone_id"].map(zone_area).fillna(0).astype(int)

    df_hourly = df_hourly.dropna(subset=["zone_kwh"])
    return df_hourly


def preprocess(data_dir: Path, processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline. Returns processed meter df, zone hourly df, zone meta, meter meta."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    readings, zones, meters = load_raw(data_dir)
    print(f"  Raw readings: {len(readings):,} rows")

    print("Imputing missing values...")
    readings = impute_missing(readings)

    print("Clipping outliers...")
    readings = clip_outliers(readings)

    print("Engineering features...")
    readings = engineer_features(readings, meters_meta=meters)

    print("Building zone aggregates...")
    zone_hourly = build_zone_aggregates(readings)

    # Save processed data
    readings.to_csv(processed_dir / "meter_readings_processed.csv", index=False)
    zone_hourly.to_csv(processed_dir / "zone_hourly.csv", index=False)
    print(f"  Zone hourly: {len(zone_hourly):,} rows")

    return readings, zone_hourly, zones, meters
