"""
GridRakshak AI -- Anomaly & Theft Detection Engine (v2)

Three-layer detection stack:
  Layer 1 - Statistical: Rolling Z-score residuals + persistence filter
  Layer 2 - ML-based:   Isolation Forest on per-meter daily features
  Layer 3 - Peer-based: Zone-relative consumption deviation + load shape dissimilarity (Euclidean DTW proxy)

Composite risk score fuses all three layers with calibrated weights.
False positive reduction via multi-signal confirmation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ── Thresholds ────────────────────────────────────────────────────────────────
ZSCORE_THRESHOLD   = 2.5     # Rolling Z-score to flag a window
PERSISTENCE_SLOTS  = 12      # >= 3 hours (12 x 15min) of continuous flagging
ROLLING_WINDOW     = 96      # 24h rolling window (96 x 15-min slots)
MIN_HISTORY_DAYS   = 7       # Min days before using meter-specific baseline
IF_CONTAMINATION   = 0.15    # Expected fraction of anomalies for IsolationForest
DTW_OUTLIER_PCT    = 85      # Euclidean peer distance percentile for outlier flag


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Statistical Residual Detection
# ══════════════════════════════════════════════════════════════════════════════

def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-meter residuals vs 7-day rolling expectation (or zone baseline for cold-start)."""
    df = df.copy().sort_values(["meter_id", "timestamp"])

    grp = df.groupby("meter_id")["kwh"]
    df["expected"] = grp.transform(lambda s: s.rolling(672, min_periods=96).mean())

    # Cold-start fallback: zone-hour mean
    zone_hour_mean = df.groupby(["zone_id", "hour"])["kwh"].transform("mean")
    df["expected"] = df["expected"].fillna(zone_hour_mean)
    df["expected"] = df["expected"].fillna(df["kwh"].mean())

    df["residual"] = df["kwh"] - df["expected"]
    df["residual_pct"] = df["residual"] / (df["expected"].abs() + 1e-6)
    return df


def _rolling_zscore_flag(s: pd.Series) -> pd.Series:
    roll_mean = s.rolling(ROLLING_WINDOW, min_periods=24).mean()
    roll_std  = s.rolling(ROLLING_WINDOW, min_periods=24).std().fillna(1e-6)
    z = (s - roll_mean) / roll_std
    return (z.abs() > ZSCORE_THRESHOLD).astype(int)


def _low_consumption_flag(s: pd.Series) -> pd.Series:
    """Flag windows where consumption is < 30% of the 7-day rolling mean (bypass/under-reporting)."""
    roll7 = s.rolling(672, min_periods=96).mean()
    ratio = s / (roll7 + 1e-6)
    return (ratio < 0.30).astype(int)


def _persistence_filter(flag: pd.Series, min_run: int) -> pd.Series:
    """Keep only anomaly flags that persist for at least min_run consecutive slots."""
    result = np.zeros(len(flag), dtype=int)
    arr = flag.values
    i = 0
    while i < len(arr):
        if arr[i] == 1:
            j = i
            while j < len(arr) and arr[j] == 1:
                j += 1
            run_len = j - i
            if run_len >= min_run:
                result[i:j] = 1
            i = j
        else:
            i += 1
    return pd.Series(result, index=flag.index)


def layer1_statistical(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-reading anomaly flags from statistical layer."""
    df = df.copy().sort_values(["meter_id", "timestamp"])

    df["flag_zscore"] = df.groupby("meter_id")["residual"].transform(_rolling_zscore_flag)
    df["flag_low"]    = df.groupby("meter_id")["kwh"].transform(_low_consumption_flag)
    df["flag_any"]    = df[["flag_zscore", "flag_low"]].max(axis=1)

    df["persistent_anomaly"] = df.groupby("meter_id")["flag_any"].transform(
        lambda s: _persistence_filter(s.reset_index(drop=True), PERSISTENCE_SLOTS).values
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Isolation Forest on Per-Meter Daily Features
# ══════════════════════════════════════════════════════════════════════════════

def extract_meter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one feature vector per meter from the full history.
    Features are designed to separate fraud from normal meters:
    - Mean, std, coefficient of variation (CV)
    - Peak-to-valley ratio (detects artificial flatness or drops)
    - Day-over-day autocorrelation (fraud breaks temporal consistency)
    - Fraction of readings below 20% of zone mean (sustained low)
    - Fraction of readings above 300% of zone mean (periodic spikes)
    - Skewness and kurtosis of residual distribution
    - Night-to-day ratio (unusual if meter is bypassed at night)
    """
    records = []
    zone_means = df.groupby("zone_id")["kwh"].transform("mean")
    df = df.copy()
    df["zone_mean"] = zone_means

    for meter_id, grp in df.groupby("meter_id"):
        grp = grp.sort_values("timestamp")
        kwh = grp["kwh"].values
        res = grp["residual"].values if "residual" in grp.columns else kwh - kwh.mean()
        zone_mean = grp["zone_mean"].mean()

        mean_kwh   = kwh.mean()
        std_kwh    = kwh.std() + 1e-6
        cv         = std_kwh / (mean_kwh + 1e-6)
        p10, p90   = np.percentile(kwh, 10), np.percentile(kwh, 90)
        peak_valley = (p90 + 1e-6) / (p10 + 1e-6)

        # Day-over-day autocorrelation (lag-96 slots = 24h)
        if len(kwh) > 192:
            autocorr = np.corrcoef(kwh[:-96], kwh[96:])[0, 1]
        else:
            autocorr = 0.5

        frac_very_low  = (kwh < 0.20 * zone_mean).mean()
        frac_very_high = (kwh > 3.00 * zone_mean).mean()

        res_skew  = stats.skew(res)
        res_kurt  = stats.kurtosis(res)

        # Night (23:00-05:00) vs day (09:00-21:00) ratio
        is_night = ((grp["hour"] >= 23) | (grp["hour"] < 5)).values
        is_day   = ((grp["hour"] >= 9) & (grp["hour"] < 21)).values
        night_mean = kwh[is_night].mean() if is_night.sum() > 0 else mean_kwh
        day_mean   = kwh[is_day].mean()   if is_day.sum() > 0   else mean_kwh
        night_day_ratio = night_mean / (day_mean + 1e-6)

        # Zone deviation
        zone_z = (mean_kwh - zone_mean) / (df[df["zone_id"] == grp["zone_id"].iloc[0]]["kwh"].std() + 1e-6)

        records.append({
            "meter_id":       meter_id,
            "zone_id":        grp["zone_id"].iloc[0],
            "mean_kwh":       mean_kwh,
            "cv":             cv,
            "peak_valley":    peak_valley,
            "autocorr":       autocorr,
            "frac_very_low":  frac_very_low,
            "frac_very_high": frac_very_high,
            "res_skew":       res_skew,
            "res_kurt":       res_kurt,
            "night_day_ratio": night_day_ratio,
            "zone_z":         zone_z,
        })

    return pd.DataFrame(records)


IFOREST_FEATURES = [
    "cv", "peak_valley", "autocorr", "frac_very_low",
    "frac_very_high", "res_skew", "res_kurt", "night_day_ratio", "zone_z"
]


def layer2_isolation_forest(meter_features: pd.DataFrame) -> pd.DataFrame:
    """
    Run Isolation Forest per zone so each meter is compared to its peers.
    Returns meter_features with 'if_score' (higher = more anomalous, 0-1).
    """
    meter_features = meter_features.copy()
    meter_features["if_score"] = 0.0

    for zone_id, grp in meter_features.groupby("zone_id"):
        X = grp[IFOREST_FEATURES].fillna(0).values
        if len(X) < 4:
            # Not enough meters to run IF in this zone
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iforest = IsolationForest(
            n_estimators=200,
            contamination=min(IF_CONTAMINATION, (len(X) - 1) / len(X)),
            random_state=42,
            n_jobs=-1,
        )
        iforest.fit(X_scaled)

        # score_samples: more negative = more anomalous; invert to 0-1
        raw_scores = iforest.score_samples(X_scaled)
        # Normalise to [0,1]: 1 = most anomalous
        min_s, max_s = raw_scores.min(), raw_scores.max()
        normalised = 1 - (raw_scores - min_s) / (max_s - min_s + 1e-9)

        meter_features.loc[grp.index, "if_score"] = normalised

    return meter_features


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Peer Comparison (Euclidean Load-Shape Distance)
# ══════════════════════════════════════════════════════════════════════════════

def layer3_peer_distance(df: pd.DataFrame, sample_days: int = 14) -> pd.Series:
    """
    Euclidean distance of each meter's average 96-slot daily profile
    against its zone peers. Returns Series[meter_id -> normalised_dist 0-1].
    """
    last_ts = df["timestamp"].max()
    cutoff  = last_ts - pd.Timedelta(days=sample_days)
    recent  = df[df["timestamp"] >= cutoff].copy()

    recent["slot"] = recent["hour"] * 4 + (recent["timestamp"].dt.minute // 15)

    # Average profile per meter
    profiles = (
        recent.groupby(["meter_id", "zone_id", "slot"])["kwh"]
        .mean()
        .reset_index()
        .pivot(index=["meter_id", "zone_id"], columns="slot", values="kwh")
        .fillna(0)
        .reset_index()
    )

    feature_cols = [c for c in profiles.columns if isinstance(c, (int, np.integer))]
    dtw_dists = {}

    for _, row in profiles.iterrows():
        mid     = row["meter_id"]
        zone_id = row["zone_id"]
        seq     = row[feature_cols].values.astype(float)

        peers = profiles[profiles["zone_id"] == zone_id]
        peers = peers[peers["meter_id"] != mid]

        if len(peers) == 0:
            dtw_dists[mid] = 0.0
            continue

        peer_seqs = peers[feature_cols].values.astype(float)
        # Normalised euclidean distances to all zone peers
        dists = np.linalg.norm(peer_seqs - seq, axis=1) / (np.linalg.norm(seq) + 1e-6)
        dtw_dists[mid] = float(np.mean(dists))

    dist_series = pd.Series(dtw_dists, name="peer_dist")
    # Normalise 0-1 across all meters
    d_min, d_max = dist_series.min(), dist_series.max()
    dist_series = (dist_series - d_min) / (d_max - d_min + 1e-9)
    return dist_series


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORING & ALERT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_layer1_per_meter(flagged_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-reading flags to per-meter Layer-1 scores."""
    stats_df = (
        flagged_df.groupby("meter_id")
        .agg(
            zone_id=("zone_id", "first"),
            total_readings=("kwh", "count"),
            anomaly_count=("persistent_anomaly", "sum"),
            low_flag_count=("flag_low", "sum"),
            zscore_flag_count=("flag_zscore", "sum"),
            mean_kwh=("kwh", "mean"),
            std_kwh=("kwh", "std"),
        )
        .reset_index()
    )
    stats_df["anomaly_rate"]   = stats_df["anomaly_count"]   / stats_df["total_readings"]
    stats_df["low_flag_rate"]  = stats_df["low_flag_count"]  / stats_df["total_readings"]
    stats_df["zscore_rate"]    = stats_df["zscore_flag_count"] / stats_df["total_readings"]

    # Layer-1 score: max of the two sub-signals
    stats_df["l1_score"] = stats_df[["anomaly_rate", "low_flag_rate"]].max(axis=1).clip(0, 1)
    return stats_df


def generate_inspection_alerts(
    l1_df: pd.DataFrame,
    meter_features: pd.DataFrame,   # has if_score
    peer_dist: pd.Series,
    meters_meta: pd.DataFrame,
    zone_names: dict,
) -> pd.DataFrame:
    """Fuse all three layers into a composite risk score and produce ranked alerts."""

    alerts = l1_df.merge(
        meter_features[["meter_id", "if_score", "cv", "autocorr",
                         "frac_very_low", "frac_very_high", "zone_z",
                         "peak_valley", "night_day_ratio"]],
        on="meter_id", how="left"
    )

    # Merge peer distance
    peer_df = peer_dist.reset_index()
    peer_df.columns = ["meter_id", "peer_score"]
    alerts = alerts.merge(peer_df, on="meter_id", how="left")
    alerts["peer_score"] = alerts["peer_score"].fillna(0)
    alerts["if_score"]   = alerts["if_score"].fillna(0)

    # ── Composite score (calibrated weights) ─────────────────────────────────
    # Layer 1 (statistical)   : 35%
    # Layer 2 (Isolation Forest): 40%
    # Layer 3 (peer shape)    : 25%
    alerts["composite_score"] = (
        0.35 * alerts["l1_score"]
        + 0.40 * alerts["if_score"]
        + 0.25 * alerts["peer_score"]
    ).clip(0, 1)

    # ── Multi-signal confirmation boost ──────────────────────────────────────
    # A meter flagged by 2+ layers gets a confidence boost
    alerts["signals_triggered"] = (
        (alerts["l1_score"]   > 0.10).astype(int) +
        (alerts["if_score"]   > 0.60).astype(int) +
        (alerts["peer_score"] > 0.50).astype(int)
    )
    boost = np.where(alerts["signals_triggered"] >= 2, 0.10, 0.0)
    alerts["composite_score"] = (alerts["composite_score"] + boost).clip(0, 1)

    # ── Alert tier ────────────────────────────────────────────────────────────
    alerts["alert_tier"] = pd.cut(
        alerts["composite_score"],
        bins=[-0.001, 0.35, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
    )

    # ── Reason code generation ────────────────────────────────────────────────
    def reason_code(row):
        reasons = []
        if row["frac_very_low"] > 0.15:
            reasons.append(
                f"Consumption below 20% of zone mean for {int(row['frac_very_low']*100)}% of time (possible bypass)"
            )
        if row["low_flag_rate"] > 0.20:
            reasons.append(
                f"Sustained low consumption: {int(row['low_flag_rate']*100)}% readings flagged (under-reporting risk)"
            )
        if row["frac_very_high"] > 0.02:
            reasons.append(
                f"Periodic high spikes: {int(row['frac_very_high']*100)}% readings exceed 3x zone mean (illegal equipment risk)"
            )
        if row["autocorr"] < 0.40:
            reasons.append(
                f"Low temporal consistency (autocorr={row['autocorr']:.2f}) — consumption pattern irregular"
            )
        if row["peer_score"] > 0.65:
            reasons.append("Load shape significantly different from zone peers (feeder mapping anomaly possible)")
        if abs(row.get("zone_z", 0)) > 2.0:
            direction = "below" if row["zone_z"] < 0 else "above"
            reasons.append(
                f"Mean consumption {abs(row['zone_z']):.1f}σ {direction} zone average"
            )
        if row["if_score"] > 0.75:
            reasons.append("Isolation Forest: meter profile classified as statistical outlier in zone peer group")
        if not reasons:
            reasons.append("Minor multi-signal deviation from expected consumption pattern")
        return "; ".join(reasons[:3])   # top 3 reasons max

    alerts["reason_codes"] = alerts.apply(reason_code, axis=1)
    alerts["zone_name"]    = alerts["zone_id"].map(zone_names)

    # Rank by composite score
    alerts = alerts.sort_values("composite_score", ascending=False).reset_index(drop=True)
    alerts["priority_rank"] = alerts.index + 1

    output_cols = [
        "priority_rank", "meter_id", "zone_id", "zone_name",
        "alert_tier", "composite_score", "l1_score", "if_score", "peer_score",
        "signals_triggered", "anomaly_rate", "low_flag_rate",
        "frac_very_low", "frac_very_high", "autocorr", "reason_codes",
    ]
    return alerts[output_cols].rename(columns={"composite_score": "risk_score"})


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_anomaly_detection(
    meter_df: pd.DataFrame,
    meters_meta: pd.DataFrame,
    zone_names: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full 3-layer anomaly detection pipeline. Returns (alerts_df, flagged_df)."""

    print("  [L1] Computing residuals...")
    df = compute_residuals(meter_df)

    print("  [L1] Rolling Z-score + low-consumption flagging...")
    df = layer1_statistical(df)

    print("  [L1] Aggregating per-meter Layer-1 scores...")
    l1_df = compute_layer1_per_meter(df)

    print("  [L2] Extracting per-meter feature vectors...")
    meter_features = extract_meter_features(df)

    print("  [L2] Isolation Forest (per-zone)...")
    meter_features = layer2_isolation_forest(meter_features)

    print("  [L3] Peer load-shape distance (Euclidean)...")
    peer_dist = layer3_peer_distance(meter_df, sample_days=14)

    print("  Generating composite inspection alerts...")
    alerts = generate_inspection_alerts(
        l1_df, meter_features, peer_dist, meters_meta, zone_names
    )

    high  = (alerts["alert_tier"] == "High").sum()
    med   = (alerts["alert_tier"] == "Medium").sum()
    multi = (alerts["signals_triggered"] >= 2).sum()
    print(f"  Alerts: {high} High | {med} Medium | {multi} confirmed by 2+ signals")

    return alerts, df
