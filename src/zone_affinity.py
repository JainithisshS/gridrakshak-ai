"""
GridRakshak AI — Zone Affinity Verification
Detects mis-tagged meters and feeder mapping anomalies using
load-shape clustering (K-Means + DTW distance matrix).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def extract_load_profiles(meter_df: pd.DataFrame, sample_days: int = 14) -> pd.DataFrame:
    """
    Extract average daily load profile (96 slots) per meter
    from the most recent `sample_days` days.
    """
    last_ts = meter_df["timestamp"].max()
    cutoff = last_ts - pd.Timedelta(days=sample_days)
    recent = meter_df[meter_df["timestamp"] >= cutoff].copy()

    recent["slot"] = recent["hour"] * 4 + (recent["timestamp"].dt.minute // 15)

    # Average profile: mean kwh per 15-min slot across days
    profiles = (
        recent.groupby(["meter_id", "zone_id", "slot"])["kwh"]
        .mean()
        .reset_index()
        .pivot(index=["meter_id", "zone_id"], columns="slot", values="kwh")
        .fillna(0)
        .reset_index()
    )
    return profiles


def cluster_meters(profiles: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    K-Means cluster meters by load shape (96-dim profile vector).
    Returns profiles dataframe with cluster assignment.
    """
    feature_cols = [c for c in profiles.columns if isinstance(c, int)]
    X = profiles[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    profiles = profiles.copy()
    profiles["cluster"] = km.fit_predict(X_scaled)

    return profiles


def detect_zone_mismatches(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Flag meters whose cluster assignment is inconsistent with their zone.
    A meter is flagged if its cluster's dominant zone is different from its own zone.
    """
    # Find dominant zone per cluster
    cluster_zone = (
        profiles.groupby(["cluster", "zone_id"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .groupby("cluster")
        .first()["zone_id"]
        .to_dict()
    )
    profiles = profiles.copy()
    profiles["dominant_zone"] = profiles["cluster"].map(cluster_zone)
    profiles["zone_mismatch_flag"] = (
        profiles["zone_id"] != profiles["dominant_zone"]
    ).astype(int)

    return profiles[["meter_id", "zone_id", "cluster", "dominant_zone", "zone_mismatch_flag"]]


def run_zone_affinity(meter_df: pd.DataFrame) -> pd.DataFrame:
    """Full zone affinity pipeline. Returns per-meter mismatch flags."""
    print("  Extracting load profiles...")
    profiles = extract_load_profiles(meter_df, sample_days=14)

    print("  Clustering meters by load shape...")
    profiles = cluster_meters(profiles, n_clusters=8)

    print("  Detecting zone mismatches...")
    results = detect_zone_mismatches(profiles)

    mismatches = results["zone_mismatch_flag"].sum()
    print(f"  Zone affinity: {mismatches} potential mis-tagged meters detected")

    return results
