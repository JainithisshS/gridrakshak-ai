"""
GridRakshak AI -- Real Dataset Loader (v2 — corrected scaling)
Downloads UCI Individual Household Electric Power Consumption dataset
and converts it to GridRakshak zone format for validation.

Dataset: UCI ML Repository (public, no auth needed)
- 2M+ rows, 1-min intervals, 2006-2010
- Zones treated as BESCOM-style feeders scaled to realistic kWh range
- Fraud injected post-download so we have KNOWN ground truth labels

Scaling note:
  UCI global_kwh after 15-min SUM ≈ 0.27 kWh (one household)
  BESCOM feeder aggregates 10+ households → target range 40-80 kWh per 15-min
  Scale factor = target_mean / uci_mean  ≈ 185 (no extra *15 multiply)
"""

import urllib.request
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path


UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00235/household_power_consumption.zip"
)

# BESCOM-style zone names for real data dashboard
ZONE_MAP = {
    "Z_A": {"zone_name": "Indiranagar Feeder",  "capacity_kw": 500,  "area_type": "mixed"},
    "Z_B": {"zone_name": "Koramangala Feeder",  "capacity_kw": 450,  "area_type": "residential"},
    "Z_C": {"zone_name": "Whitefield Feeder",   "capacity_kw": 600,  "area_type": "commercial"},
    "Z_D": {"zone_name": "Jayanagar Feeder",    "capacity_kw": 520,  "area_type": "residential"},
}

FRAUD_TYPES = ["sudden_drop", "under_reporting", "periodic_spike", "zone_mismatch"]

# Target kWh per 15-min per zone (realistic BESCOM feeder aggregate)
TARGET_MEANS = {"Z_A": 48.0, "Z_B": 42.0, "Z_C": 65.0, "Z_D": 50.0}


# ── Download & Parse ──────────────────────────────────────────────────────────

def download_uci(data_dir: Path) -> pd.DataFrame:
    """Download and parse UCI dataset; cache as parquet."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path  = data_dir / "household_power_consumption.zip"
    csv_cache = data_dir / "uci_raw.parquet"

    if csv_cache.exists():
        print("  Using cached UCI dataset.")
        return pd.read_parquet(csv_cache)

    if not zip_path.exists():
        print("  Downloading UCI Household dataset (~20 MB)...")
        urllib.request.urlretrieve(UCI_URL, zip_path)
        print("  Download complete.")

    print("  Parsing UCI dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("household_power_consumption.txt") as f:
            df = pd.read_csv(f, sep=";", na_values=["?"], low_memory=False)

    # Build timestamp (pandas 2.x dropped combined parse_dates)
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], dayfirst=True, format="mixed"
    )
    df = df.drop(columns=["Date", "Time"])

    key_cols = ["Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df = df.dropna(subset=key_cols).copy()

    # Global_active_power is kW (instantaneous)
    # Energy per 1-min interval = kW * (1/60) h = kWh
    df["global_kwh"] = df["Global_active_power"].astype(float) / 60.0
    # Sub_metering_X is in Wh per minute → kWh per minute
    df["sub1_kwh"]   = df["Sub_metering_1"].astype(float) / 1000.0
    df["sub2_kwh"]   = df["Sub_metering_2"].astype(float) / 1000.0
    df["sub3_kwh"]   = df["Sub_metering_3"].astype(float) / 1000.0
    # General load not captured by sub-meters
    sub_kw           = (df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"]).astype(float) / 1000.0
    df["general_kwh"] = ((df["Global_active_power"].astype(float) / 60.0) - sub_kw).clip(lower=0)

    df = df[["timestamp", "global_kwh", "sub1_kwh", "sub2_kwh", "sub3_kwh", "general_kwh"]]
    df.to_parquet(csv_cache, index=False)
    print(f"  Parsed {len(df):,} rows → cached to parquet.")
    return df


# ── Resample to 15-min and create zones ──────────────────────────────────────

def build_zone_format(df: pd.DataFrame, use_days: int = 150) -> tuple:
    """
    Resample 1-min readings to 15-min intervals.
    Scale to realistic BESCOM feeder kWh range using linear scaling.
    Returns (readings, zones, meters).
    """
    df = df.set_index("timestamp").sort_index()
    cutoff = df.index.max() - pd.Timedelta(days=use_days)
    df = df[df.index >= cutoff].copy()

    # Resample: sum 15 one-minute kWh → total kWh over 15 minutes
    # NOTE: NO additional * 15 — the sum IS already kWh
    resample = df.resample("15min").sum().dropna(how="all")
    resample = resample.reset_index()
    if "timestamp" not in resample.columns:
        resample = resample.rename(columns={resample.columns[0]: "timestamp"})

    # Compute actual mean of global_kwh after resampling
    uci_mean = resample["global_kwh"].mean()
    print(f"  UCI 15-min mean: {uci_mean:.4f} kWh → scaling to BESCOM feeder range")

    rows = []
    for zone_id, target_mean in TARGET_MEANS.items():
        tmp = resample[["timestamp", "global_kwh"]].copy()

        # Scale so zone mean matches BESCOM target + zone-specific noise (±5%)
        scale = target_mean / max(uci_mean, 0.01)
        rng_z = np.random.default_rng(abs(hash(zone_id)) % 10000)
        noise = 1.0 + rng_z.normal(0, 0.05, len(tmp))

        tmp["kwh"]      = (tmp["global_kwh"] * scale * noise).clip(lower=0)
        tmp["zone_id"]  = zone_id
        tmp["meter_id"] = f"REAL_{zone_id}_M01"
        rows.append(tmp[["timestamp", "kwh", "zone_id", "meter_id"]])

    readings = pd.concat(rows, ignore_index=True)
    readings["kwh"]        = readings["kwh"].clip(lower=0)
    readings["is_missing"] = False

    # Verify scaling
    for zone_id in TARGET_MEANS:
        actual = readings[readings["zone_id"] == zone_id]["kwh"].mean()
        print(f"  {zone_id} ({ZONE_MAP[zone_id]['zone_name']}): mean={actual:.2f} kWh/15-min (target={TARGET_MEANS[zone_id]})")

    zones = pd.DataFrame([{"zone_id": zid, **info} for zid, info in ZONE_MAP.items()])
    zones = zones.rename(columns={"capacity_kw": "feeder_capacity_kw"})

    meters = pd.DataFrame([{
        "meter_id":    f"REAL_{zid}_M01",
        "zone_id":     zid,
        "zone_name":   ZONE_MAP[zid]["zone_name"],
        "area_type":   ZONE_MAP[zid]["area_type"],
        "base_scale":  1.0,
        "fraud_type":  "none",
        "is_fraud":    False,
        "fraud_label": 0,
        "missing_pct": 0.0,
    } for zid in ZONE_MAP])

    return readings, zones, meters


# ── Inject known fraud into real data ─────────────────────────────────────────

def inject_fraud_into_real(
    readings: pd.DataFrame,
    meters: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple:
    """
    Inject 4 known fraud patterns into the real UCI consumption data.
    Fraud is more subtle on real data (closer to what BESCOM would see).
    Returns (augmented_readings, augmented_meters).
    """
    new_readings = readings.copy()
    new_meters   = meters.copy()
    all_meter_ids = readings["meter_id"].unique().tolist()

    fraud_meters = {}
    for i, ftype in enumerate(FRAUD_TYPES):
        source_mid = all_meter_ids[i % len(all_meter_ids)]
        new_mid    = f"REAL_FRAUD_{ftype.upper()[:3]}_M01"

        clone = readings[readings["meter_id"] == source_mid].copy()
        clone["meter_id"] = new_mid
        kwh = clone["kwh"].values.copy()
        n   = len(kwh)

        if ftype == "sudden_drop":
            # 30-45 day sustained drop (40-55% reduction)
            s = rng.integers(int(n * 0.20), int(n * 0.40))
            e = min(s + int(n * 0.25), n)
            kwh[s:e] *= rng.uniform(0.42, 0.56)

        elif ftype == "under_reporting":
            # Consistent 8-18% under-read (subtle, realistic meter bypass)
            kwh *= rng.uniform(0.82, 0.92)

        elif ftype == "periodic_spike":
            # Unauthorized equipment at 2-4 AM — capped at 3-4x (not 6x)
            night_idx = [j for j, ts in enumerate(clone["timestamp"])
                         if pd.Timestamp(ts).hour in [2, 3, 4]]
            count = max(1, len(night_idx) // 8)
            if night_idx:
                chosen = rng.choice(night_idx, size=min(count, len(night_idx)), replace=False)
                for idx in chosen:
                    for d in range(min(4, n - idx)):
                        kwh[idx + d] = min(kwh[idx + d] * rng.uniform(3.0, 4.0),
                                           TARGET_MEANS.get(clone["zone_id"].iloc[0], 65) * 5)

        elif ftype == "zone_mismatch":
            # Use different zone's pattern (different daily shape)
            other_mid = all_meter_ids[(i + 2) % len(all_meter_ids)]
            other_kwh = readings[readings["meter_id"] == other_mid]["kwh"].values
            kwh = other_kwh[:n] if len(other_kwh) >= n else np.pad(other_kwh, (0, n - len(other_kwh)), mode="wrap")

        clone = clone.copy()
        clone["kwh"] = kwh
        new_readings = pd.concat([new_readings, clone], ignore_index=True)

        zone_id = clone["zone_id"].iloc[0]
        new_meters = pd.concat([new_meters, pd.DataFrame([{
            "meter_id":    new_mid,
            "zone_id":     zone_id,
            "zone_name":   ZONE_MAP.get(zone_id, {}).get("zone_name", zone_id),
            "area_type":   ZONE_MAP.get(zone_id, {}).get("area_type", "residential"),
            "base_scale":  1.0,
            "fraud_type":  ftype,
            "is_fraud":    True,
            "fraud_label": 1,
            "missing_pct": 0.0,
        }])], ignore_index=True)

        fraud_meters[new_mid] = ftype

    print(f"  Injected {len(FRAUD_TYPES)} fraud meters into real UCI baseline")
    for mid, ft in fraud_meters.items():
        print(f"    {mid}: {ft}")

    return new_readings, new_meters


# ── Main entry point ──────────────────────────────────────────────────────────

def load_real_dataset(raw_dir: Path, use_days: int = 150) -> tuple:
    """Full pipeline: load → resample → scale → inject fraud."""
    rng = np.random.default_rng(42)

    df = download_uci(raw_dir)
    readings, zones, meters = build_zone_format(df, use_days=use_days)
    readings, meters = inject_fraud_into_real(readings, meters, rng)

    print(f"  Real dataset: {len(readings):,} rows | "
          f"{readings['meter_id'].nunique()} meters | "
          f"{meters['is_fraud'].sum()} fraud")
    return readings, zones, meters


if __name__ == "__main__":
    r, z, m = load_real_dataset(Path("data/real"))
    print("\nZone stats:")
    print(r.groupby("zone_id")["kwh"].agg(["mean", "std", "min", "max"]).round(2))
    print(z)
    print(m)
