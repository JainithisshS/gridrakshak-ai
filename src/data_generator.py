"""
GridRakshak AI -- Realistic Smart Meter Data Generator (v2)

Simulates BESCOM Bangalore distribution network with:
  - 8 real Bangalore zones with actual feeder capacities
  - Temperature-driven demand (AC load, seasonal variation)
  - Area-type specific profiles (residential / commercial / industrial / mixed)
  - Indian time-of-day patterns (morning chai, afternoon lull, evening peak)
  - Festival load spikes (Diwali, Ugadi, summer)
  - Communication blackout gaps (batch missing, not random)
  - Phase imbalance / voltage fluctuation noise
  - 4 fraud archetypes with realistic timing and depth
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
SEED             = 42
DAYS             = 90
READINGS_PER_DAY = 96          # 15-min intervals
START_DATE       = "2025-11-01"  # Nov 2025 - Jan 2026 (winter → new year)
FRAUD_FRACTION   = 0.16

# ── Real Bangalore BESCOM Zones ────────────────────────────────────────────────
ZONES = [
    {"zone_id": "Z01", "zone_name": "Jayanagar",    "capacity_kw": 850,  "area_type": "residential"},
    {"zone_id": "Z02", "zone_name": "Koramangala",  "capacity_kw": 1400, "area_type": "commercial"},
    {"zone_id": "Z03", "zone_name": "Indiranagar",  "capacity_kw": 1100, "area_type": "mixed"},
    {"zone_id": "Z04", "zone_name": "Whitefield",   "capacity_kw": 1800, "area_type": "industrial"},
    {"zone_id": "Z05", "zone_name": "Rajajinagar",  "capacity_kw": 750,  "area_type": "residential"},
    {"zone_id": "Z06", "zone_name": "HSR Layout",   "capacity_kw": 950,  "area_type": "mixed"},
    {"zone_id": "Z07", "zone_name": "Malleshwaram", "capacity_kw": 700,  "area_type": "residential"},
    {"zone_id": "Z08", "zone_name": "Electronic City","capacity_kw":2200, "area_type": "industrial"},
]
NUM_ZONES        = len(ZONES)
METERS_PER_ZONE  = 10
TOTAL_METERS     = NUM_ZONES * METERS_PER_ZONE

FRAUD_TYPES = ["sudden_drop", "periodic_spike", "under_reporting", "zone_mismatch"]

# ── Bangalore Monthly Temperature (°C avg) ─────────────────────────────────────
# Nov=22, Dec=20, Jan=19 (our window: months 11,12,1)
MONTH_TEMP = {1:21, 2:24, 3:28, 4:31, 5:31, 6:27, 7:25, 8:26, 9:26, 10:25, 11:22, 12:20}

# ── Indian Festival Calendar (day-of-year → demand multiplier) ─────────────────
# Nov 2025 = day 305-334, Dec = 335-365, Jan 2026 = 1-31
# Diwali 2025: Oct 20 (approx day 293), Christmas Dec 25, New Year Jan 1
FESTIVAL_BOOSTS = {
    # (month, day): multiplier
    (11, 1):  1.08,   # Kannada Rajyotsava
    (11, 14): 1.05,   # Children's day
    (12, 24): 1.15,   # Christmas Eve
    (12, 25): 1.18,   # Christmas
    (12, 31): 1.25,   # New Year Eve — peak demand
    (1,  1):  1.20,   # New Year Day
    (1,  14): 1.10,   # Makar Sankranti / Pongal
    (1,  26): 1.12,   # Republic Day
}


def bangalore_temperature(ts: pd.Timestamp) -> float:
    """Return estimated Bangalore temperature for given timestamp (°C)."""
    base = MONTH_TEMP.get(ts.month, 24)
    # Diurnal swing: cooler at night/morning, warmer afternoon
    hour_swing = -3 * np.cos(2 * np.pi * (ts.hour - 14) / 24)
    return base + hour_swing


def ac_load_factor(temp: float) -> float:
    """AC load kicks in above 26°C; roughly linear above that."""
    if temp <= 24:
        return 0.0
    elif temp <= 30:
        return (temp - 24) / 6 * 0.4   # 0 to 0.4 scaling
    else:
        return 0.4 + (temp - 30) * 0.05


def area_diurnal_profile(hour_float: float, area_type: str) -> float:
    """
    Generate area-specific diurnal consumption profile (0-1 normalised).
    Based on Indian electricity consumption patterns.
    """
    h = hour_float

    if area_type == "residential":
        # Morning: 6-9 AM (breakfast, geyser, school)
        morning = 0.55 * np.exp(-0.5 * ((h - 7.5) / 1.2) ** 2)
        # Afternoon lull: 12-4 PM
        afternoon = 0.15
        # Evening peak: 6-10 PM (cooking, lights, TV, AC)
        evening = 1.00 * np.exp(-0.5 * ((h - 20.0) / 2.0) ** 2)
        # Night base: 10 PM - 5 AM
        night = 0.10
        return max(night + morning + afternoon * (1 if 12 <= h <= 16 else 0) + evening, 0.05)

    elif area_type == "commercial":
        # Flat office hours 9AM-7PM, strong peak at lunch & closing
        if 9 <= h <= 20:
            base_comm = 0.70
            lunch_bump = 0.20 * np.exp(-0.5 * ((h - 13) / 1.0) ** 2)
            closing_bump = 0.15 * np.exp(-0.5 * ((h - 18.5) / 1.5) ** 2)
            return base_comm + lunch_bump + closing_bump
        elif 7 <= h < 9:
            return 0.30 + (h - 7) / 2 * 0.40   # ramp up
        elif 20 < h <= 22:
            return 0.70 - (h - 20) / 2 * 0.60   # ramp down
        else:
            return 0.08   # nighttime

    elif area_type == "industrial":
        # Three-shift factory: fairly flat with slight shift change peaks
        shift_change = (
            0.12 * np.exp(-0.5 * ((h - 6) / 0.5) ** 2) +
            0.12 * np.exp(-0.5 * ((h - 14) / 0.5) ** 2) +
            0.12 * np.exp(-0.5 * ((h - 22) / 0.5) ** 2)
        )
        return 0.75 + shift_change

    else:  # mixed
        res = area_diurnal_profile(h, "residential")
        com = area_diurnal_profile(h, "commercial")
        return 0.55 * res + 0.45 * com


def meter_scale_for_type(area_type: str, rng) -> float:
    """Household/unit size variation by area type."""
    if area_type == "residential":
        # Mix of 1BHK (small) to 3BHK (large)
        return rng.choice([0.6, 0.8, 1.0, 1.3, 1.6], p=[0.15, 0.30, 0.30, 0.18, 0.07])
    elif area_type == "commercial":
        return rng.choice([0.7, 1.0, 1.5, 2.5], p=[0.25, 0.35, 0.28, 0.12])
    elif area_type == "industrial":
        return rng.choice([1.0, 2.0, 4.0, 7.0], p=[0.30, 0.35, 0.25, 0.10])
    else:
        return rng.choice([0.7, 1.0, 1.4], p=[0.35, 0.40, 0.25])


def generate_meter_series(
    meter_id: str,
    zone: dict,
    base_scale: float,
    fraud_type: str | None,
    timestamps: pd.DatetimeIndex,
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(hash(meter_id) % (2**31))
    n = len(timestamps)
    readings = np.zeros(n)
    anomaly_flags = ["normal"] * n
    area_type = zone["area_type"]

    for i, ts in enumerate(timestamps):
        h = ts.hour + ts.minute / 60.0
        temp = bangalore_temperature(ts)

        # Base diurnal profile
        base = area_diurnal_profile(h, area_type)

        # AC load addition (temperature-driven)
        ac = ac_load_factor(temp)
        base += ac * rng.uniform(0.8, 1.2)  # some meters have AC, some don't

        # Weekend effect
        if ts.dayofweek >= 5:
            if area_type == "commercial":
                base *= 0.35   # offices mostly closed
            elif area_type == "industrial":
                base *= 0.55   # reduced production
            else:
                base *= 0.88   # residential slightly lower

        # Festival boost
        festival_mult = FESTIVAL_BOOSTS.get((ts.month, ts.day), 1.0)
        base *= festival_mult

        # Phase imbalance noise (realistic meter noise, not pure Gaussian)
        phase_noise = rng.normal(0, 0.04) + rng.choice([-0.02, 0, 0.02], p=[0.1, 0.8, 0.1])
        readings[i] = max(base * base_scale * (1 + phase_noise), 0.005)

    # ── Inject fraud anomalies ─────────────────────────────────────────────────
    if fraud_type == "sudden_drop":
        # Meter bypass: near-zero for 20-35 consecutive days
        drop_start = rng.integers(5 * READINGS_PER_DAY, 40 * READINGS_PER_DAY)
        drop_len   = rng.integers(20, 36) * READINGS_PER_DAY
        drop_end   = min(drop_start + drop_len, n)
        # Keep tiny base-load (e.g. standby) to avoid obvious zero
        readings[drop_start:drop_end] *= rng.uniform(0.02, 0.08)
        for j in range(drop_start, drop_end):
            anomaly_flags[j] = "sudden_drop"

    elif fraud_type == "periodic_spike":
        # Illegal high-load equipment run at night every few days
        night_slots = [i for i, ts in enumerate(timestamps) if ts.hour in [1, 2, 3, 4]]
        # Select ~3 nights per week
        spike_count = len(night_slots) // 8
        if spike_count > 0:
            spike_indices = rng.choice(night_slots, size=spike_count, replace=False)
            for idx in spike_indices:
                # Spike lasts 1-3 slots (15-45 min)
                duration = rng.integers(1, 4)
                for d in range(duration):
                    if idx + d < n:
                        readings[idx + d] *= rng.uniform(6.0, 12.0)
                        anomaly_flags[idx + d] = "periodic_spike"

    elif fraud_type == "under_reporting":
        # Shunt / magnet on meter — consistent deep under-reading
        factor = rng.uniform(0.08, 0.20)   # 80-92% under-reporting
        readings *= factor
        anomaly_flags = ["under_reporting"] * n

    elif fraud_type == "zone_mismatch":
        # Meter physically wired to a different feeder (e.g. industrial load on residential feeder)
        # Simulate by using industrial profile on this meter regardless of zone
        for i, ts in enumerate(timestamps):
            h = ts.hour + ts.minute / 60.0
            industrial_base = area_diurnal_profile(h, "industrial")
            phase_noise = rng.normal(0, 0.04)
            readings[i] = max(industrial_base * base_scale * (1 + phase_noise), 0.005)
        anomaly_flags = ["zone_mismatch"] * n

    return readings, anomaly_flags


def inject_realistic_missing(readings: np.ndarray, rng, meter_id: str) -> np.ndarray:
    """
    Inject missing data in realistic communication-blackout patterns:
    - Short blackouts: 1-4 consecutive slots (comm timeout)
    - Medium outages: 30-60 slots (~8-15 hours, maintenance window)
    - Random scatter: occasional single-slot misses
    """
    readings = readings.astype(float)
    n = len(readings)

    # Short comm timeouts (3-6 events per meter)
    num_short = rng.integers(3, 7)
    for _ in range(num_short):
        start = rng.integers(0, n - 5)
        length = rng.integers(1, 5)
        readings[start:start + length] = np.nan

    # 1 medium maintenance window per meter
    if rng.random() < 0.40:
        start = rng.integers(0, n - 60)
        length = rng.integers(20, 60)
        readings[start:start + length] = np.nan

    # Random scatter (~0.5%)
    scatter_mask = rng.random(n) < 0.005
    readings[scatter_mask] = np.nan

    return readings


def generate_all_data(output_dir: Path):
    np.random.seed(SEED)
    random.seed(SEED)
    rng_global = np.random.default_rng(SEED)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Timestamps ─────────────────────────────────────────────────────────────
    start = pd.Timestamp(START_DATE)
    timestamps = pd.date_range(start, periods=DAYS * READINGS_PER_DAY, freq="15min")

    # ── Zone metadata ──────────────────────────────────────────────────────────
    zone_df = pd.DataFrame(ZONES)
    zone_df.to_csv(output_dir / "zone_metadata.csv", index=False)
    print(f"  Zone metadata: {len(zone_df)} zones")

    # ── Determine fraud meters ─────────────────────────────────────────────────
    all_meter_ids = [
        f"M{z+1:02d}{m+1:03d}"
        for z in range(NUM_ZONES)
        for m in range(METERS_PER_ZONE)
    ]
    num_fraud = max(1, int(FRAUD_FRACTION * TOTAL_METERS))
    fraud_meter_ids = random.sample(all_meter_ids, num_fraud)
    fraud_assignments = {}
    # Distribute fraud types roughly equally
    for i, mid in enumerate(fraud_meter_ids):
        fraud_assignments[mid] = FRAUD_TYPES[i % len(FRAUD_TYPES)]

    # ── Generate readings ──────────────────────────────────────────────────────
    all_readings = []
    meter_meta   = []
    meter_num    = 0

    for z_idx, zone in enumerate(ZONES):
        zone_id   = zone["zone_id"]
        area_type = zone["area_type"]

        for m_idx in range(METERS_PER_ZONE):
            meter_id = f"M{z_idx+1:02d}{m_idx+1:03d}"
            meter_num += 1
            fraud_type = fraud_assignments.get(meter_id, None)
            base_scale = meter_scale_for_type(area_type, rng_global)

            readings, flags = generate_meter_series(
                meter_id, zone, base_scale, fraud_type, timestamps
            )
            readings = inject_realistic_missing(readings, rng_global, meter_id)

            for ts, kwh, flag in zip(timestamps, readings, flags):
                all_readings.append({
                    "timestamp": ts,
                    "meter_id":  meter_id,
                    "zone_id":   zone_id,
                    "kwh":       round(float(kwh), 4) if not np.isnan(kwh) else np.nan,
                    "is_missing": bool(np.isnan(kwh)),
                })

            missing_pct = np.isnan(readings).mean() * 100
            meter_meta.append({
                "meter_id":   meter_id,
                "zone_id":    zone_id,
                "zone_name":  zone["zone_name"],
                "area_type":  area_type,
                "base_scale": round(float(base_scale), 3),
                "fraud_type": fraud_type if fraud_type else "none",
                "is_fraud":   fraud_type is not None,
                "fraud_label": 1 if fraud_type is not None else 0,
                "missing_pct": round(missing_pct, 2),
            })

            if meter_num % 10 == 0:
                print(f"  Generated {meter_num}/{TOTAL_METERS}: {meter_id} ({area_type})")

    readings_df = pd.DataFrame(all_readings)
    readings_df.to_csv(output_dir / "meter_readings.csv", index=False)
    print(f"  Meter readings: {len(readings_df):,} rows")

    meter_df = pd.DataFrame(meter_meta)
    meter_df.to_csv(output_dir / "meters_metadata.csv", index=False)
    print(f"  Meter metadata: {len(meter_df)} meters ({num_fraud} fraud, {len(ZONES)} zones)")

    return readings_df, zone_df, meter_df


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    generate_all_data(project_root / "data" / "raw")
    print("Data generation complete!")
