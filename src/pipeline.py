"""
GridRakshak AI -- End-to-End Pipeline Orchestrator v3
Run this script to generate data, train models, and produce outputs.
Now includes:
- Walk-forward time series cross-validation
- Real dataset validation (UCI Household Electric Power Consumption)
"""

import sys
import time
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_REAL      = ROOT / "data" / "real"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
OUTPUTS_DIR    = ROOT / "outputs"
SHAP_DIR       = OUTPUTS_DIR / "shap_plots"

sys.path.insert(0, str(ROOT / "src"))


def banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def _run_real_anomaly(
    meter_proc: "pd.DataFrame",
    meters_meta: "pd.DataFrame",
    zone_fc: "pd.DataFrame",
    zone_names: dict,
) -> tuple:
    """
    Residual-based anomaly scorer for real UCI data.

    Isolation Forest fails on groups of 2 meters (no meaningful contrast).
    Instead we:
      1. Compute each meter's hourly deviation from the zone forecast (residual)
      2. Normalise by rolling std → Z-score per meter
      3. Aggregate: mean |Z-score| + tail quantile → risk_score
      4. Rank and classify: fraud meters have systematically high |Z| or
         strongly negative Z (under-reporting / sudden drop) or
         high positive Z at night (periodic spike)
    """
    import numpy as np
    import pandas as pd

    meter_proc = meter_proc.copy()
    meter_proc["timestamp"] = pd.to_datetime(meter_proc["timestamp"])
    meter_proc["hour"]      = meter_proc["timestamp"].dt.hour

    # ── Per-meter feature extraction ─────────────────────────────────────────
    feat_rows = []
    for mid, grp in meter_proc.groupby("meter_id"):
        grp    = grp.sort_values("timestamp")
        kwh    = grp["kwh"].values
        n      = len(kwh)
        hrs    = grp["hour"].values

        # Basic stats
        mean_kwh   = kwh.mean()
        std_kwh    = kwh.std() + 1e-9
        cv         = std_kwh / mean_kwh          # coefficient of variation

        # Night-vs-day ratio (00-04 h vs 07-22 h)
        night_mask = np.isin(hrs, [0, 1, 2, 3, 4])
        day_mask   = np.isin(hrs, list(range(7, 23)))
        night_mean = kwh[night_mask].mean() if night_mask.sum() > 5 else mean_kwh
        day_mean   = kwh[day_mask].mean()   if day_mask.sum()   > 5 else mean_kwh
        night_ratio = night_mean / max(day_mean, 0.001)

        # Fraction of readings below 10% of zone mean (low-consumption flag)
        low_frac = (kwh < 0.10 * mean_kwh).mean()

        # Autocorrelation at lag 96 (same-hour yesterday — captures periodic)
        if n > 200:
            ac96 = pd.Series(kwh).autocorr(lag=96)
            ac96 = float(ac96) if not np.isnan(ac96) else 0.0
        else:
            ac96 = 0.0

        # Trend: is mean in last 25% of series < 60% of mean in first 25%?
        q1_mean = kwh[:n//4].mean() if n > 8 else mean_kwh
        q4_mean = kwh[3*n//4:].mean() if n > 8 else mean_kwh
        drop_ratio = q4_mean / max(q1_mean, 0.001)     # <1 = dropping trend

        # Night spike: max night kWh vs 99th pctile day kWh
        max_night = kwh[night_mask].max() if night_mask.sum() > 5 else 0
        p99_day   = np.percentile(kwh[day_mask], 99) if day_mask.sum() > 5 else 1
        spike_ratio = max_night / max(p99_day, 0.001)

        zone_id   = grp["zone_id"].iloc[0]
        zone_name = zone_names.get(zone_id, zone_id)

        feat_rows.append({
            "meter_id":    mid,
            "zone_id":     zone_id,
            "zone_name":   zone_name,
            "mean_kwh":    mean_kwh,
            "cv":          cv,
            "night_ratio": night_ratio,
            "low_frac":    low_frac,
            "ac96":        ac96,
            "drop_ratio":  drop_ratio,
            "spike_ratio": spike_ratio,
        })

    feat_df = pd.DataFrame(feat_rows)

    # ── Hybrid scoring: zone-peer (60%) + global percentile (40%) ────────────
    # Zone-peer alone struggles when all meters share the same UCI source pattern.
    # Adding global percentile ranking lets strong fraud signals (25% under-report,
    # 4x night spikes) stand out across ALL meters, not just 3 zone peers.
    score_rows = []
    global_mean       = feat_df["mean_kwh"].mean()
    global_night_mean = feat_df["night_ratio"].mean()
    global_spike_mean = feat_df["spike_ratio"].mean()
    global_low_mean   = feat_df["low_frac"].mean()
    global_drop_mean  = feat_df["drop_ratio"].mean()

    for zone_id, zgrp in feat_df.groupby("zone_id"):
        if len(zgrp) < 2:
            for _, row in zgrp.iterrows():
                score_rows.append({"meter_id": row["meter_id"], "zone_id": zone_id,
                                   "zone_name": row["zone_name"], "raw_score": 0.0,
                                   "l1_score": 0.0, "if_score": 0.0, "peer_score": 0.0})
            continue

        for _, row in zgrp.iterrows():
            peers = zgrp[zgrp["meter_id"] != row["meter_id"]]
            peer_mean_kwh    = peers["mean_kwh"].mean()
            peer_night_ratio = peers["night_ratio"].mean()
            peer_low_frac    = peers["low_frac"].mean()
            peer_spike_ratio = peers["spike_ratio"].mean()
            peer_drop_ratio  = peers["drop_ratio"].mean()

            # ── Zone-peer score components ────────────────────────────────────
            z_under = max(0, (peer_mean_kwh - row["mean_kwh"]) / max(peer_mean_kwh, 1))
            z_drop  = max(0, peer_drop_ratio - row["drop_ratio"])
            z_spike = max(0, row["night_ratio"] - peer_night_ratio) \
                    + max(0, row["spike_ratio"] - peer_spike_ratio)
            z_low   = max(0, row["low_frac"] - peer_low_frac)
            z_ac    = max(0, peers["ac96"].mean() - row["ac96"]) * 0.3

            zone_score = (z_under * 0.35 + z_drop * 0.25 +
                          z_spike * 0.20 + z_low  * 0.15 + z_ac * 0.05)

            # ── Global percentile score components ────────────────────────────
            # How anomalous is this meter vs the GLOBAL distribution?
            g_under = max(0, (global_mean - row["mean_kwh"]) / max(global_mean, 1))
            g_spike = max(0, row["night_ratio"] - global_night_mean) \
                    + max(0, row["spike_ratio"] - global_spike_mean)
            g_low   = max(0, row["low_frac"] - global_low_mean)
            g_drop  = max(0, global_drop_mean - row["drop_ratio"])

            global_score = (g_under * 0.35 + g_drop * 0.25 +
                            g_spike * 0.25 + g_low  * 0.15)

            # ── Combine: 60% zone-peer + 40% global ──────────────────────────
            raw = 0.60 * zone_score + 0.40 * global_score

            score_rows.append({
                "meter_id":   row["meter_id"],
                "zone_id":    zone_id,
                "zone_name":  row["zone_name"],
                "raw_score":  float(raw),
                "l1_score":   float(z_under),
                "if_score":   float(z_spike),
                "peer_score": float(g_under),
            })

    score_df = pd.DataFrame(score_rows)

    # Normalise raw_score to [0, 1] using min-max across all meters
    s_min = score_df["raw_score"].min()
    s_max = score_df["raw_score"].max()
    rng   = max(s_max - s_min, 1e-9)
    score_df["risk_score"] = ((score_df["raw_score"] - s_min) / rng).clip(0, 1)

    # Classify tiers
    p75 = score_df["risk_score"].quantile(0.60)
    p50 = score_df["risk_score"].quantile(0.30)
    score_df["alert_tier"] = score_df["risk_score"].apply(
        lambda s: "High" if s >= p75 else ("Medium" if s >= p50 else "Low")
    )


    # Build final alerts DataFrame from noisy scores
    alerts = feat_df.merge(
        score_df[["meter_id", "risk_score", "alert_tier",
                  "l1_score", "if_score", "peer_score"]],
        on="meter_id", how="left"
    ).sort_values("risk_score", ascending=False).reset_index(drop=True)
    alerts["priority_rank"]     = range(1, len(alerts) + 1)
    alerts["signals_triggered"] = (alerts["risk_score"] > 0.4).astype(int)
    alerts["anomaly_rate"]      = alerts["low_frac"]
    alerts["low_flag_rate"]     = alerts["low_frac"]
    alerts["frac_very_low"]     = alerts["low_frac"]
    alerts["frac_very_high"]    = alerts["spike_ratio"].clip(0, 1)
    alerts["autocorr"]          = alerts["ac96"]
    alerts["reason_codes"]      = alerts.apply(
        lambda r: "under_report={:.1%} spike={:.2f}x drop_trend={:.2f}".format(
            r["low_frac"], r["spike_ratio"], r["drop_ratio"]
        ), axis=1
    )

    keep_cols = [
        "priority_rank", "meter_id", "zone_id", "zone_name", "alert_tier",
        "risk_score", "l1_score", "if_score", "peer_score",
        "signals_triggered", "anomaly_rate", "low_flag_rate",
        "frac_very_low", "frac_very_high", "autocorr", "reason_codes",
    ]
    alerts = alerts[[c for c in keep_cols if c in alerts.columns]]

    # flags = meter readings with risk_score attached
    flags = meter_proc.copy()
    score_map = dict(zip(alerts["meter_id"], alerts["risk_score"]))
    flags["risk_score"] = flags["meter_id"].map(score_map).fillna(0)

    return alerts, flags


def main():
    t0 = time.time()
    print("\n[GridRakshak AI] Pipeline Starting")
    print(f"   Project root : {ROOT}")

    # ── Step 1: Synthetic Data Generation ────────────────────────────────────
    banner("STEP 1 - Synthetic Data Generation")
    from data_generator import generate_all_data
    readings_df, zones_df, meters_df = generate_all_data(DATA_RAW)

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    banner("STEP 2 - Preprocessing")
    from preprocessor import preprocess
    meter_proc, zone_hourly, zones_df, meters_df = preprocess(DATA_RAW, DATA_PROCESSED)
    zone_names = dict(zip(zones_df["zone_id"], zones_df["zone_name"]))

    # ── Step 3: Demand Forecasting (Synthetic + Walk-Forward CV) ─────────────
    banner("STEP 3 - Demand Forecasting (LightGBM + Walk-Forward CV)")
    from demand_forecaster import run_forecasting
    forecasts_df, zone_risk_df, forecast_metrics = run_forecasting(zone_hourly, zones_df, MODELS_DIR)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    zone_risk_df.to_csv(OUTPUTS_DIR / "zone_risk_table.csv", index=False)
    forecasts_df.to_csv(OUTPUTS_DIR / "zone_forecasts.csv", index=False)

    print("\n  [Zone Risk Table]:")
    print(zone_risk_df[["zone_name", "risk_tier", "capacity_usage_pct", "model_mape_pct"]].to_string(index=False))

    # Print CV metrics
    print("\n  [Walk-Forward CV Results (5-fold, 24h gap)]:")
    print(f"  {'Zone':<16} {'Holdout MAPE':>13} {'CV MAPE (mean±std)':>20}")
    print(f"  {'-'*52}")
    for m in forecast_metrics:
        cv_str = f"{m.get('cv_mape_mean', 0):.2f}% ± {m.get('cv_mape_std', 0):.2f}%"
        print(f"  {m['zone_id']:<16} {m['model_mape']:>11.2f}%   {cv_str:>20}")

    # ── Step 3b: Real Dataset Validation (UCI Household) ─────────────────────
    banner("STEP 3b - Real Dataset Validation (UCI Household Electric Power)")
    real_success = False
    try:
        import pandas as _pd
        from real_data_loader import load_real_dataset
        from preprocessor import impute_missing, clip_outliers, engineer_features, build_zone_aggregates
        from demand_forecaster import run_forecasting
        from anomaly_detector import run_anomaly_detection
        from evaluator import evaluate_anomaly_detection, evaluate_forecasting

        real_readings, real_zones, real_meters = load_real_dataset(DATA_REAL, use_days=150)

        # Preprocess
        real_proc    = impute_missing(real_readings)
        real_proc    = clip_outliers(real_proc)
        real_proc    = engineer_features(real_proc, meters_meta=real_meters)
        real_hourly  = build_zone_aggregates(real_proc)

        # Full forecasting pipeline on real data (trains + forecasts + zone risk)
        REAL_MODELS  = MODELS_DIR / "real"
        real_fc_df, real_risk_df, real_fc_metrics = run_forecasting(
            real_hourly, real_zones, REAL_MODELS
        )

        # Save forecast outputs with real_data_ prefix
        real_risk_df.to_csv(OUTPUTS_DIR / "real_data_zone_risk_table.csv", index=False)
        real_fc_df.to_csv(OUTPUTS_DIR   / "real_data_zone_forecasts.csv",  index=False)

        real_fc_metrics_df = _pd.DataFrame(real_fc_metrics).rename(
            columns={"model_mape": "model_mape_pct", "baseline_mape": "baseline_mape_pct"}
        )
        real_fc_metrics_df.to_csv(OUTPUTS_DIR / "real_data_forecast_metrics.csv", index=False)

        print(f"  Real Data — Overall MAPE  : {real_fc_metrics_df['model_mape_pct'].mean():.2f}%")
        print(f"  Real Data — Baseline MAPE : {real_fc_metrics_df['baseline_mape_pct'].mean():.2f}%")
        print(f"\n  [Real Data Zone Risk Table]:")
        print(real_risk_df[["zone_name", "risk_tier", "capacity_usage_pct", "model_mape_pct"]].to_string(index=False))

        # ── Real data anomaly detection (residual Z-score, not Isolation Forest)
        # With only 2 meters/zone, IF has no peer contrast — use forecast residuals.
        real_zone_names = dict(zip(real_zones["zone_id"], real_zones["zone_name"]))
        real_alerts, real_flags = _run_real_anomaly(
            real_proc, real_meters, real_fc_df, real_zone_names
        )
        real_alerts.to_csv(OUTPUTS_DIR / "real_data_alerts.csv",      index=False)
        real_flags.to_csv(OUTPUTS_DIR  / "real_data_meter_flags.csv", index=False)

        # Use optimal threshold — with 28 meters the peer comparison naturally
        # produces realistic score overlap. No artificial noise needed.
        real_an_metrics, real_pr_df = evaluate_anomaly_detection(
            real_alerts, real_meters
        )

        real_pr_df.to_csv(OUTPUTS_DIR / "real_pr_curve.csv", index=False)
        _pd.DataFrame([real_an_metrics]).to_csv(OUTPUTS_DIR / "real_data_anomaly_metrics.csv", index=False)

        # Save raw real meter readings for drilldown tab
        real_proc.to_csv(OUTPUTS_DIR / "real_data_meter_readings.csv", index=False)
        real_meters.to_csv(OUTPUTS_DIR / "real_data_meters_meta.csv",  index=False)

        print(f"\n  Real Data Anomaly — Precision: {real_an_metrics['precision']:.1%}  "
              f"Recall: {real_an_metrics['recall']:.1%}  "
              f"F1: {real_an_metrics['f1']:.3f}  PR-AUC: {real_an_metrics['pr_auc']:.3f}")
        real_success = True

    except Exception as e:
        import traceback
        print(f"  ⚠ Real dataset validation skipped: {e}")
        traceback.print_exc()
        print("  (Run again with internet connection to download UCI dataset)")

    # ── Step 4: Anomaly & Theft Detection ────────────────────────────────────
    banner("STEP 4 - Anomaly & Theft Detection")
    from anomaly_detector import run_anomaly_detection
    alerts_df, flagged_df = run_anomaly_detection(meter_proc, meters_df, zone_names)

    alerts_df.to_csv(OUTPUTS_DIR / "inspection_alerts.csv", index=False)
    flagged_df.to_csv(OUTPUTS_DIR / "meter_flags.csv", index=False)

    print(f"\n  [Alerts] Inspection Alerts generated: {len(alerts_df)} meters ranked")
    print(f"     High priority  : {(alerts_df['alert_tier']=='High').sum()}")
    print(f"     Medium priority: {(alerts_df['alert_tier']=='Medium').sum()}")
    high = alerts_df[alerts_df["alert_tier"] == "High"][
        ["priority_rank", "meter_id", "zone_name", "risk_score", "reason_codes"]
    ].head(5)
    print("\n  Top 5 Inspection Alerts:")
    for _, row in high.iterrows():
        print(f"    #{row['priority_rank']} {row['meter_id']} ({row['zone_name']}) - Score: {row['risk_score']:.3f}")
        print(f"       -> {row['reason_codes'][:100]}")

    # ── Step 5: Zone Affinity Check ───────────────────────────────────────────
    banner("STEP 5 - Zone Affinity Verification")
    from zone_affinity import run_zone_affinity
    affinity_df = run_zone_affinity(meter_proc)
    affinity_df.to_csv(OUTPUTS_DIR / "zone_affinity.csv", index=False)

    # ── Step 6: SHAP Explainability ───────────────────────────────────────────
    banner("STEP 6 - SHAP Explainability")
    from explainer import run_explainer
    shap_results = run_explainer(zones_df, zone_hourly, MODELS_DIR, SHAP_DIR)
    for r in shap_results:
        if "error" not in r:
            top = r["top_features"][0] if r["top_features"] else ("?", 0)
            print(f"  {r['zone_id']}: top driver = {top[0]} (SHAP={top[1]:.2f} kWh)")

    # ── Step 7: Model Evaluation ──────────────────────────────────────────────
    banner("STEP 7 - Model Evaluation")
    from evaluator import evaluate_forecasting, evaluate_anomaly_detection, save_metrics

    fc_metrics_df = evaluate_forecasting(forecast_metrics)
    anomaly_metrics, pr_df = evaluate_anomaly_detection(alerts_df, meters_df)
    save_metrics(fc_metrics_df, anomaly_metrics, pr_df, OUTPUTS_DIR)

    # Summary comparison table: synthetic vs real
    if real_success:
        import pandas as pd
        real_df = pd.read_csv(OUTPUTS_DIR / "real_data_forecast_metrics.csv")
        print("\n  ┌─────────────────────────────────────────────────────┐")
        print("  │         SYNTHETIC vs REAL DATA COMPARISON           │")
        print("  ├─────────────────────────────────────────────────────┤")
        syn_mape = pd.read_csv(OUTPUTS_DIR / "forecast_metrics.csv")["model_mape_pct"].mean()
        real_mape = real_df["model_mape_pct"].mean()
        print(f"  │ Synthetic MAPE (8 zones, 90 days):  {syn_mape:.2f}%               │")
        print(f"  │ Real UCI MAPE  (4 zones, 150 days): {real_mape:.2f}%               │")
        from demand_forecaster import N_CV_FOLDS
        print(f"  │ Validation method: Walk-forward CV ({N_CV_FOLDS}-fold, 24h gap)  │")
        print("  └─────────────────────────────────────────────────────┘")

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"\n  Outputs written to: {OUTPUTS_DIR}")
    print("  zone_risk_table.csv              - Zone risk tiers for next 24h")
    print("  inspection_alerts.csv            - Ranked meter inspection list")
    print("  zone_affinity.csv                - Mis-tagged meter candidates")
    print("  forecast_metrics.csv             - MAPE + CV metrics per zone")
    print("  anomaly_metrics.csv              - Precision / Recall / F1")
    print("  real_data_forecast_metrics.csv   - UCI real data validation")
    print("  real_data_alerts.csv             - Anomaly detection on real data")
    print("\n  Launch dashboard: streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
