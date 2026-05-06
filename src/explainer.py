"""
GridRakshak AI — SHAP Explainer
Generates SHAP feature importance and human-readable reason codes
for demand forecasts and anomaly alerts.
"""

import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


FEATURE_COLS = [
    "hour", "day_of_week", "month", "day_of_year", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_peak_hour", "is_sleeping_hour",
    "lag_24h", "lag_168h", "roll_24h_mean", "roll_24h_std", "roll_7d_mean",
    "area_type_code",
]

FEATURE_LABELS = {
    "hour":             "Hour of Day",
    "day_of_week":      "Day of Week",
    "month":            "Month",
    "day_of_year":      "Day of Year",
    "is_weekend":       "Is Weekend",
    "hour_sin":         "Hour (sin)",
    "hour_cos":         "Hour (cos)",
    "dow_sin":          "Day-of-Week (sin)",
    "dow_cos":          "Day-of-Week (cos)",
    "is_peak_hour":     "Is Peak Hour",
    "is_sleeping_hour": "Is Sleeping Hour",
    "lag_24h":          "Yesterday Same Hour",
    "lag_168h":         "Last Week Same Hour",
    "roll_24h_mean":    "24h Rolling Mean",
    "roll_24h_std":     "24h Rolling Std Dev",
    "roll_7d_mean":     "7-Day Rolling Mean",
    "area_type_code":   "Area Type (Res/Com/Ind)",
}


def compute_shap_values(model, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values for LightGBM model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer.expected_value


def get_top_features(shap_values: np.ndarray, feature_names: list, n: int = 3) -> list[str]:
    """Return top N features by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n]
    return [(feature_names[i], round(float(mean_abs[i]), 3)) for i in top_idx]


def explain_zone_forecast(
    zone_id: str,
    zone_hourly: pd.DataFrame,
    models_dir: Path,
    output_dir: Path,
) -> dict:
    """Generate SHAP summary for a zone's forecast model."""
    model_path = models_dir / f"lgbm_{zone_id}_p50.pkl"
    if not model_path.exists():
        return {"zone_id": zone_id, "error": "Model not found"}

    model = joblib.load(model_path)
    df = zone_hourly[zone_hourly["zone_id"] == zone_id].dropna(subset=FEATURE_COLS)

    # Sample max 500 rows for speed
    sample = df[FEATURE_COLS].sample(min(500, len(df)), random_state=42)
    X = sample.values

    shap_vals, base_val = compute_shap_values(model, X)
    top_features = get_top_features(shap_vals, FEATURE_COLS)

    # Save SHAP bar plot
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [FEATURE_LABELS.get(f, f) for f, _ in top_features]
    values = [v for _, v in top_features]
    colors = ["#e63946" if v > 0 else "#457b9d" for v in values]
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP| Value (kWh impact)")
    ax.set_title(f"Zone {zone_id} — Top Forecast Drivers")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plot_path = output_dir / f"shap_{zone_id}.png"
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return {
        "zone_id": zone_id,
        "base_value": round(float(base_val), 2),
        "top_features": top_features,
        "shap_plot": str(plot_path),
    }


def run_explainer(
    zones_df: pd.DataFrame,
    zone_hourly: pd.DataFrame,
    models_dir: Path,
    output_dir: Path,
) -> list[dict]:
    """Generate SHAP explanations for all zone models."""
    results = []
    for zone_id in zones_df["zone_id"]:
        print(f"  SHAP analysis for {zone_id}...")
        result = explain_zone_forecast(zone_id, zone_hourly, models_dir, output_dir)
        results.append(result)
    return results
