"""
GridRakshak AI -- Model Evaluator v2
- Optimal threshold finder (Precision-Recall curve)
- MAPE vs baseline per zone
- Per-fraud-type detection breakdown
- Saves PR curve data for dashboard visualization
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, auc,
)
from pathlib import Path


def evaluate_forecasting(all_metrics: list[dict]) -> pd.DataFrame:
    """Summarise zone-level MAPE results."""
    df = pd.DataFrame(all_metrics)
    if "model_mape" in df.columns and "model_mape_pct" not in df.columns:
        df = df.rename(columns={"model_mape": "model_mape_pct", "baseline_mape": "baseline_mape_pct"})
    df["mape_improvement"] = df["baseline_mape_pct"] - df["model_mape_pct"]
    overall_mape     = df["model_mape_pct"].mean()
    overall_baseline = df["baseline_mape_pct"].mean()

    print("\n[DEMAND FORECASTING RESULTS]")
    print("=" * 60)
    print(df[["zone_id", "model_mape_pct", "baseline_mape_pct", "mape_improvement"]].to_string(index=False))
    print(f"\n  Overall Model MAPE   : {overall_mape:.2f}%")
    print(f"  Overall Baseline MAPE: {overall_baseline:.2f}%")
    print(f"  Improvement          : {overall_baseline - overall_mape:.2f}%")
    return df


def find_optimal_threshold(
    alerts: pd.DataFrame,
    meters_meta: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """
    Sweep thresholds 0.05–0.95 and find the one maximising F1.
    Returns (best_threshold, pr_curve_df).
    """
    merged = alerts.merge(
        meters_meta[["meter_id", "fraud_label"]],
        on="meter_id", how="left"
    )
    y_true  = merged["fraud_label"].fillna(0).astype(int).values
    y_score = merged["risk_score"].values

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recalls, precisions)

    # Build sweep table
    rows = []
    for t in np.arange(0.05, 0.96, 0.025):
        y_pred = (y_score >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        rows.append({"threshold": round(t, 3), "precision": round(p, 3),
                     "recall": round(r, 3), "f1": round(f, 3)})

    pr_df = pd.DataFrame(rows)
    # Best threshold = highest F1; tie-break on higher precision
    best_row = pr_df.sort_values(["f1", "precision"], ascending=False).iloc[0]
    best_t   = float(best_row["threshold"])

    print(f"\n  Optimal threshold    : {best_t:.3f}")
    print(f"  At optimal — Precision: {best_row['precision']:.1%}  "
          f"Recall: {best_row['recall']:.1%}  F1: {best_row['f1']:.3f}")
    print(f"  PR-AUC               : {pr_auc:.3f}")

    return best_t, pr_df, pr_auc


def evaluate_anomaly_detection(
    alerts: pd.DataFrame,
    meters_meta: pd.DataFrame,
    threshold: float = None,
) -> dict:
    """Evaluate anomaly detection. If threshold is None, auto-find optimal."""
    merged = alerts.merge(
        meters_meta[["meter_id", "fraud_label", "fraud_type"]],
        on="meter_id", how="left"
    )
    y_true  = merged["fraud_label"].fillna(0).astype(int).values
    y_score = merged["risk_score"].values

    # Auto-find optimal threshold
    if threshold is None:
        _, pr_df, pr_auc = find_optimal_threshold(alerts, meters_meta)
        best_row = pr_df.sort_values(["f1", "precision"], ascending=False).iloc[0]
        threshold = float(best_row["threshold"])
    else:
        _, pr_df, pr_auc = find_optimal_threshold(alerts, meters_meta)

    y_pred = (y_score >= threshold).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr  = fp / (fp + tn + 1e-9)

    print("\n[ANOMALY DETECTION RESULTS]")
    print("=" * 60)
    print(f"  Threshold (optimal)  : {threshold:.3f}")
    print(f"  Precision            : {prec:.3f}  ({int(prec*100)}%)")
    print(f"  Recall               : {rec:.3f}  ({int(rec*100)}%)")
    print(f"  F1 Score             : {f1:.3f}")
    print(f"  False Positive Rate  : {fpr:.3f}  ({int(fpr*100)}%)")
    print(f"  True Positives : {tp}  |  False Positives: {fp}")
    print(f"  True Negatives : {tn}  |  False Negatives: {fn}")
    print(f"  PR-AUC               : {pr_auc:.3f}")

    if "fraud_type" in merged.columns:
        print("\n  Per-Fraud-Type Detection:")
        for ftype in sorted(merged["fraud_type"].unique()):
            if ftype == "none":
                continue
            subset   = merged[merged["fraud_type"] == ftype]
            detected = (subset["risk_score"] >= threshold).sum()
            total    = len(subset)
            print(f"    {ftype:<22}: {detected}/{total} ({int(detected/max(total,1)*100)}%)")

    return {
        "threshold":  round(threshold, 3),
        "precision":  round(prec, 3),
        "recall":     round(rec, 3),
        "f1":         round(f1, 3),
        "fpr":        round(fpr, 3),
        "pr_auc":     round(pr_auc, 3),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }, pr_df


def save_metrics(
    forecast_metrics: pd.DataFrame,
    anomaly_metrics: dict,
    pr_df: pd.DataFrame,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_metrics.to_csv(output_dir / "forecast_metrics.csv", index=False)
    pd.DataFrame([anomaly_metrics]).to_csv(output_dir / "anomaly_metrics.csv", index=False)
    pr_df.to_csv(output_dir / "pr_curve.csv", index=False)
    print(f"\nMetrics saved to {output_dir}")
