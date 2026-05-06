# ⚡ GridRakshak AI
### BESCOM Smart Meter Intelligence — Theme 8: AI for Bharat

> **Predictive field dispatch for BESCOM.** Tells operators which zones may overload tomorrow and which meters to inspect today — using only existing smart meter CSV data.  
> **No new hardware. No cloud. Deployable in 30 days.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| Forecast SMAPE (8 BESCOM zones) | **4.2%** vs 6.3% baseline — **33% better** |
| Anomaly Detection Precision | **85.7%** (Synthetic) / **100%** (Real UCI) |
| Anomaly Detection Recall | **100%** both datasets |
| F1 Score | **0.923** (Synthetic) / **1.000** (Real UCI) |
| PR-AUC | **0.948** (Synthetic) / **1.000** (Real UCI) |
| Validation | 5-fold walk-forward CV (24h leakage gap) |
| Real-world validation | UCI Household dataset (2M rows, 2006-2010) |

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Run Options](#-run-options)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Dashboard](#-dashboard)
- [Key Output Files](#-key-output-files)
- [Troubleshooting](#-troubleshooting)

---

## 🔧 Installation

### Prerequisites
- Python **3.9 or higher** ([download](https://www.python.org/downloads/))
- Git ([download](https://git-scm.com/downloads))
- ~500 MB free disk space
- Internet connection (first run only, to download UCI dataset)

### Step 1 — Clone the repository
```bash
git clone https://github.com/<your-username>/gridrakshak-ai.git
cd gridrakshak-ai
```

### Step 2 — Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** On first install this takes 2-4 minutes. LightGBM and SHAP are the largest packages.

### Step 4 — Verify install
```bash
python -c "import lightgbm, streamlit, shap; print('All dependencies OK')"
```

---

## 🚀 Quick Start

```bash
# Run full pipeline + launch dashboard
python run_pipeline.py
```

This single command:
1. Generates synthetic BESCOM smart meter data (80 meters, 90 days)
2. Preprocesses and engineers 17 time-series features
3. Trains LightGBM P50 + P90 models per zone (walk-forward CV)
4. Downloads & validates on real UCI household data
5. Runs 3-layer anomaly detection
6. Detects mis-tagged meters (zone affinity)
7. Generates SHAP explainability plots
8. Creates all evaluation reports
9. Asks whether to launch the Streamlit dashboard

**Total runtime: ~25 seconds** (excluding first-time UCI download ~60s)

---

## ⚙️ Run Options

```bash
# Full pipeline + dashboard prompt
python run_pipeline.py

# Skip data generation (use existing generated data)
python run_pipeline.py --skip-data

# Skip model training (use saved models)
python run_pipeline.py --skip-training

# Run pipeline only, no dashboard
python run_pipeline.py --no-dashboard

# Launch dashboard immediately (pipeline must have run before)
python run_pipeline.py --dashboard-only

# Run dashboard directly
streamlit run dashboard/app.py
```

### Running individual pipeline stages
```bash
python -X utf8 src/data_generator.py      # Generate synthetic data
python -X utf8 src/preprocessor.py        # Feature engineering
python -X utf8 src/demand_forecaster.py   # Train LightGBM models
python -X utf8 src/anomaly_detector.py    # Anomaly detection
python -X utf8 src/zone_affinity.py       # Feeder mis-tagging check
python -X utf8 src/explainer.py           # SHAP explainability
python -X utf8 src/evaluator.py           # Evaluation metrics
python -X utf8 src/generate_artifacts.py  # Generate all output files
```

---

## 🗂️ Project Structure

```
gridrakshak-ai/
├── run_pipeline.py              ← Master runner (start here)
├── requirements.txt             ← All Python dependencies
├── README.md
├── .gitignore
│
├── src/
│   ├── pipeline.py              ← Internal pipeline orchestrator
│   ├── data_generator.py        ← Synthetic BESCOM data (90d, 80 meters)
│   ├── preprocessor.py          ← Imputation, IQR clip, 17 features
│   ├── demand_forecaster.py     ← LightGBM P50+P90, walk-forward CV
│   ├── anomaly_detector.py      ← 3-layer: Z-score + IF + peer DTW
│   ├── zone_affinity.py         ← Pearson load-shape fingerprinting
│   ├── explainer.py             ← SHAP + rule-based reason codes
│   ├── evaluator.py             ← Precision / Recall / F1 / MAPE
│   ├── real_data_loader.py      ← UCI dataset downloader & processor
│   └── generate_artifacts.py   ← Creates all spec-required output files
│
├── dashboard/
│   └── app.py                   ← Streamlit UI (5 sections, dual-dataset)
│
├── data/
│   ├── raw/                     ← Generated synthetic CSVs
│   ├── real/                    ← UCI dataset cache (auto-downloaded)
│   ├── ground_truth.csv         ← 12 known fraud meters (for evaluation)
│   ├── mape_results.json        ← Forecast accuracy per zone
│   ├── evaluation_metrics.json  ← Anomaly precision / recall / F1
│   ├── zone_risk_explained.csv  ← Zone risk with plain-English reasons
│   ├── anomaly_alerts_explained.csv  ← Alerts with full_alert_reason
│   ├── zone_affinity_flags.csv  ← Mis-tagged meter flags
│   └── affinity_matrix.csv      ← 80×8 Pearson similarity matrix
│
├── models/                      ← Trained LightGBM models (generated)
│   └── .gitkeep
│
├── outputs/                     ← All pipeline outputs
│   ├── zone_risk_table.csv
│   ├── forecast_metrics.csv
│   ├── inspection_alerts.csv
│   ├── anomaly_metrics.csv
│   ├── zone_affinity.csv
│   ├── evaluation_summary.md
│   ├── benchmark_comparison.png
│   └── shap_plots/
│
└── logs/                        ← Pipeline run logs
    └── .gitkeep
```

---

## 🧠 How It Works

### 1 — Data Layer
- **Synthetic BESCOM data**: 8 zones, 80 meters, 90 days at 15-min intervals (~691K rows)
- **Real UCI data**: 2.05M rows of French household electricity (2006-2010), scaled to BESCOM feeder kWh range and validated as a separate dataset

### 2 — Feature Engineering (17 features)
| Feature | Description |
|---|---|
| `hour_sin`, `hour_cos` | Cyclical hour encoding |
| `dow_sin`, `dow_cos` | Cyclical day-of-week encoding |
| `is_peak_hour` | 7-9 AM and 6-9 PM flag |
| `is_sleeping_hour` | Midnight-5 AM flag |
| `lag_24h`, `lag_168h` | Yesterday / last-week same-hour |
| `roll_24h_mean/std` | Rolling 24h mean and volatility |
| `roll_7d_mean` | Rolling 7-day mean |
| `area_type_code` | Residential / Mixed / Commercial / Industrial |

### 3 — Forecasting Model
- **Algorithm**: LightGBM Quantile Regression
- **Two models per zone**: P50 (median forecast) + P90 (risk threshold)
- **Validation**: 5-fold walk-forward CV with 24h leakage gap
- **Target transform**: log1p for variance stabilisation

### 4 — 3-Layer Anomaly Detection
| Layer | Method | Detects |
|---|---|---|
| 1 | Statistical Z-score residual | Sudden drops, spikes |
| 2 | Isolation Forest | Complex multivariate anomalies |
| 3 | Peer DTW comparison | Under-reporting vs zone peers |
| Filter | Persistence (3+ windows, 2+ episodes/7d) | Reduces false positives by ~55% |

### 5 — Zone Affinity (unique feature)
Pearson correlation of load-shape fingerprints detects meters physically connected to the wrong feeder — a real BESCOM operational problem invisible to standard billing systems.

---

## 📡 Dashboard

```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

| Section | What You See |
|---|---|
| **KPI Cards** | Zones at risk, Meters flagged, Forecast accuracy |
| **Zone Risk Table** | HIGH/MEDIUM/LOW with forecast chart and Q90 threshold |
| **Inspection List** | Ranked meters with plain-English alert reasons |
| **Zone Affinity** | Mis-tagged meters with confidence + recommended action |
| **Model Performance** | MAPE chart, Precision/Recall, PR-Curve |
| **Sidebar Toggle** | Switch Synthetic BESCOM ↔ Real UCI dataset live |

---

## 📁 Key Output Files

| File | Description |
|---|---|
| `data/evaluation_metrics.json` | Precision, Recall, F1, PR-AUC |
| `data/mape_results.json` | MAPE per zone + overall |
| `data/zone_risk_explained.csv` | Zone risk with human-readable reasons |
| `data/anomaly_alerts_explained.csv` | Ranked meter alerts with full reasons |
| `data/zone_affinity_flags.csv` | Mis-tagged meters |
| `data/ground_truth.csv` | 12 known fraud meters |
| `outputs/evaluation_summary.md` | Full evaluation report |
| `outputs/benchmark_comparison.png` | MAPE comparison chart |
| `outputs/shap_plots/` | SHAP feature importance per zone |
| `logs/pipeline_run.log` | Timestamped pipeline log |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Dashboard shows "data not available" | Run `python run_pipeline.py` first |
| `dtaidistance` warning | `pip install dtaidistance` — or ignore (Pearson fallback used) |
| SHAP error | Ignore — rule-based reasons used as fallback |
| Pipeline step fails | Check `logs/pipeline_run.log` |
| UCI download fails | Check internet; cache at `data/real/` after first run |
| High SMAPE on real UCI data | Expected — single household is 5-10× more volatile than grid-level aggregates. Model beats baseline by 15%. |
| `UnicodeEncodeError` on Windows | Run with `python -X utf8 src/pipeline.py` |

---

## 📦 Requirements

```
lightgbm>=4.0
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
streamlit>=1.28
shap>=0.43
matplotlib>=3.7
plotly>=5.17
pyarrow>=13.0
dtaidistance>=2.3     # optional — Pearson fallback if missing
tqdm>=4.65
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📊 System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11+ |
| RAM | 4 GB | 8 GB |
| Disk | 500 MB | 2 GB |
| OS | Windows / macOS / Linux | Any |
| Internet | First run only (UCI dataset) | — |

---

## 🏗️ Architecture

```
Smart Meter CSV Data (15-min intervals)
           │
           ▼
┌──────────────────────┐
│    Preprocessor      │  IQR clipping · forward-fill · 17 features
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  LightGBM P50 + P90  │  Quantile regression · log1p · 5-fold walk-forward CV
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Zone Risk Tiers    │  HIGH / MEDIUM / LOW (Q90 exceedance rate)
└──────────┬───────────┘
           ▼
┌─────────────────────────────────────────────┐
│          3-Layer Anomaly Detection          │
│  Layer 1: Z-score residual (statistical)   │
│  Layer 2: Isolation Forest (ML)            │
│  Layer 3: Peer DTW comparison              │
│  Filter : 3+ windows · 2+ episodes/7d     │
└──────────┬──────────────────────────────────┘
           ▼
┌──────────────────────┐
│    Zone Affinity     │  Pearson fingerprinting → mis-tagged feeders
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│   Explainability     │  SHAP values + plain-English reason codes
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│     Dashboard        │  Streamlit · dual-dataset · 5 sections
└──────────────────────┘
```

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

*GridRakshak AI | BESCOM Smart Meter Intelligence | Theme 8 — AI for Bharat*  
*No new hardware. No cloud dependency. Deployable in 30 days on existing BESCOM infrastructure.*
