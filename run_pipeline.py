"""
GridRakshak AI — Master Pipeline Runner
Usage:
  python run_pipeline.py                   # Full pipeline + dashboard
  python run_pipeline.py --skip-data       # Skip data generation
  python run_pipeline.py --skip-training   # Skip model training
  python run_pipeline.py --dashboard-only  # Dashboard only
  python run_pipeline.py --no-dashboard    # Pipeline without dashboard
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Directory setup ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SRC  = ROOT / "src"
DIRS = ["data", "data/raw", "data/real", "data/processed",
        "models", "logs", "outputs", "outputs/shap_plots"]
for d in DIRS:
    Path(ROOT / d).mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / "pipeline_run.log"

def log(msg: str):
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass   # If log write fails, continue silently


def run_step(name: str, script: Path, step_n: int, total: int) -> bool:
    """Run a pipeline step. Returns True on success."""
    log(f"\n{'─'*60}")
    log(f"[Step {step_n}/{total}] Running {name}...")
    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT)
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        log(f"  ⚠  WARNING: {name} failed (exit code {result.returncode}). Continuing pipeline...")
        return False
    else:
        log(f"  ✓  {name} complete ({elapsed:.1f}s)")
        return True


def launch_dashboard():
    log("\n🚀 Launching dashboard at http://localhost:8501 ...")
    os.system(f'streamlit run "{ROOT / "dashboard" / "app.py"}"')


# ── CLI arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="GridRakshak AI — BESCOM Smart Meter Intelligence Pipeline"
)
parser.add_argument("--skip-data",       action="store_true", help="Skip data generation step")
parser.add_argument("--skip-training",   action="store_true", help="Skip model training step")
parser.add_argument("--dashboard-only",  action="store_true", help="Launch dashboard immediately")
parser.add_argument("--no-dashboard",    action="store_true", help="Run pipeline but skip dashboard")
args = parser.parse_args()

# ── Dashboard-only mode ───────────────────────────────────────────────────────
if args.dashboard_only:
    launch_dashboard()
    sys.exit(0)

# ── Pipeline steps ────────────────────────────────────────────────────────────
log("╔══════════════════════════════════════════════════════╗")
log("║          GridRakshak AI — Pipeline Start             ║")
log("╚══════════════════════════════════════════════════════╝")
log(f"Working directory : {ROOT}")
log(f"Started           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

STEPS = [
    ("Data Generation",      SRC / "data_generator.py",   not args.skip_data),
    ("Preprocessing",        SRC / "preprocessor.py",     True),
    ("Demand Forecasting",   SRC / "demand_forecaster.py",not args.skip_training),
    ("Anomaly Detection",    SRC / "anomaly_detector.py", True),
    ("Zone Affinity",        SRC / "zone_affinity.py",    True),
    ("SHAP Explainability",  SRC / "explainer.py",        True),
    ("Model Evaluation",     SRC / "evaluator.py",        True),
    ("Generate Artifacts",   SRC / "generate_artifacts.py", True),
]

# Filter steps (keep name + script, mark skipped)
active = [(n, s) for n, s, enabled in STEPS if enabled]
skipped = [n for n, _, enabled in STEPS if not enabled]

if skipped:
    log(f"\nSkipping: {', '.join(skipped)}")

pipeline_start = time.time()
results = {}

for i, (name, script) in enumerate(active, 1):
    if not script.exists():
        log(f"  ⚠  {script.name} not found — skipping")
        results[name] = False
        continue
    results[name] = run_step(name, script, i, len(active))

# ── Summary ───────────────────────────────────────────────────────────────────
total_sec = time.time() - pipeline_start
mins = int(total_sec // 60)
secs = int(total_sec % 60)

log(f"\n{'═'*60}")
log(f"Pipeline complete in {mins} minutes {secs} seconds")
log(f"{'═'*60}")

passed = sum(results.values())
failed = len(results) - passed
for name, ok in results.items():
    log(f"  {'✓' if ok else '✗'}  {name}")

log(f"\n  {passed}/{len(results)} steps completed successfully")

if failed:
    log(f"\n  Check logs: {LOG_FILE}")

# ── Key output files ──────────────────────────────────────────────────────────
log("\nKey output files:")
KEY_FILES = [
    ROOT / "data"    / "zone_risk_explained.csv",
    ROOT / "data"    / "anomaly_alerts_explained.csv",
    ROOT / "data"    / "zone_affinity_flags.csv",
    ROOT / "data"    / "mape_results.json",
    ROOT / "data"    / "evaluation_metrics.json",
    ROOT / "data"    / "ground_truth.csv",
    ROOT / "outputs" / "evaluation_summary.md",
    ROOT / "outputs" / "benchmark_comparison.png",
]
for f in KEY_FILES:
    status = "✓" if f.exists() else "✗ MISSING"
    log(f"  [{status}] {f.relative_to(ROOT)}")

# ── Dashboard ─────────────────────────────────────────────────────────────────
if not args.no_dashboard:
    ans = input("\nLaunch dashboard now? (y/n): ").strip().lower()
    if ans == "y":
        launch_dashboard()
