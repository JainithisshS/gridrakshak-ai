"""
GridRakshak AI — Streamlit Dashboard
Interactive decision-support UI for BESCOM operators.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
OUTPUTS = ROOT / "outputs"
DATA_RAW = ROOT / "data" / "raw"
SHAP_DIR = OUTPUTS / "shap_plots"

st.set_page_config(
    page_title="GridRakshak AI — BESCOM Smart Meter Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0a0e1a; }
[data-testid="stSidebar"] { background: #0f1525; border-right: 1px solid #1e2a45; }
[data-testid="stSidebar"] * { color: #c8d6f0 !important; }

.metric-card {
    background: linear-gradient(135deg, #1a2540 0%, #0f1e38 100%);
    border: 1px solid #2a3a60;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,120,255,0.15); }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #60a5fa; margin: 4px 0; }
.metric-label { font-size: 0.85rem; color: #8899bb; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-delta { font-size: 0.8rem; margin-top: 4px; }

.risk-high { color: #ef4444 !important; font-weight: 700; }
.risk-medium { color: #f59e0b !important; font-weight: 700; }
.risk-low { color: #22c55e !important; font-weight: 700; }

.alert-card {
    background: #141e35;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.alert-card.medium { border-left-color: #f59e0b; }
.alert-card.low { border-left-color: #22c55e; }
.alert-meter { font-size: 1.1rem; font-weight: 600; color: #e2e8f0; }
.alert-reason { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }

.section-header {
    font-size: 1.4rem; font-weight: 700; color: #e2e8f0;
    border-bottom: 2px solid #2a3a60;
    padding-bottom: 8px; margin-bottom: 20px;
}

[data-testid="stDataFrame"] { background: #141e35; border-radius: 10px; }
.stTabs [data-baseweb="tab-list"] { background: #0f1525; border-radius: 10px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: #1a2540; border-radius: 8px; color: #94a3b8; }
.stTabs [aria-selected="true"] { background: #2563eb; color: white !important; }
</style>
""", unsafe_allow_html=True)


# ── Data Loaders ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_zone_risk():
    p = OUTPUTS / "zone_risk_table.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data(ttl=30)
def load_alerts():
    p = OUTPUTS / "inspection_alerts.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data(ttl=30)
def load_forecasts():
    p = OUTPUTS / "zone_forecasts.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["timestamp"])
    return None

@st.cache_data(ttl=30)
def load_meter_readings():
    p = DATA_RAW / "meter_readings.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["timestamp"])
    return None

@st.cache_data(ttl=30)
def load_meter_flags():
    p = OUTPUTS / "meter_flags.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["timestamp"])
    return None

@st.cache_data(ttl=30)
def load_metrics():
    fc = OUTPUTS / "forecast_metrics.csv"
    an = OUTPUTS / "anomaly_metrics.csv"
    return (
        pd.read_csv(fc) if fc.exists() else None,
        pd.read_csv(an) if an.exists() else None,
    )

@st.cache_data(ttl=30)
def load_affinity():
    p = OUTPUTS / "zone_affinity.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data(ttl=30)
def load_pr_curve():
    p = OUTPUTS / "pr_curve.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data(ttl=30)
def load_meters_meta():
    p = DATA_RAW / "meters_metadata.csv"
    return pd.read_csv(p) if p.exists() else None


def check_outputs():
    required = [
        OUTPUTS / "zone_risk_table.csv",
        OUTPUTS / "inspection_alerts.csv",
    ]
    return all(p.exists() for p in required)


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ GridRakshak AI")
    st.markdown("*BESCOM Smart Meter Intelligence*")
    st.divider()

    # ── Dataset selector ────────────────────────────────────────────────────
    st.markdown("**📂 Dataset**")
    dataset_choice = st.radio(
        "Select dataset to explore:",
        options=["🏙️ Synthetic BESCOM", "🏠 Real UCI Household"],
        index=0, label_visibility="collapsed",
    )
    IS_REAL = dataset_choice == "🏠 Real UCI Household"
    PREFIX  = "real_data_" if IS_REAL else ""

    st.divider()
    st.markdown("**System Status**")
    if check_outputs():
        st.success("✅ Pipeline outputs ready")
        if IS_REAL and not (OUTPUTS / "real_data_zone_risk_table.csv").exists():
            st.warning("⚠️ Real data outputs missing. Re-run pipeline.")
    else:
        st.error("❌ Run `python src/pipeline.py` first")

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown("**About**")
    st.markdown("""
GridRakshak AI converts smart meter data
into predictive, actionable intelligence.

- 🔮 24h demand forecasting
- 🚨 Theft & anomaly detection
- 🗺️ Zone affinity verification
- 🤖 SHAP explainability
""")
    st.divider()
    st.caption("Powered by LightGBM · SHAP · DTW")
    st.caption("Version 2.0 · BESCOM Pilot")


# ── Main Content ─────────────────────────────────────────────────────────────────
st.markdown("# ⚡ GridRakshak AI")
st.markdown("### Predictive Smart Meter Intelligence — BESCOM Decision Support")

if not check_outputs():
    st.warning("⚠️ No output data found. Please run the pipeline first:")
    st.code("python src/pipeline.py", language="bash")
    st.stop()

# ── Dataset context banner ────────────────────────────────────────────────────
if IS_REAL:
    st.info(
        "🏠 **Real UCI Household Dataset** — UCI Individual Household Electric Power Consumption "
        "(2006–2010, 2M rows, 1-min intervals). Fraud patterns injected into real baseline for "
        "ground-truth validation. This demonstrates the system works on **real-world data**, "
        "not just synthetic patterns.",
        icon="🔬",
    )
else:
    st.info(
        "🏙️ **Synthetic BESCOM Dataset** — 8 real Bangalore zones (Jayanagar, Koramangala, "
        "Whitefield…), 80 meters, 90 days, 691K readings. Includes area-type profiles, "
        "temperature-driven AC load, festival boosts (Diwali, New Year), and 12 fraud meters.",
        icon="📊",
    )

# ── Dynamic data loading based on dataset choice ─────────────────────────────
def _csv(name, parse_ts=False):
    """Load a CSV from OUTPUTS, trying real_data_ prefix first if IS_REAL."""
    path = OUTPUTS / f"{PREFIX}{name}" if IS_REAL else OUTPUTS / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if parse_ts and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

zone_risk   = _csv("zone_risk_table.csv")
alerts      = _csv("inspection_alerts.csv") if not IS_REAL else _csv("alerts.csv")
forecasts   = _csv("zone_forecasts.csv", parse_ts=True)
fc_metrics  = _csv("forecast_metrics.csv")
an_metrics  = _csv("anomaly_metrics.csv")
pr_curve    = _csv("pr_curve.csv")
affinity    = _csv("zone_affinity.csv") if not IS_REAL else None

# Meter readings & meta — different paths for each dataset
if IS_REAL:
    readings    = _csv("meter_readings.csv", parse_ts=True)
    flags       = _csv("meter_flags.csv", parse_ts=True)
    meters_meta = _csv("meters_meta.csv")
else:
    p = DATA_RAW / "meter_readings.csv"
    readings = pd.read_csv(p, parse_dates=["timestamp"]) if p.exists() else None
    p2 = OUTPUTS / "meter_flags.csv"
    flags = pd.read_csv(p2, parse_dates=["timestamp"]) if p2.exists() else None
    p3 = DATA_RAW / "meters_metadata.csv"
    meters_meta = pd.read_csv(p3) if p3.exists() else None

# Fix alert column name for real data
if alerts is not None and "alert_tier" not in alerts.columns and "risk_tier" in alerts.columns:
    alerts = alerts.rename(columns={"risk_tier": "alert_tier"})

# ── KPI Row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

def kpi(col, value, label, delta=None, color="#60a5fa"):
    with col:
        delta_html = f'<div class="metric-delta" style="color:{color}">{delta}</div>' if delta else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

high_zones = int((zone_risk["risk_tier"] == "High").sum()) if zone_risk is not None else 0
high_alerts = int((alerts["alert_tier"] == "High").sum()) if alerts is not None else 0
med_alerts = int((alerts["alert_tier"] == "Medium").sum()) if alerts is not None else 0

avg_mape = f"{fc_metrics['model_mape_pct'].mean():.1f}%" if fc_metrics is not None else "N/A"
prec = f"{int(an_metrics['precision'].iloc[0]*100)}%" if an_metrics is not None else "N/A"

kpi(col1, high_zones, "High Risk Zones", "⚠️ Requires Action", "#ef4444")
kpi(col2, high_alerts, "High Priority Alerts", "🔴 Inspect Immediately", "#f59e0b")
kpi(col3, med_alerts, "Medium Priority", "🟡 Monitor Closely", "#f59e0b")
kpi(col4, avg_mape, "Avg Forecast MAPE", "vs historical baseline", "#22c55e")
kpi(col5, prec, "Detection Precision", "anomaly detection", "#60a5fa")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Zone Risk Map",
    "🚨 Inspection Alerts",
    "📊 Meter Drilldown",
    "📈 Model Performance",
    "💰 Economic Impact",
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — Zone Risk Map
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">24-Hour Zone Risk Assessment</div>', unsafe_allow_html=True)

    if zone_risk is not None:
        c1, c2 = st.columns([1, 1])

        with c1:
            # Risk tier bar chart
            color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
            fig = px.bar(
                zone_risk,
                x="zone_name", y="capacity_usage_pct",
                color="risk_tier",
                color_discrete_map=color_map,
                text="risk_tier",
                labels={"capacity_usage_pct": "Capacity Usage (%)", "zone_name": "Zone"},
                title="Zone Capacity Usage & Risk Tier",
            )
            fig.update_traces(textposition="outside", textfont_size=13)
            fig.add_hline(y=85, line_dash="dash", line_color="#ef4444",
                         annotation_text="High Risk Threshold (85%)")
            fig.update_layout(
                paper_bgcolor="#0f1525", plot_bgcolor="#141e35",
                font_color="#c8d6f0", title_font_size=16,
                xaxis=dict(gridcolor="#1e2a45"),
                yaxis=dict(gridcolor="#1e2a45"),
                legend=dict(bgcolor="#1a2540"),
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Exceedance rate
            fig2 = px.bar(
                zone_risk,
                x="zone_name", y="exceedance_rate_pct",
                color="risk_tier",
                color_discrete_map=color_map,
                labels={"exceedance_rate_pct": "Exceedance Rate (%)", "zone_name": "Zone"},
                title="Demand Exceedance Probability",
            )
            fig2.update_layout(
                paper_bgcolor="#0f1525", plot_bgcolor="#141e35",
                font_color="#c8d6f0", title_font_size=16,
                xaxis=dict(gridcolor="#1e2a45"),
                yaxis=dict(gridcolor="#1e2a45"),
                legend=dict(bgcolor="#1a2540"),
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Zone Risk Summary Table")
        display_cols = ["zone_name", "risk_tier", "peak_forecast_kwh", "capacity_kw",
                        "capacity_usage_pct", "exceedance_rate_pct", "model_mape_pct"]

        def color_risk(val):
            colors = {"High": "color: #ef4444; font-weight: bold",
                      "Medium": "color: #f59e0b; font-weight: bold",
                      "Low": "color: #22c55e; font-weight: bold"}
            return colors.get(val, "")

        styled = zone_risk[display_cols].style.map(color_risk, subset=["risk_tier"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Zone forecast chart
    if forecasts is not None:
        st.markdown("#### 24-Hour Demand Forecast by Zone")
        selected_zone = st.selectbox("Select Zone", forecasts["zone_id"].unique(), key="zone_sel")
        z_fc = forecasts[forecasts["zone_id"] == selected_zone].sort_values("timestamp")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=z_fc["timestamp"], y=z_fc["actual_kwh"],
            name="Actual", line=dict(color="#60a5fa", width=2),
        ))
        fig3.add_trace(go.Scatter(
            x=z_fc["timestamp"], y=z_fc["forecast_kwh"],
            name="Forecast (P50)", line=dict(color="#f59e0b", width=2, dash="dash"),
        ))
        fig3.add_trace(go.Scatter(
            x=z_fc["timestamp"], y=z_fc["forecast_p90_kwh"],
            name="Forecast (P90)", line=dict(color="#ef4444", width=1, dash="dot"),
        ))
        cap = z_fc["capacity_kw"].iloc[0]
        fig3.add_hline(y=cap, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Capacity: {cap} kW")
        fig3.update_layout(
            paper_bgcolor="#0f1525", plot_bgcolor="#141e35",
            font_color="#c8d6f0",
            xaxis=dict(gridcolor="#1e2a45", title="Time"),
            yaxis=dict(gridcolor="#1e2a45", title="kWh"),
            legend=dict(bgcolor="#1a2540"),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — Inspection Alerts
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Ranked Inspection Alert List</div>', unsafe_allow_html=True)

    if alerts is not None:
        tier_filter = st.multiselect(
            "Filter by Alert Tier",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"],
            key="tier_filter"
        )
        filtered = alerts[alerts["alert_tier"].isin(tier_filter)]

        # Alert cards for top 10
        st.markdown(f"**Showing {len(filtered)} alerts** (top 10 displayed as cards below)")
        for _, row in filtered.head(10).iterrows():
            tier = str(row["alert_tier"]).lower()
            tier_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(tier, "⚪")
            sigs = int(row.get("signals_triggered", 0))
            sig_label = {0: "", 1: "1-signal", 2: "2-signal confirmed", 3: "3-signal confirmed"}.get(sigs, "3-signal confirmed")
            st.markdown(f"""
            <div class="alert-card {tier}">
                <div class="alert-meter">
                    {tier_emoji} #{row['priority_rank']} — Meter {row['meter_id']} | {row.get('zone_name','N/A')}
                    &nbsp;&nbsp;<span style="color:#94a3b8;font-size:0.85rem">Risk: {row['risk_score']:.3f} | {sig_label}</span>
                </div>
                <div style="display:flex;gap:16px;margin:6px 0;font-size:0.8rem;color:#64748b">
                    <span>L1 Stat: <b style='color:#60a5fa'>{row.get('l1_score',0):.2f}</b></span>
                    <span>L2 IF: <b style='color:#f59e0b'>{row.get('if_score',0):.2f}</b></span>
                    <span>L3 Peer: <b style='color:#a78bfa'>{row.get('peer_score',0):.2f}</b></span>
                </div>
                <div class="alert-reason">⚠️ {row['reason_codes']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Full Alert Table")
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        # Risk score distribution
        fig4 = px.histogram(
            alerts, x="risk_score", color="alert_tier",
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"},
            nbins=30, title="Risk Score Distribution",
            labels={"risk_score": "Composite Risk Score", "count": "Number of Meters"},
        )
        fig4.update_layout(
            paper_bgcolor="#0f1525", plot_bgcolor="#141e35",
            font_color="#c8d6f0", xaxis=dict(gridcolor="#1e2a45"),
            yaxis=dict(gridcolor="#1e2a45"), legend=dict(bgcolor="#1a2540"),
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Zone affinity results
        if affinity is not None:
            mismatched = affinity[affinity["zone_mismatch_flag"] == 1]
            if len(mismatched) > 0:
                st.markdown("#### 🔀 Zone Affinity — Potential Mis-Tagged Meters")
                st.info(f"**{len(mismatched)} meters** have load shapes inconsistent with their declared zone/feeder.")
                st.dataframe(mismatched, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# TAB 3 — Meter Drilldown
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Individual Meter Analysis</div>', unsafe_allow_html=True)

    if readings is not None and flags is not None:
        all_meters = sorted(readings["meter_id"].unique())
        sel_meter = st.selectbox("Select Meter", all_meters, key="meter_sel")

        m_readings = readings[readings["meter_id"] == sel_meter].sort_values("timestamp")
        m_flags = flags[flags["meter_id"] == sel_meter].sort_values("timestamp") if "meter_id" in flags.columns else None

        # Metadata
        if meters_meta is not None:
            m_meta = meters_meta[meters_meta["meter_id"] == sel_meter]
            if len(m_meta) > 0:
                row = m_meta.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Zone", row["zone_name"])
                c2.metric("Base Scale", f"{row['base_scale']:.2f}x")
                c3.metric("Fraud Type", row["fraud_type"])
                c4.metric("Ground Truth", "🔴 Fraud" if row["is_fraud"] else "🟢 Normal")

        # Reading chart with anomaly overlay
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=m_readings["timestamp"], y=m_readings["kwh"],
            name="kWh Reading", line=dict(color="#60a5fa", width=1.5),
            mode="lines",
        ))

        if m_flags is not None and "persistent_anomaly" in m_flags.columns:
            anomalies = m_flags[m_flags["persistent_anomaly"] == 1]
            if len(anomalies) > 0:
                fig5.add_trace(go.Scatter(
                    x=anomalies["timestamp"], y=anomalies["kwh"],
                    mode="markers", name="Anomaly Flagged",
                    marker=dict(color="#ef4444", size=5, symbol="circle"),
                ))

        fig5.update_layout(
            title=f"Meter {sel_meter} — Consumption History (90 days)",
            paper_bgcolor="#0f1525", plot_bgcolor="#141e35",
            font_color="#c8d6f0",
            xaxis=dict(gridcolor="#1e2a45", title="Date"),
            yaxis=dict(gridcolor="#1e2a45", title="kWh (15-min)"),
            legend=dict(bgcolor="#1a2540"),
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Daily total chart
        m_daily = m_readings.copy()
        m_daily["date"] = m_daily["timestamp"].dt.date
        daily = m_daily.groupby("date")["kwh"].sum().reset_index()

        fig6 = px.bar(daily, x="date", y="kwh",
                     title=f"Daily Consumption — {sel_meter}",
                     labels={"kwh": "Daily kWh", "date": "Date"},
                     color_discrete_sequence=["#3b82f6"])
        fig6.update_layout(
            paper_bgcolor="#0f1525", plot_bgcolor="#141e35",
            font_color="#c8d6f0",
            xaxis=dict(gridcolor="#1e2a45"),
            yaxis=dict(gridcolor="#1e2a45"),
        )
        st.plotly_chart(fig6, use_container_width=True)

        # Alert entry for this meter
        if alerts is not None:
            m_alert = alerts[alerts["meter_id"] == sel_meter]
            if len(m_alert) > 0:
                row = m_alert.iloc[0]
                tier_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}.get(str(row["alert_tier"]), "#94a3b8")
                st.markdown(f"""
                <div style="background:#141e35;border-left:4px solid {tier_color};border-radius:8px;padding:16px;margin-top:10px">
                    <b style="color:{tier_color}">Alert Tier: {row['alert_tier']}</b> &nbsp;|&nbsp;
                    Risk Score: <b>{row['risk_score']:.3f}</b> &nbsp;|&nbsp;
                    Priority Rank: <b>#{int(row['priority_rank'])}</b><br>
                    <span style="color:#94a3b8;font-size:0.9rem">⚠️ {row['reason_codes']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No active alert for this meter.")
    else:
        st.info("Run the pipeline to generate meter data.")


# ════════════════════════════════════════════════════════════════════
# TAB 4 — Model Performance
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Model Evaluation & Validation</div>', unsafe_allow_html=True)

    if fc_metrics is not None:
        st.markdown("#### Demand Forecasting — MAPE vs Baseline")
        fig7 = go.Figure()
        fig7.add_trace(go.Bar(
            name="Model MAPE", x=fc_metrics["zone_id"],
            y=fc_metrics["model_mape_pct"], marker_color="#3b82f6",
        ))
        fig7.add_trace(go.Bar(
            name="Baseline MAPE", x=fc_metrics["zone_id"],
            y=fc_metrics["baseline_mape_pct"], marker_color="#6b7280",
        ))
        fig7.update_layout(
            barmode="group",
            paper_bgcolor="#0f1525", plot_bgcolor="#141e35",
            font_color="#c8d6f0",
            xaxis=dict(gridcolor="#1e2a45", title="Zone"),
            yaxis=dict(gridcolor="#1e2a45", title="MAPE (%)"),
            legend=dict(bgcolor="#1a2540"),
        )
        st.plotly_chart(fig7, use_container_width=True)
        st.dataframe(fc_metrics, use_container_width=True, hide_index=True)

    if an_metrics is not None:
        st.markdown("#### Anomaly Detection — Precision / Recall / F1")
        row = an_metrics.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Precision", f"{float(row['precision']):.1%}")
        c2.metric("Recall", f"{float(row['recall']):.1%}")
        c3.metric("F1 Score", f"{float(row['f1']):.3f}")
        c4.metric("False Positive Rate", f"{float(row['fpr']):.1%}")
        c5.metric("PR-AUC", f"{float(row.get('pr_auc', 0)):.3f}")

        ch1, ch2 = st.columns(2)
        with ch1:
            # Confusion matrix
            tp, fp, tn, fn = int(row["tp"]), int(row["fp"]), int(row["tn"]), int(row["fn"])
            cm_data = [[tn, fp], [fn, tp]]
            fig8 = px.imshow(
                cm_data, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual"),
                x=["Normal", "Anomaly"], y=["Normal", "Anomaly"],
                title="Confusion Matrix",
            )
            fig8.update_layout(paper_bgcolor="#0f1525", font_color="#c8d6f0")
            st.plotly_chart(fig8, use_container_width=True)

        with ch2:
            # Precision-Recall curve
            if pr_curve is not None:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=pr_curve["recall"], y=pr_curve["precision"],
                    mode="lines+markers", name="PR Curve",
                    line=dict(color="#60a5fa", width=2),
                    marker=dict(size=4),
                    text=pr_curve["threshold"].round(2),
                    hovertemplate="Threshold: %{text}<br>Precision: %{y:.2f}<br>Recall: %{x:.2f}",
                ))
                # Mark optimal threshold
                best = pr_curve.loc[pr_curve["f1"].idxmax()]
                fig_pr.add_trace(go.Scatter(
                    x=[best["recall"]], y=[best["precision"]],
                    mode="markers", name=f"Optimal (t={best['threshold']:.2f})",
                    marker=dict(color="#ef4444", size=12, symbol="star"),
                ))
                fig_pr.update_layout(
                    title="Precision-Recall Curve",
                    paper_bgcolor="#0f1525", plot_bgcolor="#141e35",
                    font_color="#c8d6f0",
                    xaxis=dict(title="Recall", gridcolor="#1e2a45", range=[0, 1.05]),
                    yaxis=dict(title="Precision", gridcolor="#1e2a45", range=[0, 1.05]),
                    legend=dict(bgcolor="#1a2540"),
                )
                st.plotly_chart(fig_pr, use_container_width=True)

    # SHAP plots
    shap_files = list(SHAP_DIR.glob("shap_*.png")) if SHAP_DIR.exists() else []
    if shap_files:
        st.markdown("#### SHAP Feature Importance — Forecast Drivers")
        cols = st.columns(min(len(shap_files), 3))
        for i, f in enumerate(shap_files[:3]):
            with cols[i % 3]:
                st.image(str(f), caption=f.stem.replace("shap_", "Zone "))

# ════════════════════════════════════════════════════════════════════
# TAB 5 — Economic Impact Calculator
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">💰 Economic Impact Calculator</div>', unsafe_allow_html=True)
    st.markdown("Estimate the **revenue recovery potential** of deploying GridRakshak AI across BESCOM's network.")

    st.markdown("#### Network Scale Parameters")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        total_meters = st.number_input("Total Smart Meters Deployed", min_value=1000, max_value=5000000,
                                       value=2500000, step=50000, format="%d")
    with sc2:
        flagged_pct = st.slider("% Meters Flagged as Suspicious", 1, 30, 8)
    with sc3:
        confirmed_fraud_pct = st.slider("% of Flagged Confirmed as Fraud (Precision)",
                                        10, 100,
                                        int(float(an_metrics['precision'].iloc[0])*100) if an_metrics is not None else 46)

    st.markdown("#### Consumption & Tariff Parameters")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        avg_monthly_kwh = st.number_input("Avg Monthly Consumption per Fraud Meter (kWh)",
                                           100, 5000, 300, 50)
    with tc2:
        tariff = st.number_input("Average Electricity Tariff (₹/kWh)", 3.0, 15.0, 7.5, 0.5)
    with tc3:
        inspection_cost = st.number_input("Cost per Physical Inspection (₹)", 100, 5000, 800, 100)

    # ── Calculations ───────────────────────────────────────────────────────────
    flagged_meters      = int(total_meters * flagged_pct / 100)
    fraud_meters        = int(flagged_meters * confirmed_fraud_pct / 100)
    false_inspections   = flagged_meters - fraud_meters

    annual_kwh_stolen   = fraud_meters * avg_monthly_kwh * 12
    annual_revenue_lost = annual_kwh_stolen * tariff
    recoverable_revenue = annual_revenue_lost  # 100% of confirmed fraud

    inspection_cost_total = flagged_meters * inspection_cost
    net_gain              = recoverable_revenue - inspection_cost_total
    roi_pct               = (net_gain / (inspection_cost_total + 1e-6)) * 100

    # BESCOM AT&C loss context
    bescom_revenue  = 12000  # crore
    atc_loss_crore  = bescom_revenue * 0.08
    our_recovery    = recoverable_revenue / 1e7  # to crore

    st.markdown("---")
    st.markdown("#### 📊 Projected Impact")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fraud Meters Caught", f"{fraud_meters:,}",
              f"out of {flagged_meters:,} flagged")
    m2.metric("Annual Revenue Recovery", f"₹{recoverable_revenue/1e7:.1f} Cr",
              f"{our_recovery/atc_loss_crore*100:.1f}% of AT&C losses")
    m3.metric("Inspection Cost", f"₹{inspection_cost_total/1e7:.2f} Cr",
              f"{false_inspections:,} false trips")
    m4.metric("Net ROI", f"{roi_pct:.0f}%",
              f"₹{net_gain/1e7:.1f} Cr net gain")

    # Revenue waterfall chart
    fig_wf = go.Figure(go.Waterfall(
        name="₹ Crore",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Revenue Lost to Fraud", "Revenue Recovered", "Inspection Costs", "Net Gain"],
        y=[annual_revenue_lost/1e7, recoverable_revenue/1e7,
           -inspection_cost_total/1e7, 0],
        connector=dict(line=dict(color="#2a3a60")),
        increasing=dict(marker=dict(color="#22c55e")),
        decreasing=dict(marker=dict(color="#ef4444")),
        totals=dict(marker=dict(color="#60a5fa")),
    ))
    fig_wf.update_layout(
        title="Revenue Impact Waterfall (₹ Crore)",
        paper_bgcolor="#0f1525", plot_bgcolor="#141e35",
        font_color="#c8d6f0", showlegend=False,
        yaxis=dict(title="₹ Crore", gridcolor="#1e2a45"),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Scale comparison bar
    st.markdown("#### BESCOM Context")
    ctx_data = pd.DataFrame({
        "Category": ["BESCOM Annual Revenue", "Estimated AT&C Losses (8%)",
                     "GridRakshak Recovery Potential", "Net Gain after Inspection Cost"],
        "Amount (₹ Crore)": [bescom_revenue, atc_loss_crore,
                              round(our_recovery, 1), round(net_gain/1e7, 1)],
        "Color": ["#3b82f6", "#ef4444", "#f59e0b", "#22c55e"],
    })
    fig_ctx = px.bar(ctx_data, x="Category", y="Amount (₹ Crore)",
                     color="Category", color_discrete_sequence=ctx_data["Color"].tolist(),
                     title="GridRakshak Impact vs BESCOM Scale")
    fig_ctx.update_layout(
        paper_bgcolor="#0f1525", plot_bgcolor="#141e35",
        font_color="#c8d6f0", showlegend=False,
        xaxis=dict(gridcolor="#1e2a45"),
        yaxis=dict(title="₹ Crore", gridcolor="#1e2a45"),
    )
    st.plotly_chart(fig_ctx, use_container_width=True)

    st.info("""
    **Assumptions:** BESCOM serves ~2.8 million consumers, annual revenue ~₹12,000 Cr,
    AT&C losses ~8%. Tariff blended across residential/commercial/industrial.
    Recovery assumes confirmed fraud meters are fully remediated in the same year.
    """)

st.divider()
st.markdown(
    "<center><span style='color:#4a5568;font-size:0.8rem'>"
    "GridRakshak AI · BESCOM Smart Meter Intelligence · Hackathon 2026"
    "</span></center>",
    unsafe_allow_html=True,
)
