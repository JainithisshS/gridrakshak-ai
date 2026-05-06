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
    page_title="GridRakshak AI",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

* { font-family: 'Poppins', sans-serif; }

html, body, [class*="css"] { 
    font-family: 'Poppins', sans-serif;
    background: #0a0a0a;
    color: #e8e8e8;
}

.main { background: #0a0a0a; }
[data-testid="stSidebar"] { 
    background: #121212; 
    border-right: 1px solid #2a2a2a; 
}
[data-testid="stSidebar"] * { color: #d8d8d8 !important; }

/* Typography */
h1, h2, h3, h4, h5, h6 { 
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

/* Metric Cards */
.metric-card {
    background: #121212;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover { 
    transform: translateY(-2px); 
    border-color: #10b981;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.1);
}
.metric-value { 
    font-size: 2.4rem; 
    font-weight: 800; 
    color: #10b981; 
    margin: 8px 0 4px 0;
    letter-spacing: -0.5px;
}
.metric-label { 
    font-size: 0.8rem; 
    color: #9a9a9a; 
    text-transform: uppercase; 
    letter-spacing: 0.08em;
    font-weight: 600;
}
.metric-delta { 
    font-size: 0.75rem; 
    margin-top: 6px; 
    color: #9a9a9a;
    font-weight: 500;
}

/* Status Colors - Professional */
.risk-high { color: #ef4444 !important; font-weight: 700; }
.risk-medium { color: #f59e0b !important; font-weight: 700; }
.risk-low { color: #10b981 !important; font-weight: 700; }

/* Alert Cards */
.alert-card {
    background: #121212;
    border-left: 3px solid #10b981;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.alert-card:hover { 
    border-left-color: #34d399;
    background: #1a1a1a;
}
.alert-card.medium { border-left-color: #f59e0b; }
.alert-card.low { border-left-color: #10b981; }
.alert-meter { 
    font-size: 1rem; 
    font-weight: 700; 
    color: #d8d8d8;
    letter-spacing: -0.3px;
}
.alert-reason { 
    font-size: 0.8rem; 
    color: #9a9a9a; 
    margin-top: 6px; 
    font-weight: 500;
}

/* Section Headers */
.section-header {
    font-size: 1.3rem; 
    font-weight: 800; 
    color: #e8e8e8;
    border-bottom: 2px solid #2a2a2a;
    padding-bottom: 12px; 
    margin-bottom: 24px;
    letter-spacing: -0.4px;
}

/* Tables */
[data-testid="stDataFrame"] { 
    background: #0a0a0a !important;
    border-radius: 8px; 
    border: 1px solid #2a2a2a;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { 
    background: transparent;
    border-bottom: 2px solid #2a2a2a;
    gap: 8px;
}
.stTabs [data-baseweb="tab"] { 
    background: transparent;
    color: #9a9a9a;
    border-radius: 6px 6px 0 0;
    font-weight: 600;
    border: none;
    padding: 12px 20px !important;
}
.stTabs [aria-selected="true"] { 
    background: transparent;
    color: #10b981 !important; 
    border-bottom: 3px solid #10b981;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    background: #10b981 !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #059669 !important;
    transform: translateY(-1px) !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
    background: #0a0a0a !important;
    border: 1px solid #2a2a2a !important;
    color: #e8e8e8 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}

.stSelectbox > div > div > select option {
    background: #121212 !important;
    color: #e8e8e8 !important;
}

/* Info/Warning/Error boxes */
.stInfo, [data-testid="stInformationBox"] {
    background: #0d2817 !important;
    border-left: 4px solid #10b981 !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
}

.stWarning {
    background: #2d2a0d !important;
    border-left: 4px solid #f59e0b !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
}

.stError {
    background: #2d0d0d !important;
    border-left: 4px solid #ef4444 !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
}

.stSuccess {
    background: #0d2817 !important;
    border-left: 4px solid #10b981 !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
}

/* Selectbox/Multiselect */
.stMultiSelect > div > div {
    background: #0a0a0a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
}
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


# ── Sidebar Toggle State ────────────────────────────────────────────────────────
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### GridRakshak AI")
    st.markdown("Smart Meter Intelligence System", help="BESCOM Decision Support Platform")
    st.divider()

    # ── Dataset selector ────────────────────────────────────────────────────
    st.markdown("**Data Selection**")
    dataset_choice = st.radio(
        "Dataset to analyze:",
        options=["Synthetic BESCOM", "Real UCI Household"],
        index=0, label_visibility="collapsed",
    )
    IS_REAL = dataset_choice == "Real UCI Household"
    PREFIX  = "real_data_" if IS_REAL else ""

    st.divider()
    st.markdown("**System Status**")
    if check_outputs():
        st.success("Pipeline outputs ready")
        if IS_REAL and not (OUTPUTS / "real_data_zone_risk_table.csv").exists():
            st.warning("Real data outputs missing. Re-run pipeline.")
    else:
        st.error("Run pipeline first")

    st.divider()
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown("**About**")
    st.markdown("""
**GridRakshak AI** provides intelligent analysis of smart meter data:

- 24h demand forecasting
- Theft & anomaly detection
- Zone affinity analysis
- Explainable AI insights
""")
    st.divider()
    st.caption("Built with LightGBM, SHAP, DTW")
    st.caption("BESCOM Pilot - v2.0")

# ── Sidebar Hide CSS ────────────────────────────────────────────────────────────
if not st.session_state.sidebar_open:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Show Sidebar Toggle (when hidden) ───────────────────────────────────────────
if not st.session_state.sidebar_open:
    col1, col2 = st.columns([20, 1])
    with col2:
        if st.button("☰", help="Show sidebar", key="show_sidebar"):
            st.session_state.sidebar_open = True
            st.rerun()

# ── Main Content ─────────────────────────────────────────────────────────────────
st.markdown("# GridRakshak AI")
st.markdown("Intelligent smart meter analytics for demand forecasting and anomaly detection")

if not check_outputs():
    st.warning("No output data found. Run the pipeline first:")
    st.code("python src/pipeline.py", language="bash")
    st.stop()

# ── Dataset context banner ────────────────────────────────────────────────
if IS_REAL:
    st.info(
        "**Real UCI Household Dataset** — Individual household electric power consumption "
        "(2006–2010, 2M rows, 1-min intervals). Fraud patterns injected into real baseline for "
        "ground-truth validation."
    )
else:
    st.info(
        "**Synthetic BESCOM Dataset** — 8 real Bangalore zones, 80 meters, 90 days, 691K readings. "
        "Includes area-type profiles, temperature-driven AC load, festival boosts, and 12 fraud meters."
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

def kpi(col, value, label, delta=None, color="#58a6ff"):
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

kpi(col1, high_zones, "High Risk Zones", "Action Required", "#dc3545")
kpi(col2, high_alerts, "High Priority Alerts", "Immediate Inspection", "#6f42c1")
kpi(col3, med_alerts, "Medium Priority", "Monitor Closely", "#6f42c1")
kpi(col4, avg_mape, "Avg Forecast MAPE", "vs baseline", "#198754")
kpi(col5, prec, "Detection Precision", "anomaly accuracy", "#58a6ff")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Zone Risk Map",
    "Inspection Alerts",
    "Meter Drilldown",
    "Model Performance",
    "Economic Impact",
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — Zone Risk Map
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Zone Risk Assessment</div>', unsafe_allow_html=True)

    if zone_risk is not None:
        c1, c2 = st.columns([1, 1])

        with c1:
            # Risk tier bar chart
            color_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
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
                paper_bgcolor="#0a0a0a", plot_bgcolor="#121212",
                font_color="#d8d8d8", title_font_size=16, title_font=dict(family="Poppins", size=16, color="#d8d8d8"),
                xaxis=dict(gridcolor="#2a2a2a"),
                yaxis=dict(gridcolor="#2a2a2a"),
                legend=dict(bgcolor="#121212"),
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
                paper_bgcolor="#0a0a0a", plot_bgcolor="#121212",
                font_color="#d8d8d8", title_font_size=16, title_font=dict(family="Poppins", size=16, color="#d8d8d8"),
                xaxis=dict(gridcolor="#2a2a2a"),
                yaxis=dict(gridcolor="#2a2a2a"),
                legend=dict(bgcolor="#121212"),
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Risk Summary Table**")
        display_cols = ["zone_name", "risk_tier", "peak_forecast_kwh", "capacity_kw",
                        "capacity_usage_pct", "exceedance_rate_pct", "model_mape_pct"]

        def color_risk(val):
            colors = {"High": "color: #dc3545; font-weight: 700",
                      "Medium": "color: #6f42c1; font-weight: 700",
                      "Low": "color: #198754; font-weight: 700"}
            return colors.get(val, "")

        styled = zone_risk[display_cols].style.applymap(color_risk, subset=["risk_tier"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Zone forecast chart
    if forecasts is not None:
        st.markdown("**Forecast by Zone**")
        selected_zone = st.selectbox("Select Zone", forecasts["zone_id"].unique(), key="zone_sel")
        z_fc = forecasts[forecasts["zone_id"] == selected_zone].sort_values("timestamp")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=z_fc["timestamp"], y=z_fc["actual_kwh"],
            name="Actual", line=dict(color="#10b981", width=2.5),
        ))
        fig3.add_trace(go.Scatter(
            x=z_fc["timestamp"], y=z_fc["forecast_kwh"],
            name="Forecast (P50)", line=dict(color="#34d399", width=2, dash="dash"),
        ))
        fig3.add_trace(go.Scatter(
            x=z_fc["timestamp"], y=z_fc["forecast_p90_kwh"],
            name="Forecast (P90)", line=dict(color="#6ee7b7", width=1.5, dash="dot"),
        ))
        cap = z_fc["capacity_kw"].iloc[0]
        fig3.add_hline(y=cap, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Capacity: {cap} kW")
        fig3.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#121212",
            font_color="#d8d8d8", title_font=dict(family="Poppins", size=14, color="#d8d8d8"),
            xaxis=dict(gridcolor="#2a2a2a", title="Time"),
            yaxis=dict(gridcolor="#2a2a2a", title="kWh"),
            legend=dict(bgcolor="#121212"),
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
        st.markdown(f"**Showing {len(filtered)} alerts** (top 10 displayed)")
        for _, row in filtered.head(10).iterrows():
            tier = str(row["alert_tier"]).lower()
            tier_color = {"high": "#dc3545", "medium": "#6f42c1", "low": "#198754"}.get(tier, "#58a6ff")
            sigs = int(row.get("signals_triggered", 0))
            sig_label = {0: "", 1: "1-signal", 2: "2-signal confirmed", 3: "3-signal confirmed"}.get(sigs, "3-signal confirmed")
            st.markdown(f"""
            <div class="alert-card {tier}">
                <div class="alert-meter" style="color: {tier_color};">
                    Priority #{int(row['priority_rank'])} — Meter {row['meter_id']} | {row.get('zone_name','N/A')}
                    &nbsp;&nbsp;<span style="color:#8b949e;font-size:0.85rem">Risk: {row['risk_score']:.3f} | {sig_label}</span>
                </div>
                <div style="display:flex;gap:16px;margin:8px 0;font-size:0.8rem;color:#8b949e">
                    <span>L1 Stat: <b style='color:#58a6ff'>{row.get('l1_score',0):.2f}</b></span>
                    <span>L2 IF: <b style='color:#79c0ff'>{row.get('if_score',0):.2f}</b></span>
                    <span>L3 Peer: <b style='color:#a5d6ff'>{row.get('peer_score',0):.2f}</b></span>
                </div>
                <div class="alert-reason">{row['reason_codes']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**Complete Alert Table**")
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        # Risk score distribution
        fig4 = px.histogram(
            alerts, x="risk_score", color="alert_tier",
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
            nbins=30, title="Risk Score Distribution",
            labels={"risk_score": "Composite Risk Score", "count": "Number of Meters"},
        )
        fig4.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#121212",
            font_color="#d8d8d8", title_font=dict(family="Poppins", size=14, color="#d8d8d8"),
            xaxis=dict(gridcolor="#2a2a2a"),
            yaxis=dict(gridcolor="#2a2a2a"), legend=dict(bgcolor="#121212"),
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Zone affinity results
        if affinity is not None:
            mismatched = affinity[affinity["zone_mismatch_flag"] == 1]
            if len(mismatched) > 0:
                st.markdown("**Zone Affinity - Potential Mis-Tagged Meters**")
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
                c4.metric("Ground Truth", "Fraud" if row["is_fraud"] else "Normal")

        # Reading chart with anomaly overlay
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=m_readings["timestamp"], y=m_readings["kwh"],
            name="kWh Reading", line=dict(color="#10b981", width=2),
            mode="lines",
        ))

        if m_flags is not None and "persistent_anomaly" in m_flags.columns:
            anomalies = m_flags[m_flags["persistent_anomaly"] == 1]
            if len(anomalies) > 0:
                fig5.add_trace(go.Scatter(
                    x=anomalies["timestamp"], y=anomalies["kwh"],
                    mode="markers", name="Anomaly Flagged",
                    marker=dict(color="#ef4444", size=6, symbol="circle"),
                ))

        fig5.update_layout(
            title=f"Meter {sel_meter} — Consumption History",
            paper_bgcolor="#0a0a0a", plot_bgcolor="#121212",
            font_color="#d8d8d8", title_font=dict(family="Poppins", size=14, color="#d8d8d8"),
            xaxis=dict(gridcolor="#2a2a2a", title="Date"),
            yaxis=dict(gridcolor="#2a2a2a", title="kWh (15-min)"),
            legend=dict(bgcolor="#121212"),
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Daily total chart
        m_daily = m_readings.copy()
        m_daily["date"] = m_daily["timestamp"].dt.date
        daily = m_daily.groupby("date")["kwh"].sum().reset_index()

        fig6 = px.bar(daily, x="date", y="kwh",
                     title=f"Daily Consumption — {sel_meter}",
                     labels={"kwh": "Daily kWh", "date": "Date"},
                     color_discrete_sequence=["#10b981"])
        fig6.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#121212",
            font_color="#d8d8d8", title_font=dict(family="Poppins", size=14, color="#d8d8d8"),
            xaxis=dict(gridcolor="#2a2a2a"),
            yaxis=dict(gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig6, use_container_width=True)

        # Alert entry for this meter
        if alerts is not None:
            m_alert = alerts[alerts["meter_id"] == sel_meter]
            if len(m_alert) > 0:
                row = m_alert.iloc[0]
                tier_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}.get(str(row["alert_tier"]), "#9a9a9a")
                st.markdown(f"""
                <div style="background:#121212;border-left:3px solid {tier_color};border-radius:8px;padding:18px 20px;margin-top:12px">
                    <b style="color:{tier_color}">Alert Tier: {row['alert_tier']}</b> &nbsp;|&nbsp;
                    Risk Score: <b>{row['risk_score']:.3f}</b> &nbsp;|&nbsp;
                    Priority Rank: <b>#{int(row['priority_rank'])}</b><br>
                    <span style="color:#9a9a9a;font-size:0.9rem">{row['reason_codes']}</span>
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
        st.markdown("**Demand Forecasting — MAPE vs Baseline**")
        fig7 = go.Figure()
        fig7.add_trace(go.Bar(
            name="Model MAPE", x=fc_metrics["zone_id"],
            y=fc_metrics["model_mape_pct"], marker_color="#10b981",
        ))
        fig7.add_trace(go.Bar(
            name="Baseline MAPE", x=fc_metrics["zone_id"],
            y=fc_metrics["baseline_mape_pct"], marker_color="#6b7280",
        ))
        fig7.update_layout(
            barmode="group",
            paper_bgcolor="#0a0a0a", plot_bgcolor="#121212",
            font_color="#d8d8d8", title_font=dict(family="Poppins", size=14, color="#d8d8d8"),
            xaxis=dict(gridcolor="#2a2a2a", title="Zone"),
            yaxis=dict(gridcolor="#2a2a2a", title="MAPE (%)"),
            legend=dict(bgcolor="#121212"),
        )
        st.plotly_chart(fig7, use_container_width=True)
        st.dataframe(fc_metrics, use_container_width=True, hide_index=True)

    if an_metrics is not None:
        st.markdown("**Anomaly Detection — Precision / Recall / F1**")
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
                cm_data, text_auto=True, color_continuous_scale="Greys",
                labels=dict(x="Predicted", y="Actual"),
                x=["Normal", "Anomaly"], y=["Normal", "Anomaly"],
                title="Confusion Matrix",
            )
            fig8.update_layout(paper_bgcolor="#0a0a0a", font_color="#d8d8d8",
                             plot_bgcolor="#121212")
            st.plotly_chart(fig8, use_container_width=True)

        with ch2:
            # Precision-Recall curve
            if pr_curve is not None:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=pr_curve["recall"], y=pr_curve["precision"],
                    mode="lines+markers", name="PR Curve",
                    line=dict(color="#10b981", width=2.5),
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
                    paper_bgcolor="#0a0a0a", plot_bgcolor="#121212",
                    font_color="#d8d8d8", title_font=dict(family="Poppins", size=14, color="#d8d8d8"),
                    xaxis=dict(title="Recall", gridcolor="#2a2a2a", range=[0, 1.05]),
                    yaxis=dict(title="Precision", gridcolor="#2a2a2a", range=[0, 1.05]),
                    legend=dict(bgcolor="#121212"),
                )
                st.plotly_chart(fig_pr, use_container_width=True)

    # SHAP plots
    shap_files = list(SHAP_DIR.glob("shap_*.png")) if SHAP_DIR.exists() else []
    if shap_files:
        st.markdown("**SHAP Feature Importance — Forecast Drivers**")
        cols = st.columns(min(len(shap_files), 3))
        for i, f in enumerate(shap_files[:3]):
            with cols[i % 3]:
                st.image(str(f), caption=f.stem.replace("shap_", "Zone "))

# ════════════════════════════════════════════════════════════════════
# TAB 5 — Economic Impact Calculator
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Economic Impact Calculator</div>', unsafe_allow_html=True)
    st.markdown("Estimate the **revenue recovery potential** of deploying GridRakshak AI across the network.")

    st.markdown("**Network Scale Parameters**")
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

    st.markdown("**Consumption & Tariff Parameters**")
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
    st.markdown("**Projected Impact**")
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
        connector=dict(line=dict(color="#2a2a2a")),
        increasing=dict(marker=dict(color="#10b981")),
        decreasing=dict(marker=dict(color="#ef4444")),
        totals=dict(marker=dict(color="#f59e0b")),
    ))
    fig_wf.update_layout(
        title="Revenue Impact Waterfall (₹ Crore)",
        paper_bgcolor="#0a0a0a", plot_bgcolor="#121212",
        font_color="#d8d8d8", showlegend=False, title_font=dict(family="Poppins", size=14, color="#d8d8d8"),
        yaxis=dict(title="₹ Crore", gridcolor="#2a2a2a"),
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Scale comparison bar
    st.markdown("**Network Context**")
    ctx_data = pd.DataFrame({
        "Category": ["Annual Revenue", "Estimated AT&C Losses (8%)",
                     "GridRakshak Recovery Potential", "Net Gain after Inspection Cost"],
        "Amount (₹ Crore)": [bescom_revenue, atc_loss_crore,
                              round(our_recovery, 1), round(net_gain/1e7, 1)],
        "Color": ["#10b981", "#ef4444", "#f59e0b", "#34d399"],
    })
    fig_ctx = px.bar(ctx_data, x="Category", y="Amount (₹ Crore)",
                     color="Category", color_discrete_sequence=ctx_data["Color"].tolist(),
                     title="GridRakshak Impact vs Network Scale")
    fig_ctx.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#121212",
        font_color="#d8d8d8", showlegend=False, title_font=dict(family="Poppins", size=14, color="#d8d8d8"),
        xaxis=dict(gridcolor="#2a2a2a"),
        yaxis=dict(title="₹ Crore", gridcolor="#2a2a2a"),
    )
    st.plotly_chart(fig_ctx, use_container_width=True)

    st.info("""
    **Key Assumptions:** Network statistics based on typical utility parameters. Fraud detection accuracy 
    from model performance metrics. Recovery assumes confirmed fraud meters are fully remediated in the same year.
    """)

st.divider()
st.markdown(
    "<center><span style='color:#6e7681;font-size:0.8rem'>"
    "GridRakshak AI · Smart Meter Intelligence · 2026"
    "</span></center>",
    unsafe_allow_html=True,
)
