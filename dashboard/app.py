"""
app.py
======
Ben's Original Marketing Analytics & Sales Forecasting -- Streamlit
Dashboard (Part A of Assignment #2).

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

# Make the `src` package importable regardless of the working directory
# `streamlit run` was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from dashboard import charts  # noqa: E402
from src import ai_insights, config, data_prep, eda, explainability, forecasting  # noqa: E402

import importlib
importlib.reload(forecasting)

st.set_page_config(
    page_title="Ben's Original | Marketing Analytics & Forecasting",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        /* Global font override */
        html, body, [class*="css"] {{
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}

        /* KPI cards with glassmorphism */
        .kpi-card {{
            background: linear-gradient(135deg, {config.BRAND_PRIMARY} 0%, #102B43 100%);
            color: white;
            padding: 1.1rem 1.3rem;
            border-radius: 14px;
            text-align: left;
            box-shadow: 0 4px 24px rgba(28,76,116,0.18);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .kpi-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 32px rgba(28,76,116,0.28);
        }}
        .kpi-card h3 {{
            margin: 0;
            font-size: 0.78rem;
            font-weight: 500;
            opacity: 0.85;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .kpi-card p {{
            margin: 0.3rem 0 0 0;
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }}

        /* Data Quality KPI cards (neutral) */
        .dq-kpi-card {{
            background: linear-gradient(135deg, #2B2A29 0%, #3d3c3b 100%);
            color: white;
            padding: 1rem 1.2rem;
            border-radius: 14px;
            text-align: left;
            box-shadow: 0 4px 18px rgba(43,42,41,0.15);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .dq-kpi-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 28px rgba(43,42,41,0.25);
        }}
        .dq-kpi-card h3 {{
            margin: 0;
            font-size: 0.75rem;
            font-weight: 500;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .dq-kpi-card p {{
            margin: 0.2rem 0 0 0;
            font-size: 1.4rem;
            font-weight: 700;
        }}
        .dq-kpi-card .accent {{
            color: {config.BRAND_SECONDARY};
        }}

        /* Methodology / callout box */
        .methodology-box {{
            background: linear-gradient(135deg, #FFF8E8 0%, #FFF3D6 100%);
            border-left: 4px solid {config.BRAND_SECONDARY};
            padding: 1rem 1.2rem;
            border-radius: 8px;
            font-size: 0.92rem;
            box-shadow: 0 2px 12px rgba(245,158,11,0.08);
        }}

        /* Insight box */
        .insight-box {{
            background: linear-gradient(135deg, #F7F4F2 0%, #F0EDEB 100%);
            border-left: 4px solid {config.BRAND_PRIMARY};
            padding: 0.8rem 1.1rem;
            border-radius: 8px;
            margin-bottom: 0.7rem;
            font-size: 0.93rem;
            box-shadow: 0 1px 8px rgba(0,0,0,0.04);
            transition: border-left-width 0.15s ease;
        }}
        .insight-box:hover {{
            border-left-width: 6px;
        }}

        /* Severity badges */
        .severity-badge {{
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .severity-high {{
            background: rgba(239,68,68,0.12);
            color: #EF4444;
        }}
        .severity-moderate {{
            background: rgba(245,158,11,0.15);
            color: #F59E0B;
        }}
        .severity-low {{
            background: rgba(16,185,129,0.12);
            color: #10B981;
        }}

        /* Step cards for cleaning pipeline */
        .step-card {{
            background: white;
            border: 1px solid #E8E4E0;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .step-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        }}
        .step-card .step-icon {{
            font-size: 1.8rem;
            margin-bottom: 0.4rem;
        }}
        .step-card .step-title {{
            font-size: 0.82rem;
            font-weight: 600;
            color: {config.BRAND_DARK};
            margin-bottom: 0.2rem;
        }}
        .step-card .step-value {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {config.BRAND_PRIMARY};
        }}
        .step-card .step-desc {{
            font-size: 0.72rem;
            color: #888;
            margin-top: 0.15rem;
        }}

        /* Evidence callout */
        .evidence-callout {{
            background: linear-gradient(135deg, #EEF2FF 0%, #F8FAFC 100%);
            border: 1px solid rgba(28,76,116,0.15);
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            font-size: 0.9rem;
            box-shadow: 0 2px 12px rgba(28,76,116,0.06);
        }}
        .evidence-callout .ev-title {{
            font-weight: 700;
            color: {config.BRAND_PRIMARY};
            font-size: 0.95rem;
            margin-bottom: 0.4rem;
        }}

        /* Section header styling */
        .section-header {{
            background: linear-gradient(90deg, {config.BRAND_PRIMARY}08, transparent);
            border-left: 3px solid {config.BRAND_PRIMARY};
            padding: 0.5rem 0.9rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 0.8rem;
        }}
        .section-header h4 {{
            margin: 0;
            font-weight: 700;
            color: {config.BRAND_DARK};
            font-size: 1rem;
        }}

        /* Gradient dividers */
        .gradient-divider {{
            height: 2px;
            background: linear-gradient(90deg,
                {config.BRAND_PRIMARY} 0%,
                {config.BRAND_SECONDARY} 50%,
                transparent 100%);
            border: none;
            margin: 1.2rem 0;
            border-radius: 1px;
        }}

        /* Footer */
        .dashboard-footer {{
            background: linear-gradient(135deg, {config.BRAND_DARK} 0%, #1a1918 100%);
            color: rgba(255,255,255,0.7);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            text-align: center;
            margin-top: 2rem;
            font-size: 0.85rem;
        }}
        .dashboard-footer .footer-brand {{
            font-weight: 700;
            color: {config.BRAND_SECONDARY};
            font-size: 0.95rem;
        }}
        .dashboard-footer .footer-sub {{
            font-size: 0.78rem;
            opacity: 0.6;
            margin-top: 0.3rem;
        }}

        /* Action taken table styling */
        .action-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        }}
        .action-table th {{
            background: {config.BRAND_DARK};
            color: white;
            padding: 0.7rem 1rem;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            text-align: left;
        }}
        .action-table td {{
            padding: 0.6rem 1rem;
            font-size: 0.88rem;
            border-bottom: 1px solid #F0EDEB;
        }}
        .action-table tr:last-child td {{
            border-bottom: none;
        }}
        .action-table tr:nth-child(even) {{
            background: #FAFAF8;
        }}

        /* Forecast metric cards */
        .fc-metric-card {{
            background: white;
            border: 1px solid #E8E4E0;
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            text-align: center;
            box-shadow: 0 3px 16px rgba(0,0,0,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .fc-metric-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 28px rgba(0,0,0,0.1);
        }}
        .fc-metric-card .fc-label {{
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #888;
            margin-bottom: 0.3rem;
        }}
        .fc-metric-card .fc-value {{
            font-size: 1.5rem;
            font-weight: 800;
            color: {config.BRAND_DARK};
            letter-spacing: -0.02em;
        }}
        .fc-metric-card .fc-sub {{
            font-size: 0.72rem;
            color: #aaa;
            margin-top: 0.15rem;
        }}

        /* Model winner card */
        .model-winner {{
            background: linear-gradient(135deg, #10B981 0%, #059669 100%);
            color: white;
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            box-shadow: 0 4px 20px rgba(16,185,129,0.2);
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .model-winner .mw-icon {{
            font-size: 2rem;
        }}
        .model-winner .mw-text h4 {{
            margin: 0;
            font-size: 1rem;
            font-weight: 700;
        }}
        .model-winner .mw-text p {{
            margin: 0.2rem 0 0 0;
            font-size: 0.82rem;
            opacity: 0.85;
        }}

        /* Forecast summary card */
        .fc-summary-card {{
            background: linear-gradient(135deg, #F7F4F2 0%, #EFECEB 100%);
            border: 1px solid #E0DCD8;
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.03);
        }}
        .fc-summary-card h4 {{
            margin: 0 0 0.5rem 0;
            font-size: 0.95rem;
            font-weight: 700;
            color: {config.BRAND_DARK};
        }}
        .fc-summary-card .fc-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.35rem 0;
            border-bottom: 1px solid #E8E4E0;
            font-size: 0.88rem;
        }}
        .fc-summary-card .fc-row:last-child {{
            border-bottom: none;
        }}
        .fc-summary-card .fc-row-label {{
            color: #888;
        }}
        .fc-summary-card .fc-row-value {{
            font-weight: 700;
            color: {config.BRAND_DARK};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)



# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
def ensure_assets():
    required_files = [
        config.REPORTS_DIR / "row_level_clean.csv",
        config.REPORTS_DIR / "weekly_feature_table.csv",
        config.REPORTS_DIR / "weekly_sales.csv",
        config.REPORTS_DIR / "raw_data.csv",
        config.MODELS_DIR / "model_result.joblib",
        config.MODELS_DIR / "shap_data.joblib"
    ]
    if not all(f.exists() for f in required_files):
        with st.spinner("Initializing first-time setup: Running data prep and model training pipelines..."):
            try:
                import main
                main.main()
            except Exception as e:
                st.error(f"Failed to run the background analytics pipeline: {e}")
                st.stop()


ensure_assets()


@st.cache_data(show_spinner="Loading pre-computed data...")
def get_data():
    # Force cache reload after data prep script update
    row_df = pd.read_csv(config.REPORTS_DIR / "row_level_clean.csv", parse_dates=[config.DATE_COL])
    weekly_df = pd.read_csv(config.REPORTS_DIR / "weekly_feature_table.csv", parse_dates=[config.DATE_COL])
    weekly_sales = pd.read_csv(config.REPORTS_DIR / "weekly_sales.csv", parse_dates=[config.DATE_COL])
    raw_df = pd.read_csv(config.REPORTS_DIR / "raw_data.csv")
    return row_df, weekly_df, weekly_sales, raw_df


@st.cache_resource(show_spinner="Loading saved forecasting models...")
def get_models():
    # Force cache reload
    return joblib.load(config.MODELS_DIR / "model_result.joblib")


@st.cache_resource(show_spinner="Loading saved SHAP values...")
def get_shap():
    # Force cache reload
    shap_data = joblib.load(config.MODELS_DIR / "shap_data.joblib")
    return shap_data["shap_values"], shap_data["shap_importance"]


row_df, weekly_df, weekly_sales, raw_df = get_data()
model_result = get_models()


def kpi_card(col, label: str, value: str):
    col.markdown(f'<div class="kpi-card"><h3>{label}</h3><p>{value}</p></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar -- global filters (Deliverable 3: Filters)
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style="text-align:center; padding-top:1rem; padding-bottom:1.5rem;">
        <h1 style="margin:0; font-weight:800; color:{config.BRAND_DARK}; font-size:1.8rem; letter-spacing:-0.03em;">
            <span style="color:{config.BRAND_PRIMARY};">Ben's</span> Original
        </h1>
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:#888; margin-top:0.3rem;">
            Marketing Analytics & Forecasting
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()

filtered_row_df = row_df.copy()

st.sidebar.divider()
st.sidebar.caption(f"Rows in view: {len(filtered_row_df):,} / {len(row_df):,}")
st.sidebar.caption("Data: Jan 2023 - Apr 2025 (weekly)")

# Weekly aggregates recomputed from the FILTERED row-level data, used for
# the trend/correlation views that should react to the sidebar filters.
filtered_weekly = (
    filtered_row_df.groupby(config.DATE_COL)[config.ADDITIVE_METRIC_COLS]
    .sum()
    .reset_index()
)
filtered_weekly[config.DATE_COL] = pd.to_datetime(filtered_weekly[config.DATE_COL], format="mixed")
weekly_sales[config.DATE_COL] = pd.to_datetime(weekly_sales[config.DATE_COL], format="mixed")
filtered_weekly = (
    filtered_weekly
    .merge(weekly_sales, on=config.DATE_COL, how="left")
    .sort_values(config.DATE_COL)
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Ben's Original -- Marketing Analytics Dashboard & Sales Forecasting")
st.caption(
    "Internship Assignment #2 deliverable -- Adbureau Analytics. "
    "Use the tabs below to move through Data Quality, EDA, Trends, Sales Drivers, "
    "Correlation, Top Campaigns, Media Efficiency, Forecasting, Explainability, and AI Insights."
)

tabs = st.tabs(
    [
        "🏠 Summary",
        "🧪 Data Quality & EDA",
        "📈 Trends",
        "🎯 Drivers",
        "🔗 Correlation",
        "🏆 Campaigns",
        "⚙️ Efficiency",
        "🔮 Forecast",
        "🧠 Explain",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1: Executive Summary
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Executive Summary")
    kpis = eda.kpi_summary(row_df, weekly_sales)
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Total Sales", f"${kpis['Total Sales']/1e6:,.1f}M")
    kpi_card(c2, "Total Spend", f"${kpis['Total Spend']/1e6:,.2f}M")
    kpi_card(c3, "Total Impressions", f"{kpis['Total Impressions']/1e9:,.2f}B")
    kpi_card(c4, "Total Clicks", f"{kpis['Total Clicks']/1e6:,.2f}M")
    kpi_card(c5, "Total Engagements", f"{kpis['Total Engagements']/1e3:,.1f}K")

    st.caption(
        "Total Sales is the de-duplicated weekly sales figure summed across all known weeks -- "
        "**not** a raw sum of the `Sales` column (see the Methodology tab for why that distinction matters)."
    )

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Weekly Sales Trend (company-wide)")
        st.plotly_chart(
            charts.line_trend(weekly_sales.dropna(subset=["Total Sales"]), config.DATE_COL, "Total Sales", y_title="Sales ($)"),
            use_container_width=True,
        )
    with right:
        st.markdown("##### Estimated Sales Contribution by Media Type")
        mt = eda.breakdown_by(filtered_row_df, "Media Type")
        st.plotly_chart(charts.bar_breakdown(mt, "Media Type", "Estimated Sales Contribution"), use_container_width=True)

    st.markdown("##### Media Performance Snapshot")
    qcols = st.columns(5)
    snapshot_dims = ["Media Type", "Channel", "Site", "Device", "Platform Type"]
    for col, dim in zip(qcols, snapshot_dims):
        top = eda.breakdown_by(filtered_row_df, dim, top_n=1)
        if not top.empty:
            col.metric(f"Top {dim}", str(top.iloc[0, 0]), f"${top.iloc[0,1]/1e6:,.1f}M est. sales")


# ---------------------------------------------------------------------------
# TAB 2: Data Quality & Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Data Quality & Exploratory Data Analysis (EDA)")
    st.caption("A streamlined view of data quality and cleaned data distributions.")

    # --- Section 1: Data Quality Overview KPIs ---
    st.markdown('<div class="section-header"><h4>📋 Data Quality Overview</h4></div>', unsafe_allow_html=True)

    quality_stats = eda.data_quality_summary(raw_df, row_df)

    qc1, qc2, qc3, qc4, qc5, qc6 = st.columns(6)
    qc1.markdown(f'<div class="dq-kpi-card"><h3>Raw Rows</h3><p>{quality_stats["total_rows"]:,}</p></div>', unsafe_allow_html=True)
    qc2.markdown(f'<div class="dq-kpi-card"><h3>Columns</h3><p>{quality_stats["total_columns"]}</p></div>', unsafe_allow_html=True)
    qc3.markdown(f'<div class="dq-kpi-card"><h3>Missing Cells</h3><p class="accent">{quality_stats["total_missing_cells"]:,}</p></div>', unsafe_allow_html=True)
    qc4.markdown(f'<div class="dq-kpi-card"><h3>Missing %</h3><p>{quality_stats["missing_pct"]:.1f}%</p></div>', unsafe_allow_html=True)
    qc5.markdown(f'<div class="dq-kpi-card"><h3>Distinct Weeks</h3><p>{quality_stats["distinct_weeks"]}</p></div>', unsafe_allow_html=True)
    qc6.markdown(f'<div class="dq-kpi-card"><h3>Duplicates</h3><p>{quality_stats["duplicates_dropped"]}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # --- Section 2: Missing Values & Actions ---
    missing_detailed = eda.missingness_report_detailed(raw_df)
    
    st.markdown('<div class="section-header"><h4>🔍 Missing Values Analysis</h4></div>', unsafe_allow_html=True)
    mv_left, mv_right = st.columns([1, 1])
    with mv_left:
        st.plotly_chart(charts.missing_values_chart(missing_detailed, "Missing Values by Column"), use_container_width=True)
    with mv_right:
        st.markdown("##### Cleaning Actions Taken")
        st.markdown(
            "<ul>"
            "<li><b>Whitespace Stripped:</b> Leading/trailing spaces removed from all text columns</li>"
            "<li><b>Typos Fixed:</b> Addressed inconsistent capitalization (e.g. 'VIdeo' to 'Video')</li>"
            "<li><b>Structural NAs Filled:</b> Explicitly labeled N/A values in Audience/Platform</li>"
            "<li><b>Efficiency Ratios:</b> Made division-by-zero safe (returns NaN instead of Inf)</li>"
            "</ul>", unsafe_allow_html=True
        )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # --- Section 3: EDA (Exploratory Data Analysis) ---
    st.markdown('<div class="section-header"><h4>📊 Exploratory Data Analysis (EDA)</h4></div>', unsafe_allow_html=True)
    
    eda_l, eda_r = st.columns([3, 2])
    with eda_l:
        st.markdown("##### Cleaned Data Sample (First 10 Rows)")
        st.dataframe(row_df.head(10), use_container_width=True)
    with eda_r:
        st.markdown("##### Key Metrics Summary")
        # Ensure we only describe numeric columns that exist
        numeric_cols = [c for c in ["Spend", "Impressions", "Clicks", "Engagements", "Base Dollar Amount"] if c in row_df.columns]
        st.dataframe(row_df[numeric_cols].describe().T, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Trend Analysis
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Trend Analysis")
    st.caption("Weekly Sales is always company-wide. Spend, Impressions, and Engagement trends reflect the sidebar filters.")

    lookback_weeks = st.slider(
        "Lookback period (weeks)", min_value=12, max_value=122,
        value=min(52, len(weekly_sales)), step=4,
        help="Slide to zoom in on recent trends or zoom out to see the full history.",
    )

    trend_sales = weekly_sales.dropna(subset=["Total Sales"]).tail(lookback_weeks)
    trend_weekly = filtered_weekly.tail(lookback_weeks)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            charts.line_trend(trend_sales, config.DATE_COL, "Total Sales", "Weekly Sales Trend", "Sales ($)"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            charts.line_trend(trend_weekly, config.DATE_COL, "Spend", "Weekly Spend Trend", "Spend ($)"),
            use_container_width=True,
        )
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            charts.line_trend(trend_weekly, config.DATE_COL, "Impressions", "Weekly Impressions Trend", "Impressions"),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            charts.line_trend(trend_weekly, config.DATE_COL, "Engagements", "Weekly Engagement Trend", "Engagements"),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# TAB 4: Sales Drivers
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Sales Drivers")
    st.info(
        "⚠️ **Estimated, not audited.** There is no genuine row-level sales attribution in the source "
        "data -- only a single true sales figure per week. These bars allocate each week's true sales "
        "proportionally to each row's share of that week's media activity (Spend where available, "
        "otherwise a normalized Impressions/Reach/GRPs/Engagements/Likes intensity score). See the "
        "Methodology tab for full reasoning.",
        icon="⚠️",
    )

    driver_dims = ["Media Type", "Channel", "Site", "Device", "Platform Type"]
    cols = st.columns(2)
    for i, dim in enumerate(driver_dims):
        data = eda.breakdown_by(filtered_row_df, dim)
        with cols[i % 2]:
            st.plotly_chart(
                charts.bar_breakdown(data, dim, "Estimated Sales Contribution", f"Sales by {dim}", orientation="h"),
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# TAB 5: Correlation Analysis
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Correlation Analysis")
    st.caption(
        "Computed at the weekly-aggregate grain (media metrics summed per week against the "
        "de-duplicated weekly Sales figure) -- mixing row-level dimensions with a week-level Sales "
        "number would be statistically meaningless."
    )
    corr = eda.correlation_matrix(filtered_weekly, ["Spend", "Impressions", "Clicks", "Engagements", "Total Sales"])
    st.plotly_chart(charts.correlation_heatmap(corr, "Spend / Impressions / Clicks / Engagements / Sales"), use_container_width=True)
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("##### Spend & Sales Over Time")
    st.caption(
        "This timeline displays weekly Spend alongside de-duplicated Sales to identify "
        "lag effects, showing how marketing spend relates to sales performance chronologically."
    )
    st.plotly_chart(
        charts.spend_and_sales_over_time(filtered_weekly, "Weekly Spend & Total Sales Over Time"),
        use_container_width=True,
    )

    st.markdown(
        "**Reading this honestly:** correlations between any single weekly media metric and Total "
        "Sales are weak in this dataset (|r| typically < 0.15). Even with temporal alignment, the relationship "
        "is not always straightforward. That's a real, useful finding -- see the Explainability tab, "
        "where the forecasting model's own feature importances tell the same story."
    )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("##### 🧮 Simple Spend-to-Sales Estimator")
    st.caption("A naive linear estimate based purely on the historical weekly correlation (ignoring momentum and other features).")
    
    calc_left, calc_right = st.columns([1, 1])
    with calc_left:
        avg_spend = float(filtered_weekly["Spend"].mean())
        max_spend = float(filtered_weekly["Spend"].max())
        if pd.isna(avg_spend) or pd.isna(max_spend):
            avg_spend, max_spend = 0.0, 1000000.0
            
        sim_spend = st.number_input(
            "Enter hypothetical Weekly Spend ($)",
            min_value=0, max_value=int(max_spend * 5), value=int(avg_spend), step=10000,
            help="Input a spend amount to see the naive sales estimate."
        )
        
    with calc_right:
        valid_df = filtered_weekly.dropna(subset=["Spend", "Total Sales"])
        if len(valid_df) > 2:
            import numpy as np
            x = valid_df["Spend"].values
            y = valid_df["Total Sales"].values
            m, b = np.polyfit(x, y, 1)
            est_sales = (m * sim_spend) + b
            
            st.markdown(
                f'<div class="fc-metric-card" style="margin-top:1.2rem;"><div class="fc-label">Estimated Weekly Sales</div>'
                f'<div class="fc-value" style="color:{config.BRAND_PRIMARY}">${est_sales:,.0f}</div>'
                f'<div class="fc-sub">Based on naive linear trend (y = {m:.2f}x + {b:,.0f})</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Not enough data to calculate trend.")

# ---------------------------------------------------------------------------
# TAB 6: Top Campaigns
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Top 20 Creatives by Estimated Sales Contribution")
    top20 = eda.top_creatives(filtered_row_df, n=20)
    st.plotly_chart(
        charts.bar_breakdown(top20, "Creative", "Estimated Sales Contribution", orientation="h"),
        use_container_width=True,
    )
    st.dataframe(
        top20.style.format({"Estimated Sales Contribution": "${:,.0f}", "Spend": "${:,.0f}", "Impressions": "{:,.0f}", "Engagements": "{:,.0f}"}),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# TAB 7: Media Efficiency
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Media Efficiency Analysis")
    st.caption(
        "CTR, CPC, CPM, and Cost Per Engagement are re-derived from summed numerators/denominators "
        "per category (sum(Clicks)/sum(Impressions), etc.) rather than averaging the row-level ratio "
        "column directly -- averaging a ratio can be misleading when row volumes differ a lot (a "
        "classic Simpson's-paradox trap)."
    )
    for eff_dim in config.FILTER_DIMENSIONS:
        eff_table = eda.efficiency_by_dimension(filtered_row_df, eff_dim)
        
        # Skip if table is empty
        if len(eff_table) == 0:
            continue
            
        # Merge with sales data for the joint histogram
        sales_table = eda.breakdown_by(filtered_row_df, eff_dim)
        merged_table = pd.merge(eff_table, sales_table, on=eff_dim, how="left")
            
        # Clean missing values so Plotly doesn't drop them from the axis
        merged_table[eff_dim] = merged_table[eff_dim].fillna("Unknown").replace({"None": "Unknown", "nan": "Unknown", "NaN": "Unknown"})
        merged_table[eff_dim] = merged_table[eff_dim].astype(str)
            
        st.markdown(f'<div class="section-header"><h4>{eff_dim}</h4></div>', unsafe_allow_html=True)
        
        col_table, col_chart = st.columns([1, 1])
        
        with col_table:
            st.dataframe(
                merged_table.style.format(
                    {
                        "Spend": "${:,.0f}",
                        "Impressions": "{:,.0f}",
                        "Clicks": "{:,.0f}",
                        "Engagements": "{:,.0f}",
                        "CTR": "{:.3%}",
                        "CPC": "${:,.2f}",
                        "CPM": "${:,.2f}",
                        "Cost Per Engagement": "${:,.2f}",
                        "Estimated Sales Contribution": "${:,.0f}",
                    }
                ),
                use_container_width=True,
            )
            
        with col_chart:
            if eff_dim in ["Media Type", "Platform Type"]:
                pass
            elif eff_dim == "Channel":
                plot_data = merged_table.dropna(subset=["CTR"]).copy()
                if len(plot_data) > 0 and plot_data["CTR"].sum() > 0:
                    fig = charts.bar_breakdown(plot_data, eff_dim, "CTR", title=f"Click-Through Rate (CTR) by {eff_dim}", orientation="v")
                    fig.update_traces(hovertemplate="<b>%{x}</b><br>CTR: %{y:.3%}<extra></extra>")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # We can plot CPC if available
                plot_data = merged_table.dropna(subset=["CPC"]).copy()
                if len(plot_data) > 0 and plot_data["CPC"].sum() > 0:
                    fig = charts.bar_breakdown(plot_data, eff_dim, "CPC", title=f"Cost Per Click (CPC) by {eff_dim}", orientation="v")
                    # Update formatting for CPC
                    fig.update_traces(hovertemplate="<b>%{x}</b><br>CPC: $%{y:,.2f}<extra></extra>")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    plot_data_spend = merged_table.dropna(subset=["Spend"]).copy()
                    if len(plot_data_spend) > 0 and plot_data_spend["Spend"].sum() > 0:
                        fig = charts.bar_breakdown(plot_data_spend, eff_dim, "Spend", title=f"Spend by {eff_dim}", orientation="v")
                        st.plotly_chart(fig, use_container_width=True)
                        
        st.plotly_chart(charts.joint_spend_sales_bar(merged_table, eff_dim, title=f"Spend vs Est. Sales by {eff_dim}"), use_container_width=True)
                    
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 8: Forecasting (Deliverable 4) -- VISUAL REDESIGN
# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Sales Forecasting")
    st.caption("Forecasting operates on the full, company-wide weekly Sales series and is not affected by the sidebar filters.")

    leaderboard = model_result["leaderboard"]
    best_name = leaderboard.index[0]
    best_model = model_result["models"][best_name]

    X_test, y_test, test_df = model_result["test"]
    test_pred = forecasting.predict_in_dollar_space(best_model, X_test)
    eval_df = test_df[[config.DATE_COL, "Total Sales"]].copy()
    eval_df["Predicted Sales"] = test_pred

    mape = leaderboard.loc[best_name, "MAPE (%)"]
    rmse = leaderboard.loc[best_name, "RMSE"]
    mae = leaderboard.loc[best_name, "MAE"]
    r2 = leaderboard.loc[best_name, "R2"]

    # --- Section 1: Model Winner + Performance KPIs ---
    st.markdown('<div class="section-header"><h4>🏆 Best Model Performance</h4></div>', unsafe_allow_html=True)

    winner_col, kpi_cols = st.columns([2, 3])
    with winner_col:
        st.markdown(
            f'''<div class="model-winner">
                <div class="mw-icon">🏆</div>
                <div class="mw-text">
                    <h4>{best_name}</h4>
                    <p>Selected automatically — lowest holdout RMSE across {len(leaderboard)} models tested
                    on the last {config.TEST_SIZE_WEEKS} weeks.</p>
                </div>
            </div>''',
            unsafe_allow_html=True,
        )
    with kpi_cols:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(
            f'<div class="fc-metric-card"><div class="fc-label">MAPE</div>'
            f'<div class="fc-value">{mape:.1f}%</div>'
            f'<div class="fc-sub">Mean Abs % Error</div></div>',
            unsafe_allow_html=True,
        )
        k2.markdown(
            f'<div class="fc-metric-card"><div class="fc-label">RMSE</div>'
            f'<div class="fc-value">${rmse/1e6:.1f}M</div>'
            f'<div class="fc-sub">${rmse:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
        k3.markdown(
            f'<div class="fc-metric-card"><div class="fc-label">MAE</div>'
            f'<div class="fc-value">${mae/1e6:.1f}M</div>'
            f'<div class="fc-sub">${mae:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
        k4.markdown(
            f'<div class="fc-metric-card"><div class="fc-label">R²</div>'
            f'<div class="fc-value">{r2:.3f}</div>'
            f'<div class="fc-sub">Explained Variance</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # --- Section 2: Model Comparison ---
    st.markdown('<div class="section-header"><h4>📊 Model Comparison</h4></div>', unsafe_allow_html=True)

    compare_metric = st.selectbox(
        "Compare models by:",
        ["RMSE", "MAE", "MAPE (%)", "R2", "R2_holdout"],
        index=0,
        help="Switch between metrics to compare model performance from different angles.",
    )

    lb_left, lb_right = st.columns([3, 2])
    with lb_left:
        st.plotly_chart(
            charts.leaderboard_bar(leaderboard, compare_metric, f"{compare_metric} by Model"),
            use_container_width=True,
        )
    with lb_right:
        st.dataframe(
            leaderboard.style.format(
                {"RMSE": "${:,.0f}", "MAE": "${:,.0f}", "MAPE (%)": "{:.1f}%", "R2": "{:.3f}", "R2_holdout": "{:.3f}"}
            ).highlight_min(subset=["RMSE"], color="rgba(16,185,129,0.15)"),
            use_container_width=True,
        )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # --- Section 3: Holdout Performance ---
    st.markdown('<div class="section-header"><h4>🎯 Holdout Accuracy</h4></div>', unsafe_allow_html=True)

    st.plotly_chart(
        charts.actual_vs_predicted_chart(eval_df, "Actual vs. Predicted Sales (Holdout Set)"),
        use_container_width=True,
    )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # --- Section 4: Interactive Forecast ---
    st.markdown('<div class="section-header"><h4>🔮 Interactive Forecast</h4></div>', unsafe_allow_html=True)

    fc_ctrl_left, fc_ctrl_right = st.columns([1, 1])
    with fc_ctrl_left:
        horizon = st.slider(
            "Forecast horizon (weeks)",
            min_value=1, max_value=12, value=config.N_FORECAST_WEEKS,
            help="Drag to extend or shorten the forecast window.",
        )
    with fc_ctrl_right:
        history_depth = st.slider(
            "Historical context (weeks shown)",
            min_value=8, max_value=52, value=26, step=4,
            help="How many weeks of actual sales history to display alongside the forecast.",
        )

    forecast = forecasting.recursive_forecast(
        weekly_df, best_model, model_result["feature_cols"], model_result["ratio_caps"], n_weeks=horizon
    )
    forecast = forecasting.attach_forecast_interval(forecast, leaderboard.loc[best_name].to_dict())

    history = weekly_sales.dropna(subset=["Total Sales"]).tail(history_depth)
    st.plotly_chart(
        charts.forecast_chart(history, forecast, f"Actual vs. Forecasted Weekly Sales (next {horizon} weeks)"),
        use_container_width=True,
    )

    # --- Forecast Summary Card + Table side by side ---
    fc_sum_left, fc_sum_right = st.columns([1, 2])
    with fc_sum_left:
        avg_forecast = forecast["Forecasted Sales"].mean()
        total_forecast = forecast["Forecasted Sales"].sum()
        last_actual = history["Total Sales"].iloc[-1] if not history.empty else 0
        trend_pct = ((avg_forecast - last_actual) / last_actual * 100) if last_actual else 0
        trend_arrow = "↑" if trend_pct > 0 else "↓"
        trend_color = "#10B981" if trend_pct > 0 else "#EF4444"

        summary_html = f'''<div class="fc-summary-card">
            <h4>📋 Forecast Summary</h4>
            <div class="fc-row">
                <span class="fc-row-label">Horizon</span>
                <span class="fc-row-value">{horizon} weeks</span>
            </div>
            <div class="fc-row">
                <span class="fc-row-label">Avg Weekly Forecast</span>
                <span class="fc-row-value">${avg_forecast:,.0f}</span>
            </div>
            <div class="fc-row">
                <span class="fc-row-label">Total Forecast</span>
                <span class="fc-row-value">${total_forecast:,.0f}</span>
            </div>
            <div class="fc-row">
                <span class="fc-row-label">vs. Last Actual</span>
                <span class="fc-row-value" style="color:{trend_color}">{trend_arrow} {abs(trend_pct):.1f}%</span>
            </div>
            <div class="fc-row">
                <span class="fc-row-label">Model</span>
                <span class="fc-row-value" style="font-size:0.8rem">{best_name}</span>
            </div>
        </div>'''
        st.markdown(summary_html, unsafe_allow_html=True)

    with fc_sum_right:
        st.markdown("##### Forecast Detail")
        st.dataframe(
            forecast.style.format(
                {"Forecasted Sales": "${:,.0f}", "Lower Bound": "${:,.0f}", "Upper Bound": "${:,.0f}"}
            ),
            use_container_width=True,
        )
        st.download_button(
            "⬇ Download forecast as CSV",
            forecast.to_csv(index=False).encode("utf-8"),
            file_name="sales_forecast.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# TAB 9: Explainability (Deliverable 5)
# ---------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Explainability — What Drives Sales?")
    st.caption(
        "Three complementary views: standardized linear coefficients, tree-based feature "
        "importances, and SHAP values (computed on the XGBoost model as a faithful, fast-to-explain "
        "proxy — TreeExplainer doesn't support the stacking meta-estimator directly)."
    )

    feature_cols = model_result["feature_cols"]
    linear_model = model_result["models"]["Linear Regression (RidgeCV)"]
    xgb_model = model_result["models"]["XGBoost"]
    X_train, y_train, train_df = model_result["train"]

    top_n_features = st.slider(
        "Number of top features to display",
        min_value=5, max_value=min(30, len(feature_cols)), value=15, step=1,
        help="Adjust how many features are shown in the importance charts below.",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header"><h4>📐 Linear Model: Standardized Coefficients</h4></div>', unsafe_allow_html=True)
        coefs = explainability.linear_coefficients(linear_model, feature_cols)
        st.plotly_chart(
            charts.feature_importance_bar(coefs, "Standardized Coefficient", "Top Linear Drivers (log-Sales space)", top_n=top_n_features),
            use_container_width=True,
        )
    with c2:
        st.markdown('<div class="section-header"><h4>🌲 XGBoost: Feature Importances</h4></div>', unsafe_allow_html=True)
        importances = explainability.tree_feature_importance(xgb_model, feature_cols)
        st.plotly_chart(
            charts.feature_importance_bar(importances, "Importance", "Top XGBoost Drivers", top_n=top_n_features),
            use_container_width=True,
        )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h4>🧬 SHAP Global Feature Importance (XGBoost)</h4></div>', unsafe_allow_html=True)
    shap_values, shap_importance = get_shap()
    sc1, sc2 = st.columns([1, 1])
    with sc1:
        st.plotly_chart(
            charts.feature_importance_bar(shap_importance, "Mean |SHAP value|", "Mean |SHAP value| (log-Sales space)", top_n=top_n_features),
            use_container_width=True,
        )
    with sc2:
        try:
            fig = charts.shap_beeswarm(shap_values, feature_cols, max_display=top_n_features)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:  # pragma: no cover -- SHAP plotting is best-effort
            st.warning(f"SHAP beeswarm plot unavailable: {e}")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="insight-box">
        <b>What this is telling us:</b><br>
        • The autoregressive <b>Sales lag features dominate</b> every view — recent sales momentum is the
          single strongest predictor of next week's sales, more so than any individual media metric.<br>
        • <b>Does spend correlate with sales?</b> Only weakly in isolation (see the Correlation tab) —
          consistent with how an established CPG brand's week-to-week sales are driven more by
          underlying demand/distribution than by any one week's media flighting.<br>
        • <b>Which engagement metrics matter most?</b> Click-based and impression-based signals edge out
          pure engagement (likes/comments) metrics, but none individually dominates.
        </div>
        """,
        unsafe_allow_html=True,
    )

