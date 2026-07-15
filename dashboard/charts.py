"""
charts.py
=========
Reusable Plotly figure builders for the Streamlit dashboard. Keeping chart
construction here (rather than inline in app.py) means every tab gets a
consistent look, and any styling tweak only needs to happen in one place.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config

BRAND_SEQUENCE = [
    config.BRAND_PRIMARY,
    config.BRAND_SECONDARY,
    "#6CB4E4",
    "#648CAC",
    "#354551",
    "#9CA3AF",
]

SEVERITY_COLORS = {
    "High": "#EF4444",
    "Moderate": "#F59E0B",
    "Low": "#10B981",
}


def _style(fig: go.Figure, title: str | None = None, y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        title=dict(
            text=title,
            font=dict(size=16, color=config.BRAND_DARK, family="Inter, Segoe UI, sans-serif"),
        ) if title else None,
        margin=dict(l=10, r=10, t=60 if title else 10, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=config.BRAND_DARK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=BRAND_SEQUENCE,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB")
    if y_title:
        fig.update_yaxes(title=y_title)
    return fig


def line_trend(df, x, y, title=None, y_title=None) -> go.Figure:
    fig = px.line(df, x=x, y=y, markers=True)
    fig.update_traces(
        line=dict(color=config.BRAND_PRIMARY, width=2),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>%{y:$,.0f}<extra></extra>",
    )
    
    # Add peak annotation if data exists
    if not df.empty and y in df.columns:
        max_idx = df[y].idxmax()
        if not pd.isna(max_idx):
            max_row = df.loc[max_idx]
            max_val = max_row[y]
            max_date = max_row[x]
            
            # Format value depending on size
            if max_val > 1_000_000:
                val_text = f"${max_val/1_000_000:.1f}M"
            elif max_val > 1_000:
                val_text = f"${max_val/1_000:.1f}k"
            else:
                val_text = f"${max_val:,.0f}"
                
            fig.add_annotation(
                x=max_date, y=max_val,
                text=f"Peak: {val_text}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=config.BRAND_DARK,
                ax=0, ay=-40,
                font=dict(size=11, color="white"),
                bgcolor=config.BRAND_DARK,
                bordercolor=config.BRAND_DARK,
                borderwidth=1,
                borderpad=4,
                opacity=0.9
            )
            
    return _style(fig, title, y_title)


def bar_breakdown(df, x, y, title=None, orientation="v", y_title=None) -> go.Figure:
    if orientation == "h":
        fig = px.bar(df.sort_values(y), x=y, y=x, orientation="h", text_auto=".2s")
        fig.update_traces(
            marker_color=config.BRAND_PRIMARY,
            hovertemplate="<b>%{y}</b><br>%{x:$,.0f}<extra></extra>",
        )
    else:
        fig = px.bar(df, x=x, y=y, text_auto=".2s")
        fig.update_traces(
            marker_color=config.BRAND_PRIMARY,
            hovertemplate="<b>%{x}</b><br>%{y:$,.0f}<extra></extra>",
        )
    return _style(fig, title, y_title)


def correlation_heatmap(corr_df, title=None) -> go.Figure:
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale=["#FFFFFF", config.BRAND_SECONDARY, config.BRAND_PRIMARY],
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    return _style(fig, title)


def forecast_chart(history_df, forecast_df, title=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df[config.DATE_COL],
            y=history_df["Total Sales"],
            mode="lines+markers",
            line=dict(color=config.BRAND_DARK, width=1.5),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Actual: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df[config.DATE_COL],
            y=forecast_df["Forecasted Sales"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=config.BRAND_PRIMARY, width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>",
        )
    )
    if "Lower Bound" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=list(forecast_df[config.DATE_COL]) + list(forecast_df[config.DATE_COL])[::-1],
                y=list(forecast_df["Upper Bound"]) + list(forecast_df["Lower Bound"])[::-1],
                fill="toself",
                fillcolor="rgba(28,76,116,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Uncertainty Band (+/- holdout RMSE)",
                showlegend=True,
                hoverinfo="skip",
            )
        )
    return _style(fig, title, y_title="Sales ($)")


def leaderboard_bar(leaderboard_df, metric="RMSE", title=None) -> go.Figure:
    df = leaderboard_df.reset_index().sort_values(metric)
    fig = px.bar(df, x="Model", y=metric, text_auto=".2s", color="Model", color_discrete_sequence=BRAND_SEQUENCE)
    fig.update_layout(showlegend=False)
    fig.update_traces(hovertemplate="<b>%{x}</b><br>RMSE: $%{y:,.0f}<extra></extra>")
    return _style(fig, title, y_title=metric)


def feature_importance_bar(importance_df, value_col, title=None, top_n=15) -> go.Figure:
    df = importance_df.head(top_n).reset_index()
    df.columns = ["Feature", value_col]
    fig = px.bar(df.sort_values(value_col), x=value_col, y="Feature", orientation="h")
    fig.update_traces(marker_color=config.BRAND_SECONDARY)
    return _style(fig, title)


def shap_beeswarm(shap_values, feature_cols, max_display=15):
    """Returns a matplotlib figure (SHAP's own plotting is matplotlib-based)."""
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_cols, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def actual_vs_predicted_chart(df, title=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[config.DATE_COL],
            y=df["Total Sales"],
            mode="lines+markers",
            name="Actual Sales",
            line=dict(color=config.BRAND_DARK, width=1.5),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Actual: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[config.DATE_COL],
            y=df["Predicted Sales"],
            mode="lines+markers",
            name="Predicted Sales",
            line=dict(color=config.BRAND_PRIMARY, width=1.5, dash="dot"),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Predicted: $%{y:,.0f}<extra></extra>",
        )
    )
    return _style(fig, title, y_title="Sales ($)")


# -----------------------------------------------------------------------
# NEW: Data Quality & Methodology visual charts
# -----------------------------------------------------------------------

def missing_values_chart(missing_df, title="Missing Values by Column") -> go.Figure:
    """
    Horizontal bar chart of missing values per column, color-coded by severity.
    Expects a DataFrame with columns: Column, Missing Count, Missing %, Severity.
    """
    df = missing_df.sort_values("Missing Count", ascending=True).copy()

    colors = df["Severity"].map(SEVERITY_COLORS).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=df["Column"],
            x=df["Missing Count"],
            orientation="h",
            marker_color=colors,
            text=df.apply(lambda r: f'{r["Missing Count"]:,}  ({r["Missing %"]:.1f}%)', axis=1),
            textposition="outside",
            textfont=dict(size=12, color=config.BRAND_DARK),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Missing: %{x:,}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Add severity legend as annotations
    for sev, color in SEVERITY_COLORS.items():
        fig.add_trace(
            go.Bar(
                y=[None], x=[None],
                marker_color=color,
                name=f"{sev} Severity",
                showlegend=True,
            )
        )

    fig.update_layout(
        template=config.PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=16, color=config.BRAND_DARK)),
        margin=dict(l=10, r=120, t=60, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=config.BRAND_DARK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Missing Count",
        barmode="overlay",
        height=max(300, len(df) * 45 + 100),
    )
    return fig


def sales_duplication_scatter(evidence_df, title="Sales Column: Multiplier Pattern Evidence") -> go.Figure:
    """
    Scatter plot showing distinct Sales values within each week, with dot size
    proportional to the multiplier. The 1x base (true value) is highlighted in
    green, while inflated multiples are shown in red gradient.
    """
    if evidence_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No evidence data available", showarrow=False, font=dict(size=16))
        return _style(fig, title)

    df = evidence_df.copy()

    # Separate base values (multiplier == 1) from inflated values
    base = df[df["Multiplier"] == 1.0]
    inflated = df[df["Multiplier"] > 1.0]

    fig = go.Figure()

    # Base values (the "true" weekly sales)
    fig.add_trace(
        go.Scatter(
            x=base[config.DATE_COL],
            y=base["Sales Value"],
            mode="markers",
            name="True Base Value (1×)",
            marker=dict(
                color="#10B981",
                size=10,
                symbol="circle",
                line=dict(width=1, color="white"),
            ),
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Sales: $%{y:,.0f}<br>"
                "Multiplier: 1× (base)<br>"
                "<extra></extra>"
            ),
        )
    )

    # Inflated multiples
    if not inflated.empty:
        fig.add_trace(
            go.Scatter(
                x=inflated[config.DATE_COL],
                y=inflated["Sales Value"],
                mode="markers",
                name="Inflated Multiples (>1×)",
                marker=dict(
                    color=config.BRAND_PRIMARY,
                    size=inflated["Multiplier"].clip(upper=30).values * 3 + 4,
                    opacity=0.5,
                    symbol="circle",
                    line=dict(width=0.5, color="rgba(28,76,116,0.3)"),
                ),
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "Sales: $%{y:,.0f}<br>"
                    "Multiplier: %{customdata:.0f}×<br>"
                    "<extra></extra>"
                ),
                customdata=inflated["Multiplier"],
            )
        )

    fig.update_layout(
        yaxis_title="Sales Value ($)",
        yaxis_type="log",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _style(fig, title)


def sales_multiplier_distribution(evidence_df, title="Distribution of Sales Multipliers Across Weeks") -> go.Figure:
    """
    Histogram showing how many distinct multiplier values exist per week,
    illustrating the fan-out / duplication severity.
    """
    if evidence_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False, font=dict(size=16))
        return _style(fig, title)

    # Count of distinct values per week
    weekly_counts = evidence_df.groupby(config.DATE_COL)["Distinct Values This Week"].first().reset_index()

    fig = px.histogram(
        weekly_counts,
        x="Distinct Values This Week",
        nbins=max(int(weekly_counts["Distinct Values This Week"].max()), 5),
        color_discrete_sequence=[config.BRAND_SECONDARY],
    )
    fig.update_traces(
        marker_line_color=config.BRAND_DARK,
        marker_line_width=1,
        hovertemplate="Distinct values: %{x}<br>Weeks: %{y}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="# Distinct Sales Values in a Single Week",
        yaxis_title="Number of Weeks",
        bargap=0.1,
    )
    return _style(fig, title)


def cleaning_impact_waterfall(quality_stats: dict, title="Data Cleaning Pipeline Impact") -> go.Figure:
    """
    Waterfall chart showing the impact of each cleaning step.
    """
    steps = []
    values = []
    measures = []
    colors = []

    steps.append("Raw Data Issues")
    values.append(quality_stats["total_missing_cells"])
    measures.append("absolute")
    colors.append("#E8E8E8")

    # Each cleaning step reduces issues
    if quality_stats["structural_nas_filled"] > 0:
        steps.append(f"Structural NAs Filled")
        values.append(-quality_stats["structural_nas_filled"])
        measures.append("relative")
        colors.append("#349CE4")

    if quality_stats["typos_fixed"] > 0:
        steps.append(f"Typos Corrected")
        values.append(-quality_stats["typos_fixed"])
        measures.append("relative")
        colors.append("#6CB4E4")

    if quality_stats["duplicates_dropped"] > 0:
        steps.append(f"Duplicates Removed")
        values.append(-quality_stats["duplicates_dropped"])
        measures.append("relative")
        colors.append("#648CAC")

    if quality_stats["negative_spend_rows"] > 0:
        steps.append(f"Negative Spend Flagged")
        values.append(-quality_stats["negative_spend_rows"])
        measures.append("relative")
        colors.append(config.BRAND_SECONDARY)

    # Remaining (metric NAs left as zero-fill)
    remaining = (
        quality_stats["total_missing_cells"]
        - quality_stats["structural_nas_filled"]
        - quality_stats["typos_fixed"]
        - quality_stats["duplicates_dropped"]
        - quality_stats["negative_spend_rows"]
    )
    if remaining > 0:
        steps.append("Metric NAs (zero-filled)")
        values.append(-remaining)
        measures.append("relative")
        colors.append("#A3A3A3")

    steps.append("Remaining Issues")
    values.append(0)
    measures.append("total")
    colors.append(config.BRAND_PRIMARY)

    fig = go.Figure(
        go.Waterfall(
            name="Cleaning Impact",
            orientation="v",
            measure=measures,
            x=steps,
            y=values,
            connector=dict(line=dict(color="rgba(0,0,0,0.15)", width=1)),
            increasing=dict(marker=dict(color="#E8E8E8")),
            decreasing=dict(marker=dict(color="#10B981")),
            totals=dict(marker=dict(color=config.BRAND_PRIMARY)),
            textposition="outside",
            text=[f"{abs(v):,}" for v in values],
            textfont=dict(size=12),
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        showlegend=False,
        yaxis_title="Cell Count",
        height=420,
    )
    return _style(fig, title)


def data_pipeline_sankey(quality_stats: dict, title="Data Processing Pipeline Flow") -> go.Figure:
    """
    Sankey diagram showing the data flow from raw to clean, with node labels
    showing row/column counts at each stage.
    """
    labels = [
        f"Raw Data\n{quality_stats['total_rows']:,} rows × {quality_stats['total_columns']} cols",
        f"Missing Cells\n{quality_stats['total_missing_cells']:,}",
        f"Text Cleaning\n{quality_stats['typos_fixed']} typos fixed",
        f"Structural NA Fill\n{quality_stats['structural_nas_filled']:,} filled",
        f"Duplicate Check\n{quality_stats['duplicates_dropped']} removed",
        f"Spend Flagging\n{quality_stats['negative_spend_rows']} flagged",
        f"Clean Data\n{quality_stats['clean_rows']:,} rows × {quality_stats['clean_columns']} cols",
    ]

    source = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    target = [1, 2, 3, 4, 5, 6, 6, 6, 6, 6]
    value = [
        quality_stats["total_missing_cells"],
        max(quality_stats["typos_fixed"], 1),
        max(quality_stats["structural_nas_filled"], 1),
        max(quality_stats["duplicates_dropped"], 1),
        max(quality_stats["negative_spend_rows"], 1),
        max(quality_stats["total_missing_cells"] - quality_stats["structural_nas_filled"], 1),
        max(quality_stats["typos_fixed"], 1),
        max(quality_stats["structural_nas_filled"], 1),
        max(quality_stats["duplicates_dropped"], 1),
        max(quality_stats["negative_spend_rows"], 1),
    ]

    node_colors = [
        "#E5E7EB",       # Raw Data
        config.BRAND_PRIMARY,  # Missing Cells
        "#6CB4E4",       # Text Cleaning
        config.BRAND_SECONDARY,  # Structural NA Fill
        "#648CAC",       # Duplicate Check
        "#354551",       # Spend Flagging
        config.BRAND_PRIMARY,  # Clean Data
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=25,
                thickness=25,
                line=dict(color="rgba(0,0,0,0.1)", width=1),
                label=labels,
                color=node_colors,
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=[
                    "rgba(28,76,116,0.15)",
                    "rgba(108,180,228,0.15)",
                    "rgba(108,180,228,0.15)",
                    "rgba(100,140,172,0.15)",
                    "rgba(53,69,81,0.15)",
                    "rgba(28,76,116,0.1)",
                    "rgba(108,180,228,0.1)",
                    "rgba(108,180,228,0.1)",
                    "rgba(100,140,172,0.1)",
                    "rgba(53,69,81,0.1)",
                ],
            ),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=config.BRAND_DARK)),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=config.BRAND_DARK),
        margin=dict(l=10, r=10, t=60, b=10),
        height=400,
    )
    return fig


def reconciliation_gauge(max_diff: float, title="Allocation Reconciliation") -> go.Figure:
    """
    A gauge/indicator chart showing that the Sales allocation reconciles to
    the true weekly total (max difference should be ~0).
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=max_diff,
            number=dict(prefix="$", valueformat=",.6f", font=dict(size=28)),
            delta=dict(reference=0.01, decreasing=dict(color="#10B981")),
            title=dict(text="Max Reconciliation Diff", font=dict(size=14)),
            gauge=dict(
                axis=dict(range=[0, max(0.01, max_diff * 10)], tickformat="$.6f"),
                bar=dict(color="#10B981"),
                bgcolor="rgba(0,0,0,0.03)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 0.001], color="rgba(16,185,129,0.2)"),
                    dict(range=[0.001, 0.01], color="rgba(245,158,11,0.2)"),
                ],
                threshold=dict(
                    line=dict(color=config.BRAND_PRIMARY, width=3),
                    thickness=0.8,
                    value=0.01,
                ),
            ),
        )
    )
    fig.update_layout(
        margin=dict(l=30, r=30, t=40, b=10),
        height=250,
        font=dict(family="Inter, Segoe UI, sans-serif", color=config.BRAND_DARK),
    )
    return fig


def spend_and_sales_over_time(weekly_df: pd.DataFrame, title: str = "Spend and Sales Over Time") -> go.Figure:
    """
    Dual-axis line chart of weekly Spend and Total Sales over time,
    showing how marketing spend spikes relate to sales performance chronologically.
    """
    df = weekly_df.dropna(subset=["Spend", "Total Sales"]).copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=16))
        return _style(fig, title)

    # Sort by date
    df = df.sort_values(config.DATE_COL)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Spend Trace
    fig.add_trace(
        go.Scatter(
            x=df[config.DATE_COL],
            y=df["Spend"],
            mode="lines+markers",
            name="Spend",
            line=dict(color=config.BRAND_SECONDARY, width=1.5),
            marker=dict(size=6),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Spend: $%{y:,.0f}<extra></extra>"
        ),
        secondary_y=False,
    )

    # Sales Trace
    fig.add_trace(
        go.Scatter(
            x=df[config.DATE_COL],
            y=df["Total Sales"],
            mode="lines+markers",
            name="Total Sales",
            line=dict(color=config.BRAND_PRIMARY, width=2),
            marker=dict(size=8),
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sales: $%{y:,.0f}<extra></extra>"
        ),
        secondary_y=True,
    )

    # Compute and annotate correlation
    corr = df["Spend"].corr(df["Total Sales"])
    fig.add_annotation(
        text=f"r = {corr:.3f}",
        xref="paper", yref="paper",
        x=0.01, y=0.98,
        showarrow=False,
        font=dict(size=14, color=config.BRAND_DARK, family="Inter, Segoe UI, sans-serif"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=config.BRAND_DARK,
        borderwidth=1,
        borderpad=6,
    )

    fig = _style(fig, title)
    fig.update_yaxes(title_text="Spend ($)", secondary_y=False)
    fig.update_yaxes(title_text="Total Sales ($)", secondary_y=True, showgrid=False)

    return fig


def spend_vs_sales_scatter(weekly_df: pd.DataFrame, title: str = "Spend vs Sales") -> go.Figure:
    """
    Scatter plot of weekly Spend vs Total Sales.
    """
    df = weekly_df.dropna(subset=["Spend", "Total Sales"]).copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=16))
        return _style(fig, title)
        
    fig = px.scatter(
        df,
        x="Spend",
        y="Total Sales",
        hover_data=[config.DATE_COL]
    )
    fig.update_traces(
        marker=dict(size=8, color=config.BRAND_PRIMARY),
        hovertemplate="<b>%{customdata[0]|%b %d, %Y}</b><br>Spend: $%{x:,.0f}<br>Sales: $%{y:,.0f}<extra></extra>"
    )
    
    corr = df["Spend"].corr(df["Total Sales"])
    fig.add_annotation(
        text=f"r = {corr:.3f}",
        xref="paper", yref="paper",
        x=0.01, y=0.98,
        showarrow=False,
        font=dict(size=14, color=config.BRAND_DARK, family="Inter, Segoe UI, sans-serif"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=config.BRAND_DARK,
        borderwidth=1,
        borderpad=6,
    )
    
    return _style(fig, title, y_title="Total Sales ($)")


def joint_spend_sales_bar(df: pd.DataFrame, dimension: str, title: str = None) -> go.Figure:
    """
    Grouped bar chart showing Spend and Estimated Sales Contribution for a dimension.
    """
    fig = go.Figure()
    
    if df.empty:
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=16))
        return _style(fig, title)
        
    fig.add_trace(go.Bar(
        x=df[dimension],
        y=df["Spend"],
        name="Spend",
        marker_color=config.BRAND_SECONDARY,
        hovertemplate="<b>%{x}</b><br>Spend: $%{y:,.0f}<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        x=df[dimension],
        y=df["Estimated Sales Contribution"],
        name="Est. Sales",
        marker_color=config.BRAND_PRIMARY,
        hovertemplate="<b>%{x}</b><br>Sales: $%{y:,.0f}<extra></extra>"
    ))
    
    fig.update_layout(barmode="group")
    return _style(fig, title, y_title="Amount ($)")

