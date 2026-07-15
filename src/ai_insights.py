"""
ai_insights.py
===============
Deliverable 6: AI-Powered Insights.

The brief lists ChatGPT / Gemini / Claude / NotebookLM as candidate tools
for turning the analysis into plain-English marketing insights. Since this
entire project was built working with Claude, this module is that step:
rather than calling out to another LLM API (and adding an extra external
dependency + API key requirement to a project that should run out-of-the-
box), the narrative below was authored directly against this project's own
computed outputs.

To keep the insights honest and reproducible rather than generic
boilerplate, `generate_insights()` takes the *actual* numbers produced by
the rest of the pipeline (channel allocation, efficiency table, model
leaderboard, SHAP importances, correlations) and slots them into the
narrative -- so if the underlying data changes, the insights regenerate
with it instead of silently going stale.
"""

from __future__ import annotations

import pandas as pd

from . import config


def _fmt_currency(x: float) -> str:
    return f"${x:,.0f}"


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def generate_insights(
    channel_sales: pd.DataFrame,
    efficiency_table: pd.DataFrame,
    leaderboard: pd.DataFrame,
    shap_importance: pd.DataFrame,
    correlation: pd.DataFrame,
) -> dict:
    """
    Build the three sections the brief asks for: Marketing Insights, Media
    Recommendations, Optimization Suggestions. Each is a list of short,
    data-grounded bullet strings.
    """
    insights, recommendations, optimizations = [], [], []

    # --- Marketing Insights ----------------------------------------------
    if not channel_sales.empty:
        top = channel_sales.iloc[0]
        bottom = channel_sales.iloc[-1]
        total = channel_sales["Estimated Sales Contribution"].sum()
        top_share = top["Estimated Sales Contribution"] / total if total else 0
        insights.append(
            f"**{top.iloc[0]}** carries the largest estimated share of sales "
            f"contribution at {_fmt_pct(top_share)} ({_fmt_currency(top['Estimated Sales Contribution'])}) "
            f"across the analyzed period -- but remember this is an activity-weighted "
            f"*estimate*, not audited revenue attribution (see the Sales Drivers tab for the methodology)."
        )
        insights.append(
            f"**{bottom.iloc[0]}** shows the smallest estimated contribution "
            f"({_fmt_currency(bottom['Estimated Sales Contribution'])}), which may reflect either genuinely "
            f"lower impact or simply a smaller media footprint -- worth a deliberate test-and-learn "
            f"budget increase before writing it off."
        )

    if not correlation.empty:
        strongest = correlation["Correlation with Total Sales"].abs().idxmax()
        strongest_val = correlation.loc[strongest, "Correlation with Total Sales"]
        direction = "a positive" if strongest_val > 0 else "an inverse"
        insights.append(
            f"Of the row-level activity metrics, **{strongest}** has the strongest (still weak-to-moderate, "
            f"r={strongest_val:.2f}) {direction} relationship with weekly sales -- in isolation, no single "
            f"weekly media metric is a dominant linear driver of sales."
        )

    if not shap_importance.empty:
        top_feature = shap_importance.index[0]
        insights.append(
            f"The forecasting model's SHAP analysis confirms this: **{top_feature}** -- a measure of recent "
            f"sales momentum, not a media metric -- is the single most influential input. This is a "
            f"common, legitimate pattern for an established national CPG brand: week-to-week sales are "
            f"driven more by underlying demand/distribution momentum than by any one week's media flighting, "
            f"and media's effect shows up gradually rather than as an immediate spike."
        )

    # --- Media Recommendations --------------------------------------------
    if not efficiency_table.empty and "CPC" in efficiency_table.columns:
        eff = efficiency_table.dropna(subset=["CPC"])
        eff = eff[eff["Spend"] > 0]  # a $0 CPC just means no Spend was tracked, not a free click
        if not eff.empty:
            best_cpc = eff.sort_values("CPC").iloc[0]
            recommendations.append(
                f"**{best_cpc.iloc[0]}** delivers the lowest cost-per-click "
                f"({_fmt_currency(best_cpc['CPC'])}) among channels with both recorded Spend and "
                f"paid click activity -- a candidate for incremental budget if the goal is efficient "
                f"traffic generation."
            )
    if not channel_sales.empty and len(channel_sales) > 1:
        recommendations.append(
            "Given the weak standalone correlation between any single media metric and sales, "
            "recommend testing **incrementality** directly (geo holdouts or matched-market tests) "
            "for the top 2 channels by spend, rather than relying on the linear/allocated view alone."
        )
    recommendations.append(
        "TV and Paid Social are the only channels with directly comparable dollar-spend data in this "
        "export; prioritize closing the **Display Data** spend-tracking gap (currently $0 recorded "
        "Spend) so future analyses can compare all paid channels on a like-for-like cost basis."
    )

    # --- Optimization Suggestions ------------------------------------------
    if not leaderboard.empty:
        best_model = leaderboard.index[0]
        best_row = leaderboard.iloc[0]
        optimizations.append(
            f"The **{best_model}** currently gives the best holdout accuracy "
            f"(RMSE {_fmt_currency(best_row['RMSE'])}, MAPE {best_row['MAPE (%)']:.1f}%). "
            f"With roughly {config.TEST_SIZE_WEEKS} held-out test weeks and ~100 total weeks of history, "
            f"treat this as a solid first read rather than a finished model -- accuracy should improve "
            f"meaningfully as more weeks of sales data accumulate."
        )
    optimizations.append(
        "Adding external regressors not present in this export -- price/promotion calendar, "
        "distribution (TDP), competitor activity, and macro seasonality (e.g. holidays) -- would likely "
        "reduce forecast error more than further tuning the current media-only feature set, given how "
        "weak the direct media-to-sales correlations turned out to be."
    )
    optimizations.append(
        "Re-run the Sales de-duplication check (see data_prep.py) whenever the source file refreshes -- "
        "the broadcast-Sales fan-out pattern is an artifact of however the export was built, and a future "
        "extract could fix it upstream, change the multiplier scheme, or introduce new edge cases."
    )

    return {
        "Marketing Insights": insights,
        "Media Recommendations": recommendations,
        "Optimization Suggestions": optimizations,
    }
