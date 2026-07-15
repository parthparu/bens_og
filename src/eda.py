"""
eda.py
======
Reusable exploratory-analysis aggregations shared by the Streamlit dashboard
and (optionally) any ad-hoc analysis script. Keeping these as small, pure
functions means the same exact numbers are guaranteed to show up in both the
dashboard and any report we generate from `main.py` -- no risk of the two
silently drifting apart.

A note on "average" efficiency ratios (CTR, Engagement Rate, etc.) by
category: we deliberately do NOT average the row-level ratio column when
rolling up to e.g. "CTR by Media Type". Averaging a ratio is a classic
Simpson's-paradox trap -- a category with one huge-Impressions row and a
low CTR can still produce a high "average CTR" if it also has many small
rows with artificially high CTR. The correct rollup is to re-derive the
ratio from the SUMS: sum(Clicks)/sum(Impressions), not mean(CTR). Every
function below follows that rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def kpi_summary(row_df: pd.DataFrame, weekly_sales: pd.DataFrame) -> dict:
    """
    Executive Summary KPIs (Deliverable 3). Total Sales uses the
    de-duplicated weekly figure (summed across known weeks), NOT a sum of
    the raw row-level Sales column -- see data_prep.py for why.
    """
    return {
        "Total Sales": float(weekly_sales["Total Sales"].sum(skipna=True)),
        "Total Spend": float(row_df["Spend"].sum()),
        "Total Impressions": float(row_df["Impressions"].sum()),
        "Total Clicks": float(row_df["Clicks"].sum()),
        "Total Engagements": float(row_df["Engagements"].sum()),
    }


def breakdown_by(
    row_df: pd.DataFrame,
    dimension: str,
    metric: str = "Estimated Sales Contribution",
    top_n: int | None = None,
) -> pd.DataFrame:
    """Sum `metric` grouped by `dimension`, sorted descending."""
    out = (
        row_df.groupby(dimension, dropna=False)[metric]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    if top_n:
        out = out.head(top_n)
    return out


def top_creatives(row_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Top N creatives ranked by Estimated Sales Contribution (Deliverable 3).
    Also reports Spend and Impressions alongside so the ranking can be
    sanity-checked against raw activity.
    """
    agg = (
        row_df.groupby("Creative", dropna=False)
        .agg(
            **{
                "Estimated Sales Contribution": ("Estimated Sales Contribution", "sum"),
                "Spend": ("Spend", "sum"),
                "Impressions": ("Impressions", "sum"),
                "Engagements": ("Engagements", "sum"),
            }
        )
        .reset_index()
        .sort_values("Estimated Sales Contribution", ascending=False)
        .head(n)
    )
    return agg


def weekly_trend(weekly_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """A simple Week, metric two-column frame for line charts."""
    return weekly_df[[config.DATE_COL, metric]].dropna().sort_values(config.DATE_COL)


def correlation_matrix(weekly_df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    Correlation analysis (Deliverable 3) computed at the WEEKLY-AGGREGATE
    grain (Spend/Impressions/Clicks/Engagements summed per week against the
    de-duplicated weekly Sales). Row-level correlation would mix a
    week-level Sales figure against media-mix-level dimensions inconsistently,
    so we deliberately use the weekly feature table here, not the row-level
    dataframe.
    """
    cols = cols or ["Spend", "Impressions", "Clicks", "Engagements", "Total Sales"]
    cols = [c for c in cols if c in weekly_df.columns]
    return weekly_df[cols].corr()


def efficiency_by_dimension(row_df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """
    Media Efficiency Analysis (Deliverable 3): CTR, CPC, CPM, and Cost Per
    Engagement re-derived from summed numerators/denominators per category
    (see module docstring on why we don't average the ratio column directly).
    """
    grouped = row_df.groupby(dimension, dropna=False).agg(
        Spend=("Spend", "sum"),
        Impressions=("Impressions", "sum"),
        Clicks=("Clicks", "sum"),
        Engagements=("Engagements", "sum"),
    )
    grouped["CTR"] = grouped["Clicks"] / grouped["Impressions"].replace(0, np.nan)
    grouped["CPC"] = grouped["Spend"] / grouped["Clicks"].replace(0, np.nan)
    grouped["CPM"] = grouped["Spend"] / grouped["Impressions"].replace(0, np.nan) * 1000
    grouped["Cost Per Engagement"] = grouped["Spend"] / grouped["Engagements"].replace(0, np.nan)
    return grouped.reset_index().sort_values("Spend", ascending=False)


def missingness_report(raw_df: pd.DataFrame) -> pd.DataFrame:
    """A quick missing-value summary, useful for the Data Preparation writeup."""
    miss = raw_df.isna().sum()
    pct = (miss / len(raw_df) * 100).round(2)
    out = pd.DataFrame({"Missing Count": miss, "Missing %": pct})
    return out[out["Missing Count"] > 0].sort_values("Missing Count", ascending=False)


def missingness_report_detailed(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced missing-value summary with action taken and severity classification.
    Returns a DataFrame with columns: Column, Missing Count, Missing %, Severity,
    Action Taken, Category.
    """
    miss = raw_df.isna().sum()
    pct = (miss / len(raw_df) * 100).round(2)
    out = pd.DataFrame({"Column": miss.index, "Missing Count": miss.values, "Missing %": pct.values})
    out = out[out["Missing Count"] > 0].sort_values("Missing Count", ascending=False).reset_index(drop=True)

    # Classify severity
    def _severity(p):
        if p > 50:
            return "High"
        elif p > 10:
            return "Moderate"
        else:
            return "Low"

    out["Severity"] = out["Missing %"].apply(_severity)

    # Map actions taken per column (from config.STRUCTURAL_NA_FILL and data_prep logic)
    action_map = {
        "Site": "Filled → 'Unknown' (structural)",
        "Device": "Filled → 'Not Tracked' (structural)",
        "Platform Type": "Filled → 'Unknown' (structural)",
        "Partner Type": "Filled → 'Not Applicable' (structural)",
        "Audience": "Filled → 'Not Applicable' (structural)",
        "Creative": "Filled → 'Unspecified Creative' (structural)",
    }
    out["Action Taken"] = out["Column"].map(action_map).fillna("Left as NaN (metric — zero-fill)")
    out["Category"] = out["Column"].apply(
        lambda c: "Structural" if c in config.STRUCTURAL_NA_FILL else "Metric"
    )
    return out


def data_quality_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """
    Returns a dict with key cleaning statistics for visual KPI display:
    total raw rows, columns, total missing cells, duplicates dropped,
    typos fixed, NAs filled, negative spend rows, etc.
    """
    total_cells = raw_df.shape[0] * raw_df.shape[1]
    total_missing = int(raw_df.isna().sum().sum())
    missing_pct = round(total_missing / total_cells * 100, 2) if total_cells else 0

    # Typo fix count: count rows where Sub-Channel was 'VIdeo' in raw
    typo_count = 0
    if "Sub-Channel" in raw_df.columns:
        typo_count = int((raw_df["Sub-Channel"].astype(str).str.strip() == "VIdeo").sum())

    # Structural NA fills
    na_fills = 0
    for col, fill_val in config.STRUCTURAL_NA_FILL.items():
        if col in raw_df.columns:
            na_fills += int(raw_df[col].isna().sum())

    # Negative spend rows
    neg_spend = 0
    if "Spend" in raw_df.columns:
        neg_spend = int((pd.to_numeric(raw_df["Spend"], errors="coerce") < 0).sum())

    # Duplicates
    dupes = raw_df.shape[0] - raw_df.drop_duplicates().shape[0]

    # Distinct weeks
    distinct_weeks = 0
    if config.DATE_COL in raw_df.columns:
        distinct_weeks = int(raw_df[config.DATE_COL].nunique())

    return {
        "total_rows": raw_df.shape[0],
        "total_columns": raw_df.shape[1],
        "total_missing_cells": total_missing,
        "missing_pct": missing_pct,
        "duplicates_dropped": dupes,
        "typos_fixed": typo_count,
        "structural_nas_filled": na_fills,
        "negative_spend_rows": neg_spend,
        "distinct_weeks": distinct_weeks,
        "clean_rows": clean_df.shape[0],
        "clean_columns": clean_df.shape[1],
    }


def sales_duplication_evidence(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame showing the Sales multiplier pattern within each week.
    For each week, computes the min positive Sales value and the ratio of each
    distinct Sales value to that minimum. Used for the visual evidence chart.
    """
    df = raw_df.copy()
    if config.DATE_COL in df.columns:
        df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    if config.TARGET_COL not in df.columns:
        return pd.DataFrame()

    df["_sales_numeric"] = pd.to_numeric(df[config.TARGET_COL], errors="coerce")
    pos = df[df["_sales_numeric"] > 0].copy()
    if pos.empty:
        return pd.DataFrame()

    # Get the min positive Sales per week
    week_min = pos.groupby(config.DATE_COL)["_sales_numeric"].min().rename("Min Sales")

    # Get unique Sales values per week
    unique_per_week = (
        pos.groupby(config.DATE_COL)["_sales_numeric"]
        .apply(lambda s: sorted(s.unique()))
        .reset_index()
        .rename(columns={"_sales_numeric": "Unique Sales Values"})
    )
    unique_per_week = unique_per_week.merge(week_min, on=config.DATE_COL)

    # Explode unique values and compute multipliers
    rows = []
    for _, row in unique_per_week.iterrows():
        week = row[config.DATE_COL]
        min_val = row["Min Sales"]
        for val in row["Unique Sales Values"]:
            multiplier = round(val / min_val, 1) if min_val > 0 else 0
            rows.append({
                config.DATE_COL: week,
                "Sales Value": val,
                "Min Sales (True Base)": min_val,
                "Multiplier": multiplier,
                "Distinct Values This Week": len(row["Unique Sales Values"]),
            })

    return pd.DataFrame(rows)
