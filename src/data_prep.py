"""
data_prep.py
============
Data loading, cleaning, and feature engineering for the Ben's Original
marketing dataset.

THE "SALES" COLUMN PROBLEM (read this before touching anything else)
----------------------------------------------------------------------
This is the single most important data-quality issue in the whole dataset,
and it directly drives several design decisions downstream (the KPIs, the
"Sales by X" charts, and the forecasting target). It's worth writing out in
full because it is *not* a simple missing-value or dtype problem -- it's a
many-to-one join / fan-out artifact, and naively `sum()`-ing the raw column
would silently produce numbers that are 10x-1000x too large.

What we observed during EDA:
  * The raw file has 12,529 rows but only 122 distinct Weeks.
  * Within a single Week (and even within a single TAB / data source for
    that week), the `Sales` column takes on *multiple distinct values* --
    anywhere from 1 to 17 unique values per week.
  * Those distinct values are (almost always) clean integer multiples of
    one another -- e.g. within one Pinterest/week slice we found values of
    9,772,220.36, 19,544,440.70 (2x), 58,633,322.20 (6x), 87,949,983 (9x),
    175,900,000 (18x), 234,533,300 (24x) ... all exact multiples of the
    same base number.
  * The *minimum* non-zero Sales value observed in a given week is
    extremely stable and economically sensible (consistently ~$7M-$10M/week
    across 2023-2024, tracking a smooth seasonal curve), while the *larger*
    multiples within that same week jump around unpredictably (sometimes
    into the billions) purely as a function of how many Creative/Audience/
    Site rows happen to share that week.

Conclusion: `Sales` was originally a single **weekly, brand-level** KPI
(one true number per week) that got broadcast/copied onto every granular
media row for that week somewhere upstream of this export -- almost
certainly because the original source had a multi-valued field (e.g. an
Audience or Creative tag that legitimately listed several values in one
cell) that was "exploded" into one row per value, carrying the un-split
Sales figure along with it into every resulting row. The result is that
summing `Sales` across rows over-counts the true number by a factor equal
to how many rows happen to share that broadcast value -- and that factor
is different for every Channel/Site/Creative slice, so naive aggregation
doesn't just inflate the totals, it *distorts the relative ranking* between
channels too (whichever channel happens to have the most granular rows
"wins" regardless of real performance).

Our fix, in two parts:

  1. TRUE WEEKLY SALES (the actual KPI / forecasting target)
     -> `compute_weekly_sales()`
     We take the **minimum non-zero Sales value observed across the whole
     dataset in a given week** as that week's real, de-duplicated sales
     figure. This is the number used for the Executive Summary KPI, the
     weekly sales trend chart, Cost-Per-Sale, and the forecasting target.

  2. ESTIMATED SALES CONTRIBUTION (for "Sales by Channel/Site/Device" bars)
     -> `allocate_estimated_sales()`
     The brief explicitly asks for "Sales by Media Type / Channel / Site /
     Device / Platform Type" bar charts, but there is no genuine row-level
     sales attribution in the source data (only one true number exists per
     week). Rather than display a meaningless sum of the broadcast column,
     we allocate each week's TRUE total sales down to individual rows in
     proportion to each row's share of that week's media activity (Spend
     where available -- dollars are directly comparable across every paid
     channel -- falling back to a normalized Impressions / Reach / GRPs /
     Engagements / Likes intensity score for organic/earned/owned rows that
     report no spend at all -- see config.ALLOCATION_WEIGHT_PRIORITY). The
     resulting
     "Estimated Sales Contribution" column is clearly labelled as an
     estimate everywhere it's surfaced in the dashboard, and by
     construction it always sums back to the true weekly total -- so it's
     directionally useful for comparing channels without re-introducing
     the inflation problem.

This reasoning is also reproduced in the README and on the dashboard itself
(an info banner on the "Sales Drivers" tab) so nobody mistakes the
allocated figure for ground-truth, audited revenue attribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def load_raw_data(path=None) -> pd.DataFrame:
    """Load the raw Excel export exactly as provided."""
    path = path or config.RAW_DATA_PATH
    df = pd.read_excel(path, sheet_name=config.RAW_SHEET_NAME)
    return df


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all row-level cleaning steps:
      - strip whitespace from text columns
      - fix the known "VIdeo" -> "Video" typo in Sub-Channel
      - fill *structurally* missing dimension values with an explicit label
        (these are NOT random missing values -- e.g. "Device" is simply
        never tracked for TV or social-listening rows, so imputing it would
        be misleading; an explicit "Not Tracked"/"Not Applicable" label is
        more honest than a mean/mode fill)
      - flag (but keep) the 2 rows with negative Spend, which read like
        legitimate credit/rebate adjustments on TV buys rather than data
        errors
      - drop exact full-row duplicates (none were found in this file, but
        the check is kept so the pipeline stays correct if the data refreshes)
      - enforce sane dtypes
    """
    df = df.copy()

    # Strip whitespace on all string/object columns
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()

    # Ensure all dates align to Sunday (fixes typos like 2024-11-21 Thursday)
    if config.DATE_COL in df.columns:
        df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], format="mixed")
        df[config.DATE_COL] = df[config.DATE_COL] + pd.to_timedelta((6 - df[config.DATE_COL].dt.dayofweek) % 7, unit="d")

    # Fix known categorical typo (only when Sub-Channel has string values)
    if "Sub-Channel" in df.columns and df["Sub-Channel"].dtype == "object" or df["Sub-Channel"].dtype == "string":
        df["Sub-Channel"] = df["Sub-Channel"].replace(config.SUB_CHANNEL_TYPO_FIX)

    # Structural missing-value fills (see config.STRUCTURAL_NA_FILL)
    for col, fill_value in config.STRUCTURAL_NA_FILL.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    # Flag negative-spend rows instead of silently clipping/deleting them
    df["Spend Is Credit Adjustment"] = df["Spend"] < 0

    # Drop exact duplicate rows, if any
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        df.attrs["duplicates_dropped"] = dropped

    # Dtype corrections
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], format="mixed")
    numeric_cols = [c for c in config.METRIC_COLS if c in df.columns]
    for c in numeric_cols:
        # Clip to 0 to prevent negative values (like refunds/credits or tracking errors)
        # from appearing in the dashboard and skewing analysis.
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    return df


# ---------------------------------------------------------------------------
# 3. Date engineering
# ---------------------------------------------------------------------------
def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Year, Quarter, Month, and ISO Week Number columns from `Week`."""
    df = df.copy()
    wk = df[config.DATE_COL]
    df["Year"] = wk.dt.year
    df["Quarter"] = wk.dt.quarter
    df["Month"] = wk.dt.month
    df["Month Name"] = wk.dt.strftime("%b")
    df["Week Number"] = wk.dt.isocalendar().week.astype(int)
    return df


# ---------------------------------------------------------------------------
# 4. The Sales de-duplication logic (see module docstring)
# ---------------------------------------------------------------------------
def compute_weekly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean, one-row-per-week DataFrame with the de-duplicated TRUE
    Sales figure for that week (see module docstring for the full
    reasoning). Weeks with no positive Sales rows at all (e.g. the most
    recent weeks, where sales reporting lags media activity) come back as
    NaN rather than 0, so they aren't mistaken for a genuine zero-sales week.
    """
    pos = df.loc[df[config.TARGET_COL] > 0, [config.DATE_COL, config.TARGET_COL]]
    weekly = (
        pos.groupby(config.DATE_COL)[config.TARGET_COL]
        .min()
        .rename("Total Sales")
        .reset_index()
    )

    all_weeks = pd.DataFrame({config.DATE_COL: sorted(df[config.DATE_COL].unique())})
    weekly = all_weeks.merge(weekly, on=config.DATE_COL, how="left")
    weekly = weekly.sort_values(config.DATE_COL).reset_index(drop=True)
    return weekly


def allocate_estimated_sales(df: pd.DataFrame, weekly_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Add an "Estimated Sales Contribution" column to the row-level dataframe
    by allocating each week's TRUE total sales proportionally to each row's
    share of that week's media "activity" (see module docstring). This is
    what powers the "Sales by Media Type / Channel / Site / Device /
    Platform Type" charts -- it is an *estimate*, not ground-truth
    attribution, and is labelled as such throughout the dashboard.
    """
    df = df.copy()

    # Build a single "activity weight" per row by falling down the metric
    # priority list until we find one that's actually non-zero for that row.
    #
    # IMPORTANT: different data sources populate metrics on wildly
    # different scales -- Impressions run into the hundreds of thousands /
    # millions, while TV's HH GRPs are typically single/low-double digits.
    # If we used raw values directly, a TV row with real spend would be
    # allocated an almost-zero share purely because "20 GRPs" looks tiny
    # sitting next to "2,000,000 impressions" in the same week's pool --
    # even though GRPs and Impressions represent comparable amounts of real
    # media weight in their own units. We min-max normalize each candidate
    # metric to a comparable [0, 1] "relative intensity" scale (using that
    # metric's own dataset-wide max) BEFORE applying it as a weight, so a
    # channel running at its own typical full intensity contributes a
    # comparable weight regardless of which underlying metric it happens to
    # report.
    weight = pd.Series(0.0, index=df.index)
    remaining = weight == 0
    for col in config.ALLOCATION_WEIGHT_PRIORITY:
        if col not in df.columns:
            continue
        col_max = df[col].clip(lower=0).max()
        if not col_max or col_max <= 0:
            continue
        normalized = df[col].clip(lower=0) / col_max
        use = remaining & (normalized > 0)
        weight.loc[use] = normalized.loc[use]
        remaining = weight == 0

    # Any row that still has zero weight (e.g. a row with literally no
    # activity metrics populated) gets an equal-share fallback of 1 so it
    # isn't dropped from the allocation entirely.
    weight.loc[remaining] = 1.0
    df["_activity_weight"] = weight

    week_weight_total = df.groupby(config.DATE_COL)["_activity_weight"].transform("sum")
    df["_weight_share"] = df["_activity_weight"] / week_weight_total.replace(0, np.nan)

    sales_lookup = weekly_sales.set_index(config.DATE_COL)["Total Sales"]
    df["_true_weekly_sales"] = df[config.DATE_COL].map(sales_lookup)

    df["Estimated Sales Contribution"] = (
        df["_weight_share"] * df["_true_weekly_sales"]
    ).fillna(0.0)

    df = df.drop(columns=["_activity_weight", "_weight_share", "_true_weekly_sales"])
    return df


# ---------------------------------------------------------------------------
# 5. Efficiency metrics (Deliverable 2)
# ---------------------------------------------------------------------------
def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series, returning NaN (not inf) wherever the denominator is 0."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def compute_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the standard media-efficiency ratios requested in the brief:
      CTR, Engagement Rate, Video Completion Rate, Cost Per Engagement,
      Cost Per Sale (using the de-duplicated "Estimated Sales Contribution"
      so it isn't distorted by the broadcast-Sales artifact), plus CPC and
      CPM which the Media Efficiency Analysis section also calls for.
    All ratios return NaN (rather than inf or a misleading 0) when their
    denominator is zero, so averages computed downstream aren't skewed.
    """
    df = df.copy()
    df["CTR"] = _safe_divide(df["Clicks"], df["Impressions"])
    df["Engagement Rate"] = _safe_divide(df["Engagements"], df["Impressions"])
    df["Video Completion Rate"] = _safe_divide(df["Video Completes"], df["Video Starts"])
    df["Cost Per Engagement"] = _safe_divide(df["Spend"], df["Engagements"])
    df["CPC"] = _safe_divide(df["Spend"], df["Clicks"])
    df["CPM"] = _safe_divide(df["Spend"], df["Impressions"]) * 1000

    if "Estimated Sales Contribution" in df.columns:
        df["Cost Per Sale"] = _safe_divide(df["Spend"], df["Estimated Sales Contribution"])
    else:
        df["Cost Per Sale"] = _safe_divide(df["Spend"], df[config.TARGET_COL])

    return df


# ---------------------------------------------------------------------------
# 6. Weekly feature table (used by forecasting.py)
# ---------------------------------------------------------------------------
def build_weekly_feature_table(df: pd.DataFrame, weekly_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the row-level dataset to one row per Week with the additive
    media metrics summed (Spend, Impressions, Clicks, Engagements, etc. are
    genuine row-level quantities, unlike Sales, so summing them is correct)
    and join on the de-duplicated true weekly Sales figure as the target.
    """
    agg_cols = [c for c in config.ADDITIVE_METRIC_COLS if c in df.columns]
    weekly_metrics = (
        df.groupby(config.DATE_COL)[agg_cols].sum().reset_index().sort_values(config.DATE_COL)
    )

    weekly = weekly_metrics.merge(weekly_sales, on=config.DATE_COL, how="left")
    weekly = engineer_date_features(weekly)
    weekly = weekly.sort_values(config.DATE_COL).reset_index(drop=True)

    # Re-derive efficiency ratios at the weekly grain too (handy for the
    # Trend Analysis tab and for use as model features)
    weekly["CTR"] = _safe_divide(weekly["Clicks"], weekly["Impressions"])
    weekly["Engagement Rate"] = _safe_divide(weekly["Engagements"], weekly["Impressions"])
    weekly["Video Completion Rate"] = _safe_divide(weekly["Video Completes"], weekly["Video Starts"])
    weekly["Cost Per Engagement"] = _safe_divide(weekly["Spend"], weekly["Engagements"])
    weekly["Cost Per Sale"] = _safe_divide(weekly["Spend"], weekly["Total Sales"])

    return weekly


# ---------------------------------------------------------------------------
# 7. One-call pipeline
# ---------------------------------------------------------------------------
def run_full_pipeline(path=None):
    """
    Run the complete data-prep pipeline and return:
      row_df      -- cleaned, row-level dataframe with date parts, the
                     allocated "Estimated Sales Contribution" column, and
                     all efficiency ratios
      weekly_df   -- one-row-per-week feature table (additive metrics
                     summed + true de-duplicated Sales) ready for modeling
      weekly_sales -- the de-duplicated weekly Sales lookup table on its own
    """
    raw = load_raw_data(path)
    cleaned = clean_data(raw)
    cleaned = engineer_date_features(cleaned)

    weekly_sales = compute_weekly_sales(cleaned)

    row_df = allocate_estimated_sales(cleaned, weekly_sales)
    row_df = compute_efficiency_metrics(row_df)

    weekly_df = build_weekly_feature_table(cleaned, weekly_sales)

    return row_df, weekly_df, weekly_sales
