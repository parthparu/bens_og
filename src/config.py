"""
config.py
=========
Central configuration for the Ben's Original Marketing Analytics project.

Keeping every path, column name, and "magic number" in one place means the
rest of the codebase (data_prep, eda, forecasting, explainability, the
Streamlit app) never hard-codes a string twice. If a column gets renamed in
a future data refresh, this is the only file that needs to change.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RAW_DATA_PATH = DATA_DIR / "df.xlsx"
RAW_SHEET_NAME = "Sheet2"

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
DATE_COL = "Week Starting Sunday"

DIMENSION_COLS = [
    "TAB",
    "Media Type",
    "Channel",
    "Sub-Channel",
    "Site",
    "Device",
    "Platform Type",
    "Partner Type",
    "Audience",
    "Creative",
]

METRIC_COLS = [
    "Spend",
    "Impressions",
    "Clicks",
    "Video Starts",
    "Video Completes",
    "HH GRPs (TV)",
    "Impact",
    "Reach",
    "Likes",
    "Engagements",
    "Positive Comments",
    "Neutral comments",
    "Base Dollar Amount",
    "Replies",
    "Reposts",
    "Mention Volume",
    "Post Volume",
    "Comments",
]

# Metrics that are genuinely additive at the row level (i.e. NOT the
# duplicated/broadcast "Sales" column -- see data_prep.py for the full
# write-up of why Sales is treated completely differently).
ADDITIVE_METRIC_COLS = [c for c in METRIC_COLS if c != "Base Dollar Amount"]

TARGET_COL = "Base Dollar Amount"

FILTER_DIMENSIONS = [
    "Media Type",
    "Channel",
    "Sub-Channel",
    "Site",
    "Device",
    "Platform Type",
]

# Candidate columns used (in priority order) as the "activity weight" when
# allocating the deduplicated weekly Sales figure down to a row/category
# level for the "Sales by X" bar charts. Spend is tried first since dollars
# are a directly comparable unit across every *paid* channel (TV, Paid
# Social, Digital); the non-monetary metrics are fallbacks for organic /
# earned / owned rows that have no recorded spend at all. Each candidate is
# min-max normalized to its own dataset-wide max before use (see
# data_prep.allocate_estimated_sales) so e.g. TV's small-magnitude GRPs and
# Social's large-magnitude Impressions become comparable "relative
# intensity" scores rather than being compared as raw units.
ALLOCATION_WEIGHT_PRIORITY = [
    "Spend",
    "Impressions",
    "Reach",
    "HH GRPs (TV)",
    "Engagements",
    "Likes",
]

# ---------------------------------------------------------------------------
# Cleaning lookups
# ---------------------------------------------------------------------------
# Known free-text inconsistency spotted during EDA: "VIdeo" is a typo of
# "Video" in the Sub-Channel column (13 of 12,529 rows).
SUB_CHANNEL_TYPO_FIX = {"VIdeo": "Video"}

# Columns whose missingness is *structural* (a given TAB/data source simply
# doesn't track that dimension) rather than random/accidental. We fill these
# with an explicit "Not Applicable" label instead of dropping rows or
# imputing a misleading value.
STRUCTURAL_NA_FILL = {
    "Site": "Unknown",
    "Device": "Not Tracked",
    "Platform Type": "Unknown",
    "Partner Type": "Not Applicable",
    "Audience": "Not Applicable",
    "Creative": "Unspecified Creative",
}

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_FORECAST_WEEKS = 12
N_LAG_WEEKS = 4          # how many weeks of sales history to use as lag features
TEST_SIZE_WEEKS = 12     # holdout window for time-based train/test split

# ---------------------------------------------------------------------------
# Dashboard look & feel
# ---------------------------------------------------------------------------
BRAND_PRIMARY = "#1C4C74"   # Dark Blue
BRAND_SECONDARY = "#349CE4"  # Bright Blue
BRAND_DARK = "#354551"
PLOTLY_TEMPLATE = "plotly_white"

CURRENCY_FMT = "${:,.0f}"
