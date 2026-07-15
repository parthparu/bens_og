# Ben's Original -- Marketing Analytics Dashboard & Sales Forecasting

Adbureau Analytics Internship Assignment #2.

An interactive Streamlit dashboard (Part A) and a weekly sales forecasting
pipeline (Part B) built on Ben's Original's 2023-2025 cross-channel media
dataset (Digital, Display, Video, Social, Retail Media, TV).

## Quick start

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run dashboard/app.py

# (optional) Run the full pipeline from the command line instead
# -- prints a full EDA/forecast/explainability summary and saves
#    artifacts (trained models, cleaned CSVs, forecast) to outputs/
python main.py
```

The dashboard reads `data/Bens_Original.xlsx` by default -- no extra setup
needed.

## Project structure

```
bens_original_analytics/
├── data/
│   └── Bens_Original.xlsx        # source dataset (as provided)
├── src/                           # all core logic -- no Streamlit imports here
│   ├── config.py                  # paths, column names, cleaning lookups, model/UI constants
│   ├── data_prep.py                # cleaning, date features, the Sales de-duplication fix,
│   │                                #   the sales-allocation logic, efficiency ratios
│   ├── eda.py                      # reusable aggregations (KPIs, breakdowns, correlation, efficiency)
│   ├── forecasting.py              # feature engineering, model training/evaluation, recursive forecast
│   ├── explainability.py           # linear coefficients, tree importances, SHAP
│   └── ai_insights.py              # Deliverable 6 -- data-grounded marketing narrative
├── dashboard/
│   ├── app.py                      # the Streamlit app (10 tabs, see below)
│   └── charts.py                   # reusable, brand-themed Plotly figure builders
├── outputs/                        # created when you run main.py
│   ├── models/                     # joblib-serialized trained models
│   ├── reports/                    # CSV exports (cleaned data, leaderboard, forecast, SHAP table)
│   └── figures/
├── main.py                         # CLI pipeline runner (clean -> EDA -> train -> forecast -> explain)
├── requirements.txt
└── README.md
```

Every notebook-style computation lives in `src/`, with no Streamlit
dependency at all -- `dashboard/app.py` only *calls* those functions and
renders the results. That means the exact same cleaning/modeling code
powers both the dashboard and the standalone `main.py` CLI run, so the two
can never silently disagree with each other.

## The most important data-cleaning decision: the "Sales" column

This is worth reading before anything else, and it's also explained live in
the dashboard's **Data Quality & Methodology** tab.

The raw file has **12,529 rows but only 122 distinct weeks**. Within a
single week, the `Sales` column takes on **multiple distinct values** --
anywhere from 1 to 17 unique values per week -- and those values are almost
always clean integer multiples of one another (2x, 6x, 9x, 18x, 24x...).
The **minimum non-zero Sales value observed in a given week** is
remarkably stable and economically sensible (~$7M-$10M/week, smooth
seasonal variation), while the larger multiples within that same week jump
unpredictably -- sometimes into the billions -- purely as a function of how
many Creative/Audience/Site rows happen to share that week.

**Conclusion:** `Sales` was originally a single weekly, brand-level KPI
(one true number per week) that got broadcast onto every granular media row
for that week somewhere upstream of this export -- most likely because the
original source had a multi-valued field (e.g. an Audience or Creative tag
listing several values in one cell) that was "exploded" into one row per
value, carrying the un-split Sales figure along with it into every
resulting row. **Summing the raw column would over-count by a different
factor for every channel/creative slice**, distorting not just the totals
but the *relative ranking* between channels.

**Our fix, in two parts** (see `src/data_prep.py` for the full, heavily
commented implementation):

1. **True Weekly Sales** (the actual KPI / forecasting target) = the
   minimum non-zero `Sales` value observed across the whole dataset in a
   given week. This is what powers the Executive Summary KPI, the weekly
   trend chart, and the forecasting target.
2. **Estimated Sales Contribution** (for the "Sales by Channel/Site/Device"
   charts) = that week's true total, allocated to individual rows
   proportionally to each row's share of that week's media activity
   (**Spend first** -- dollars are directly comparable across every paid
   channel -- falling back to a normalized Impressions / Reach / GRPs /
   Engagements / Likes intensity score for organic/earned/owned rows with
   no recorded spend). By construction, this always sums back exactly to
   the true weekly total, so it's a directionally useful, clearly-labelled
   *estimate* without re-introducing the inflation problem.

**Known limitation, stated plainly:** for the first ~9 weeks of 2023
(before TV media launched), no low-multiplier data source was present to
anchor the true base value, so the de-duplicated Sales figure for those
early weeks may still be modestly inflated relative to its true atomic
value. This is flagged in the dashboard rather than hidden.

### Other cleaning steps
- Whitespace stripped from every text column; fixed the `"VIdeo"` ->
  `"Video"` typo in Sub-Channel.
- *Structurally* missing dimensions (Device, Partner Type, Audience,
  Platform Type, Site, Creative) filled with an explicit label rather than
  dropped or imputed -- these columns simply aren't tracked by certain data
  sources (e.g. TV has no Device), so an explicit "Not Applicable"/"Not
  Tracked" label is more honest than a mean/mode fill.
- 2 rows with negative Spend are flagged (not deleted) -- they read like
  legitimate TV credit/rebate adjustments.
- All efficiency ratios (CTR, Engagement Rate, Video Completion Rate, CPC,
  CPM, Cost Per Engagement, Cost Per Sale) return `NaN` rather than `inf`
  or a misleading `0` when their denominator is zero, and category-level
  rollups re-derive the ratio from summed numerators/denominators rather
  than averaging the row-level ratio (a classic Simpson's-paradox trap).

## The dashboard (Part A)

10 tabs, covering every item in the brief:

| Tab | Covers |
|---|---|
| Executive Summary | KPIs, weekly sales trend, media-type breakdown, top-performer snapshot |
| Trend Analysis | Weekly Sales / Spend / Impressions / Engagement trends |
| Sales Drivers | Estimated Sales Contribution by Media Type, Channel, Site, Device, Platform Type |
| Correlation | Spend/Impressions/Clicks/Engagements vs. Sales, at the weekly grain |
| Top Campaigns | Top 20 creatives by Estimated Sales Contribution |
| Media Efficiency | CTR, CPC, CPM, Cost Per Engagement, by any chosen dimension |
| Forecasting | Model leaderboard, actual-vs-forecast chart, adjustable horizon, CSV export |
| Explainability | Linear coefficients, tree feature importances, SHAP (bar + beeswarm) |
| AI Insights | Marketing Insights / Media Recommendations / Optimization Suggestions |
| Data Quality & Methodology | The full Sales write-up above, live in the app, plus a reconciliation check |

All six filter dimensions from the brief (Media Type, Channel, Sub-Channel,
Site, Device, Platform Type) live in the sidebar and apply to every
activity-based view. Weekly Sales itself and the Forecasting tab are
explicitly **not** filtered, since there's no genuine channel-level sales
ground truth to filter down to -- this is called out in-app so it's never
ambiguous which numbers move with the filters and which don't.

## Sales forecasting (Part B)

**Target:** the de-duplicated weekly `Total Sales` figure above.

**Features, for week T:** that week's own media activity (Spend,
Impressions, Clicks, Engagements, Reach, Video Starts/Completes, HH GRPs,
etc. -- genuinely known at forecast time, since the export already
contains real, already-logged media activity for the most recent weeks
even though their Sales hasn't posted yet), four weeks of autoregressive
Sales lags, an 8-week rolling Sales mean, and cyclical calendar features
(Month, Quarter, sine/cosine of ISO week number).

**Models trained and compared on an identical time-based holdout** (the
data is never shuffled -- the last several weeks are always held out
chronologically):
- **Baseline:** Linear Regression, fit as a Ridge-regularized pipeline.
  Plain unregularized OLS was tested first and found to be numerically
  unstable on this feature set (highly correlated features + ~100 training
  rows + log-space target = a small fitting error gets blown up
  exponentially once converted back to dollars) -- the regularization
  strength was tuned via a holdout sweep and is documented in
  `src/forecasting.py`.
- **Advanced:** Random Forest and XGBoost.
- **Ensemble:** a Stacking Regressor (Random Forest + XGBoost + Ridge,
  with a Ridge meta-learner).

All four are trained in `log1p(Sales)` space (the weekly Sales series is
right-skewed) and every reported metric (RMSE, MAE, MAPE, R²) is converted
back to real dollars before evaluation. **Whichever model has the best
holdout RMSE is automatically promoted** for the 4-week-ahead forecast --
nothing is hard-coded to a particular model family, and the dashboard shows
exactly which one won and why.

The 4-week-ahead forecast is generated **recursively**: real, already-
logged media activity is used for each of the next 4 weeks (the export
already contains it), and the autoregressive lag/rolling features are
filled in from actual history first, then from the model's own prior
predictions once the forecast horizon runs past the edge of known sales.

## Explainability (Deliverable 5)

Three complementary views, all available in the **Explainability** tab:
1. Standardized Ridge coefficients (directly interpretable, linear/additive
   view).
2. Random Forest / XGBoost `feature_importances_` (captures non-linear
   splits and interactions).
3. SHAP values computed on the XGBoost model (a game-theoretic, per-week
   decomposition, used as a fast, faithful proxy for explaining the
   ensemble -- `TreeExplainer` doesn't support stacking meta-estimators
   directly).

**Headline finding:** recent sales momentum (the autoregressive lag
features) dominates every one of these views -- more so than any single
media metric. Individually, no weekly media metric shows more than a weak
correlation with sales (`|r| < 0.15` throughout). This is reported plainly
rather than dressed up, since it's a genuinely useful, honest takeaway: for
an established national CPG brand, week-to-week sales tend to be driven
more by underlying demand and distribution momentum than by any single
week's media flighting -- and a future iteration would likely benefit more
from adding price/promotion/distribution/competitor data than from further
tuning the current media-only feature set.

## AI-Powered Insights (Deliverable 6)

`src/ai_insights.py` generates the Marketing Insights / Media
Recommendations / Optimization Suggestions narrative directly from this
project's own computed outputs (channel allocation, efficiency table,
model leaderboard, SHAP importances, correlations) rather than calling out
to an external LLM API -- this entire project was built working with
Claude, so that step already happened, and the insights regenerate
automatically if the underlying numbers change rather than going stale.

## Notes on scope

This deliverable focuses on the dashboard (Part A) and the forecasting
model (Part B), per the brief's two named objectives -- the optional Bonus
Challenge (a LangChain/LlamaIndex/ChromaDB conversational agent) was
intentionally left out of scope to keep the core analytics and modeling
work as strong as possible.
