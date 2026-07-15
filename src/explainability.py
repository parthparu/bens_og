"""
explainability.py
==================
"What drives sales?" (Deliverable 5).

We surface three complementary views, since each model family answers a
slightly different question:

  1. Ridge coefficients (standardized) from the linear baseline -- the most
     directly interpretable: "holding everything else fixed, which features
     move the (log) sales prediction up or down, and by how much, in a
     purely linear/additive sense?"
  2. Gini-based feature_importances_ from the Random Forest / XGBoost
     models -- "which features did the trees split on most, and how much
     did those splits reduce error?" (captures non-linear effects and
     interactions that the linear view can't).
  3. SHAP values computed on the XGBoost model -- the most rigorous view:
     a game-theoretic decomposition of *each individual week's* prediction
     into per-feature contributions, which can then be averaged for a
     global ranking or inspected week-by-week. We use XGBoost (rather than
     the Stacking ensemble itself) as the SHAP "explainer model" because
     TreeExplainer doesn't support meta-estimators directly -- this is a
     common, accepted practice: explain the best available single tree
     model as a faithful proxy for what's driving the ensemble, since the
     ensemble's other members (Ridge, Random Forest) broadly agree with it
     directionally on this dataset.

All three are reported in log1p(Sales) space (since that's the space every
model was actually fit in) -- a positive SHAP value means "this feature
pushed the predicted log-sales up", not a literal dollar amount. We label
this clearly in the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def linear_coefficients(fitted_pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """
    Pull standardized coefficients out of a (StandardScaler -> Ridge)
    pipeline. Because the inputs are standardized, coefficient magnitude is
    directly comparable across features -- the largest |coefficient| is the
    feature that moves log(Sales) the most per one-standard-deviation
    change.
    """
    ridge = fitted_pipeline.named_steps.get("ridge") or fitted_pipeline.steps[-1][1]
    coefs = pd.Series(ridge.coef_, index=feature_cols, name="Standardized Coefficient")
    out = coefs.to_frame()
    out["Abs Coefficient"] = out["Standardized Coefficient"].abs()
    return out.sort_values("Abs Coefficient", ascending=False).drop(columns="Abs Coefficient")


def tree_feature_importance(fitted_model, feature_cols: list[str]) -> pd.DataFrame:
    """Gini/gain-based feature_importances_ from a Random Forest or XGBoost model."""
    importances = pd.Series(fitted_model.feature_importances_, index=feature_cols, name="Importance")
    return importances.sort_values(ascending=False).to_frame()


def compute_shap_values(xgb_model, X: pd.DataFrame):
    """
    Return (shap_values, explainer) for an XGBoost model using the fast,
    exact TreeExplainer. `X` should be the feature matrix the model was
    trained/evaluated on (e.g. the training set, for the most stable global
    ranking).
    """
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X)
    return shap_values, explainer


def shap_global_importance(shap_values, feature_cols: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature -- a global importance ranking."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    out = pd.Series(mean_abs, index=feature_cols, name="Mean |SHAP value|")
    return out.sort_values(ascending=False).to_frame()


def channel_engagement_correlation(row_df: pd.DataFrame) -> pd.DataFrame:
    """
    Answers "which engagement metrics matter most?" / "does spend
    correlate with sales?" at the WEEKLY-AGGREGATE level (correlating raw
    row-level activity against a deduplicated weekly Sales figure would mix
    granularities incorrectly -- this expects a weekly-aggregated frame to
    be passed in, e.g. the `weekly_df` from data_prep, not the row-level
    `row_df`. Kept here for a single, shared place to compute/describe this
    correlation table for both the EDA tab and the explainability writeup).
    """
    cols = [
        "Total Sales",
        "Spend",
        "Impressions",
        "Clicks",
        "Engagements",
        "Reach",
        "Video Starts",
        "Video Completes",
        "HH GRPs (TV)",
        "Likes",
        "Comments",
    ]
    cols = [c for c in cols if c in row_df.columns]
    corr = row_df[cols].corr()["Total Sales"].drop("Total Sales")
    return corr.sort_values(ascending=False).to_frame(name="Correlation with Total Sales")
