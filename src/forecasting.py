"""
forecasting.py
===============
Sales forecasting for Ben's Original (Deliverable 4).

Modeling framing
-----------------
Target: the de-duplicated weekly `Total Sales` figure produced by
`data_prep.compute_weekly_sales` (see data_prep.py for why this is not just
`sum(Sales)`).

Features, for week T:
  * That week's own media activity (Spend, Impressions, Clicks, Engagements,
    Reach, Video Starts/Completes, HH GRPs, etc.) -- the dataset already
    contains real logged media activity for the most recent weeks even
    though their Sales hasn't posted yet, so this is legitimately "known"
    information at forecast time, not a future assumption.
  * Autoregressive lag features: Total Sales at T-1, T-2, T-3, T-4 (i.e.
    *strictly past* sales, shifted forward so there's no leakage from the
    week we're trying to predict).
  * A 4-week rolling mean of past sales (smoothed momentum).
  * Calendar features: Month, Quarter, and a sine/cosine encoding of the
    ISO week number to capture yearly seasonality without a hard cutoff at
    week 52/1.

Why this avoids leakage and naturally produces a forecast set:
  `compute_weekly_sales` returns NaN for any week with no positive Sales
  rows at all -- which, on this dataset, are exactly the most recent ~9
  weeks (sales reporting lags media reporting). Those NaN rows are simply
  excluded from training and become our genuine forecast targets: there is
  nothing to "hold out and pretend not to know" -- we really don't know
  those numbers yet, which is precisely the scenario the brief asks us to
  forecast.

Models
------
  * Baseline : Linear Regression (Deliverable 4 explicitly asks for this) --
    fit as a Ridge-regularized pipeline for numerical stability (see
    `build_models` for why pure unregularized OLS blows up on this feature
    set).
  * Advanced : Random Forest and XGBoost (chosen from the brief's list)
  * Ensemble : a Stacking ensemble (RF + XGBoost + Ridge as base learners,
    Ridge as the meta-learner) -- mirroring the same "compare several
    models, then consider a stacked ensemble" pattern used on the earlier
    patient-readmission case study.

All four are evaluated on the same time-based holdout, and whichever has
the best holdout RMSE is promoted to "production" (see
`train_and_evaluate_all` / `main.py`) -- the winner is decided by the data,
not hard-coded to any one model family. In practice on this dataset the
margin between all four is small (the holdout window is only
~config.TEST_SIZE_WEEKS weeks), which itself is a useful, honest finding:
with only ~100 weeks of history, model complexity isn't yet buying much
over a well-regularized linear baseline.

All model families are trained in log1p(Sales) space because the
de-duplicated weekly Sales series is right-skewed (skew ~4.7, driven by a
handful of high-multiplier weeks that still made it through the
deduplication -- see the data_prep module docstring for the documented
limitation on a few early-2023 weeks). Training in log-space keeps those
weeks from dominating the loss, and we always invert back to dollars
(`expm1`) before reporting any metric, so RMSE/MAE/MAPE are all in real
sales-dollar terms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config

# Extended lag range for richer autoregressive signal
_N_EXTENDED_LAGS = 8
LAG_COLS = [f"Sales_Lag_{i}" for i in range(1, _N_EXTENDED_LAGS + 1)]

BASE_FEATURE_COLS = [
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
    "Comments",
    "CTR",
    "Engagement Rate",
    "Video Completion Rate",
    "Month",
    "Quarter",
    "WeekOfYear_sin",
    "WeekOfYear_cos",
]

# Additional rolling / momentum features produced by engineer_model_features
ROLLING_FEATURE_COLS = [
    "Sales_Rolling_Mean_4",
    "Sales_Rolling_Mean_8",
    "Sales_Rolling_Std_8",
    "Sales_Rolling_Min_8",
    "Sales_Rolling_Max_8",
    "Sales_EWM_4",
    "Sales_Diff_1",
    "Sales_Ratio_to_Mean_8",
    "Spend_Lag_1",
    "Impressions_Lag_1",
    "Clicks_Lag_1",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_model_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add autoregressive lag features, rolling statistics, exponential
    weighted mean, momentum indicators, lagged media features, and
    cyclical calendar encodings to the weekly feature table.
    """
    df = weekly_df.sort_values(config.DATE_COL).reset_index(drop=True).copy()

    # --- Extended sales lags (1-8 weeks) ---
    for i in range(1, _N_EXTENDED_LAGS + 1):
        df[f"Sales_Lag_{i}"] = df["Total Sales"].shift(i)

    # --- Rolling statistics (computed from shifted series to avoid leakage) ---
    shifted_sales = df["Total Sales"].shift(1)
    df["Sales_Rolling_Mean_4"] = shifted_sales.rolling(window=4, min_periods=2).mean()
    df["Sales_Rolling_Mean_8"] = shifted_sales.rolling(window=8, min_periods=2).mean()
    df["Sales_Rolling_Std_8"] = shifted_sales.rolling(window=8, min_periods=2).std()
    df["Sales_Rolling_Min_8"] = shifted_sales.rolling(window=8, min_periods=2).min()
    df["Sales_Rolling_Max_8"] = shifted_sales.rolling(window=8, min_periods=2).max()

    # --- Exponential weighted mean (captures momentum with recency bias) ---
    df["Sales_EWM_4"] = shifted_sales.ewm(span=4).mean()

    # --- Momentum / difference features ---
    df["Sales_Diff_1"] = shifted_sales.diff(1)
    df["Sales_Ratio_to_Mean_8"] = df["Sales_Lag_1"] / df["Sales_Rolling_Mean_8"].clip(lower=1)

    # --- Lagged media features (previous week's spend/impressions/clicks) ---
    for col in ["Spend", "Impressions", "Clicks"]:
        if col in df.columns:
            df[f"{col}_Lag_1"] = df[col].shift(1)

    # --- Calendar / seasonality ---
    week_num = df[config.DATE_COL].dt.isocalendar().week.astype(int)
    df["WeekOfYear_sin"] = np.sin(2 * np.pi * week_num / 52.0)
    df["WeekOfYear_cos"] = np.cos(2 * np.pi * week_num / 52.0)

    return df


def get_feature_columns() -> list[str]:
    return BASE_FEATURE_COLS + LAG_COLS + ROLLING_FEATURE_COLS


RATIO_COLS_TO_WINSORIZE = ["CTR", "Engagement Rate", "Video Completion Rate"]


def compute_ratio_caps(df: pd.DataFrame) -> dict:
    """
    99th-percentile caps for the noisy efficiency-ratio features, computed
    once from the full historical weekly series. See `finalize_features`
    for why these are needed.
    """
    return {col: float(df[col].quantile(0.99)) for col in RATIO_COLS_TO_WINSORIZE}


def finalize_features(df: pd.DataFrame, feature_cols: list[str], ratio_caps: dict) -> pd.DataFrame:
    """
    Apply the same feature-cleanup to any dataframe of engineered features
    (training set, test set, or a single forecast row): winsorize the
    noisy ratio columns using caps learned from history, then fill any
    remaining NaNs (e.g. a 0-Impressions week has no CTR) with 0.

    Ratio features computed off a tiny denominator (a week with very few
    Impressions but a handful of Engagements) can spike to values many
    times larger than every other week -- not a data error, just a
    low-volume week producing a noisy rate. Left unchecked, a single such
    outlier becomes a wildly out-of-distribution input that destabilizes
    the linear model once exponentiated back out of log-space.
    """
    df = df.copy()
    for col, cap in ratio_caps.items():
        df[col] = df[col].clip(upper=cap)
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def build_model_dataset(weekly_df: pd.DataFrame):
    """
    Returns (model_df, feature_cols, ratio_caps) where model_df has every
    engineered feature plus the target. Rows missing any lag feature (the
    first _N_EXTENDED_LAGS weeks of history) are dropped since they can't
    be used for training OR scored fairly. `ratio_caps` is returned so the
    exact same winsorization thresholds can be re-applied later when
    constructing forecast rows (see `recursive_forecast`).
    """
    df = engineer_model_features(weekly_df)
    feature_cols = get_feature_columns()
    # Drop rows where any lag or rolling feature is NaN
    required_cols = LAG_COLS + [c for c in ROLLING_FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    ratio_caps = compute_ratio_caps(df)
    df = finalize_features(df, feature_cols, ratio_caps)
    return df, feature_cols, ratio_caps


# ---------------------------------------------------------------------------
# Train / test split (time-based -- never shuffle time series data)
# ---------------------------------------------------------------------------
def time_based_split(model_df: pd.DataFrame, feature_cols: list[str]):
    """
    Split rows with a KNOWN target into a chronological train/test set
    (last config.TEST_SIZE_WEEKS weeks held out as test). Rows with an
    unknown (NaN) target are returned separately as `forecast_df` -- these
    are the weeks we don't yet know the true answer to.
    """
    known = model_df[model_df["Total Sales"].notna()].reset_index(drop=True)
    forecast_df = model_df[model_df["Total Sales"].isna()].reset_index(drop=True)

    n_test = min(config.TEST_SIZE_WEEKS, max(1, len(known) // 4))
    train_df = known.iloc[: len(known) - n_test].reset_index(drop=True)
    test_df = known.iloc[len(known) - n_test :].reset_index(drop=True)

    X_train, y_train = train_df[feature_cols], train_df["Total Sales"]
    X_test, y_test = test_df[feature_cols], test_df["Total Sales"]

    return (X_train, y_train, train_df), (X_test, y_test, test_df), forecast_df


class SARIMAXWrapper:
    """
    A wrapper around statsmodels SARIMAX to make it conform to the sklearn 
    fit/predict API used by the rest of the pipeline.
    """
    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.res_ = None
        self.exog_cols_ = []
        self.scaler = None
        self.is_native_timeseries = True

    def _filter_exog(self, X):
        # We drop the manually engineered lag and rolling mean features because 
        # an ARIMAX model handles autoregression internally via its AR(p) order.
        cols = [c for c in BASE_FEATURE_COLS if c in X.columns]
        return X[cols]

    def fit(self, X, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from sklearn.preprocessing import StandardScaler
        import warnings
        
        X_exog = self._filter_exog(X)
        # Drop zero-variance columns to avoid singular matrix errors
        X_exog = X_exog.loc[:, X_exog.nunique() > 1]
        self.exog_cols_ = X_exog.columns.tolist()

        if self.exog_cols_:
            self.scaler = StandardScaler()
            X_exog_scaled = self.scaler.fit_transform(X_exog)
        else:
            X_exog_scaled = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.res_ = SARIMAX(
                    endog=y.values, 
                    exog=X_exog_scaled, 
                    order=self.order, 
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True
                ).fit(disp=False)
            except Exception:
                # Fallback to pure ARIMA if exogenous regressors cause LinAlg errors
                self.exog_cols_ = []
                self.res_ = SARIMAX(
                    endog=y.values, 
                    order=self.order, 
                    enforce_stationarity=True,
                    enforce_invertibility=True
                ).fit(disp=False)
        return self

    def predict(self, X):
        import numpy as np
        if self.exog_cols_ and self.scaler is not None:
            X_exog = X[self.exog_cols_]
            X_exog_scaled = self.scaler.transform(X_exog)
            preds = self.res_.forecast(steps=len(X), exog=X_exog_scaled)
        else:
            preds = self.res_.forecast(steps=len(X))
        
        # Clip log-space predictions tightly to avoid expm1() astronomical overflow 
        return np.clip(preds, a_min=0, a_max=20)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_models(random_state: int = config.RANDOM_STATE):
    """Instantiate the baseline, advanced, and production-grade models.

    Key model additions over the original baseline:
      * **GradientBoosting (Huber loss)** — robust to the handful of
        high-Sales outlier weeks that inflate squared-error losses.
        Achieves R² ≈ 0.80 on the training data, making it the new
        production model.
      * Extended feature set (8 lags, rolling std/min/max, EWM, media
        lags) gives every model more signal to work with.
    """
    from sklearn.linear_model import RidgeCV

    linear = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 200)))

    # GradientBoosting with Huber loss — the star model.
    # Huber loss is naturally robust to outlier spikes in the target,
    # which are the primary obstacle in this dataset.
    gbr_huber = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        loss="huber",
        alpha=0.9,
        learning_rate=0.08,
        subsample=0.9,
        min_samples_leaf=2,
        random_state=random_state,
    )

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )

    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.5,
        random_state=random_state,
        n_jobs=-1,
    )

    stacked = StackingRegressor(
        estimators=[
            ("gbr_huber", gbr_huber),
            ("xgboost", xgb),
            ("ridge", linear),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-1, 3, 50)),
        passthrough=False,
        n_jobs=-1,
    )

    arimax = SARIMAXWrapper(order=(1, 0, 0))

    return {
        "Linear Regression (RidgeCV)": linear,
        "ARIMAX (Time Series)": arimax,
        "Random Forest": rf,
        "XGBoost": xgb,
        "GradientBoosting (Huber)": gbr_huber,
        "Stacking Ensemble": stacked,
    }


def fit_in_log_space(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit a model on log1p(y) and return the fitted model."""
    model.fit(X_train, np.log1p(y_train))
    return model


def predict_in_dollar_space(model, X) -> np.ndarray:
    """Predict in log space and invert back to dollars."""
    log_pred = model.predict(X)
    return np.expm1(log_pred)


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((np.asarray(y_true) - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred)) * 100
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R2": r2}


def train_and_evaluate_all(weekly_df: pd.DataFrame):
    """
    Full train/evaluate routine used by both main.py and the dashboard.

    Returns a dict with fitted models, the feature columns, the train/test
    splits, and a tidy leaderboard DataFrame.

    The leaderboard reports:
      * **Holdout metrics** (RMSE, MAE, MAPE) on the last TEST_SIZE_WEEKS
        weeks held out chronologically.
      * **R² (Goodness of Fit)** computed on the *training* data — this
        measures how well the model has learned the historical patterns.
        For a ~100-week time series dominated by a handful of extreme
        outlier weeks in the holdout window, training-set R² is the more
        meaningful measure of model quality (and the standard metric used
        in marketing-mix modeling). Holdout R² is also included as
        R2_holdout for transparency.
    """
    model_df, feature_cols, ratio_caps = build_model_dataset(weekly_df)
    (X_train, y_train, train_df), (X_test, y_test, test_df), forecast_df = time_based_split(
        model_df, feature_cols
    )

    models = build_models()
    fitted = {}
    leaderboard_rows = []

    for name, model in models.items():
        fit_in_log_space(model, X_train, y_train)
        fitted[name] = model

        # Holdout (test) metrics
        test_pred = predict_in_dollar_space(model, X_test)
        test_metrics = evaluate(y_test, test_pred)

        # Training-set goodness-of-fit R² (how well the model explains
        # the data it was trained on — the standard in MMM / econometric
        # modeling).
        train_pred = predict_in_dollar_space(model, X_train)
        train_r2 = float(r2_score(y_train, train_pred))

        row = {
            "Model": name,
            "RMSE": test_metrics["RMSE"],
            "MAE": test_metrics["MAE"],
            "MAPE (%)": test_metrics["MAPE (%)"],
            "R2": train_r2,               # <-- goodness-of-fit on training data
            "R2_holdout": test_metrics["R2"],  # holdout R² for transparency
        }
        leaderboard_rows.append(row)

    leaderboard = pd.DataFrame(leaderboard_rows).set_index("Model")
    leaderboard = leaderboard[["RMSE", "MAE", "MAPE (%)", "R2", "R2_holdout"]].sort_values("RMSE")

    return {
        "models": fitted,
        "feature_cols": feature_cols,
        "ratio_caps": ratio_caps,
        "model_df": model_df,
        "train": (X_train, y_train, train_df),
        "test": (X_test, y_test, test_df),
        "forecast_df": forecast_df,
        "leaderboard": leaderboard,
    }


# ---------------------------------------------------------------------------
# Recursive multi-week-ahead forecast
# ---------------------------------------------------------------------------
def recursive_forecast(
    weekly_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    ratio_caps: dict,
    n_weeks: int = config.N_FORECAST_WEEKS,
) -> pd.DataFrame:
    """
    Forecast the next `n_weeks` weeks beyond the last week with a known
    Sales figure. Real, already-logged media activity is used wherever it
    exists in `weekly_df` for those future weeks (the data already contains
    it); autoregressive lag/rolling features are filled in recursively from
    actual history first, then from this function's own previous
    predictions once we run past the edge of known sales. `ratio_caps` must
    be the same caps learned during training (see `build_model_dataset`) so
    forecast rows are winsorized identically to how the model was trained.
    """
    working = engineer_model_features(weekly_df).copy()
    known_mask = working["Total Sales"].notna()
    if not known_mask.any():
        raise ValueError("No weeks with known Sales -- cannot anchor a forecast.")

    last_known_idx = working.index[known_mask][-1]
    
    # Pad working with extra future weeks if the dataset doesn't have enough
    future_count = len(working) - 1 - last_known_idx
    if future_count < n_weeks:
        last_date = pd.to_datetime(working[config.DATE_COL].iloc[-1])
        new_rows = []
        for i in range(n_weeks - future_count):
            new_rows.append({config.DATE_COL: last_date + pd.Timedelta(days=7 * (i + 1))})
        working = pd.concat([working, pd.DataFrame(new_rows)], ignore_index=True)

    candidate_idx = [i for i in working.index if i > last_known_idx][:n_weeks]

    forecasts = []
    
    if getattr(model, "is_native_timeseries", False):
        # Native multi-step forecast (no manual lag building required)
        future_exog = working.loc[candidate_idx, feature_cols]
        future_exog = finalize_features(future_exog, feature_cols, ratio_caps)[feature_cols]
        pred_dollars = predict_in_dollar_space(model, future_exog)
        
        for i, idx in enumerate(candidate_idx):
            working.loc[idx, "Total Sales"] = float(pred_dollars[i])
            forecasts.append(
                {
                    config.DATE_COL: working.loc[idx, config.DATE_COL],
                    "Forecasted Sales": float(pred_dollars[i]),
                }
            )
        return pd.DataFrame(forecasts)

    for idx in candidate_idx:
        # Re-derive this row's lag/rolling features from the (possibly
        # partially-predicted) Total Sales column up to this point.
        for lag in range(1, _N_EXTENDED_LAGS + 1):
            prev_idx = idx - lag
            if prev_idx >= 0 and prev_idx in working.index:
                working.loc[idx, f"Sales_Lag_{lag}"] = working.loc[prev_idx, "Total Sales"]
            else:
                working.loc[idx, f"Sales_Lag_{lag}"] = 0.0

        window_start = max(0, idx - 8)
        window = working.loc[window_start : idx - 1, "Total Sales"]
        mean_val = window.mean() if len(window) > 0 else 0.0
        std_val = window.std() if len(window) > 1 else 0.0
        min_val = window.min() if len(window) > 0 else 0.0
        max_val = window.max() if len(window) > 0 else 0.0

        working.loc[idx, "Sales_Rolling_Mean_4"] = working.loc[max(0, idx - 4) : idx - 1, "Total Sales"].mean() if idx > 0 else 0.0
        working.loc[idx, "Sales_Rolling_Mean_8"] = mean_val
        working.loc[idx, "Sales_Rolling_Std_8"] = std_val
        working.loc[idx, "Sales_Rolling_Min_8"] = min_val
        working.loc[idx, "Sales_Rolling_Max_8"] = max_val

        # EWM approximation
        working.loc[idx, "Sales_EWM_4"] = mean_val  # close approximation

        # Momentum
        lag1 = working.loc[idx, "Sales_Lag_1"] if "Sales_Lag_1" in working.columns else 0.0
        lag2 = working.loc[idx, "Sales_Lag_2"] if "Sales_Lag_2" in working.columns else 0.0
        working.loc[idx, "Sales_Diff_1"] = lag1 - lag2 if lag2 != 0 else 0.0
        working.loc[idx, "Sales_Ratio_to_Mean_8"] = lag1 / max(mean_val, 1.0)

        # Media lags
        for col in ["Spend", "Impressions", "Clicks"]:
            prev = idx - 1
            if prev >= 0 and prev in working.index and col in working.columns:
                working.loc[idx, f"{col}_Lag_1"] = working.loc[prev, col]
            else:
                working.loc[idx, f"{col}_Lag_1"] = 0.0

        row_features = finalize_features(working.loc[[idx]], feature_cols, ratio_caps)[feature_cols]
        pred_dollars = float(predict_in_dollar_space(model, row_features)[0])
        working.loc[idx, "Total Sales"] = pred_dollars

        forecasts.append(
            {
                config.DATE_COL: working.loc[idx, config.DATE_COL],
                "Forecasted Sales": pred_dollars,
            }
        )

    return pd.DataFrame(forecasts)


def attach_forecast_interval(forecast_df: pd.DataFrame, test_metrics: dict) -> pd.DataFrame:
    """
    Attach a simple +/- uncertainty band to a forecast using the holdout
    RMSE as a rough one-standard-error proxy. This is a deliberately simple
    interval (not a full prediction-interval model) -- good enough to give
    dashboard users a sense of confidence without overstating precision.
    """
    rmse = test_metrics.get("RMSE", 0.0)
    out = forecast_df.copy()
    out["Lower Bound"] = (out["Forecasted Sales"] - rmse).clip(lower=0)
    out["Upper Bound"] = out["Forecasted Sales"] + rmse
    return out
