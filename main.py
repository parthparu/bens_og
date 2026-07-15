"""
main.py
=======
Command-line entry point that runs the full pipeline end-to-end:
clean -> EDA summary -> train/evaluate models -> forecast -> explainability
-> AI insights, printing a readable summary to the console and saving
artifacts (trained models, the cleaned weekly dataset, and the leaderboard)
to outputs/.

Usage:
    python main.py
"""

from __future__ import annotations

import warnings

import joblib
import pandas as pd

from dashboard import charts
from src import ai_insights, config, data_prep, eda, explainability, forecasting

warnings.filterwarnings("ignore")

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
pd.set_option("display.width", 160)


def _section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    _section("1. DATA PREPARATION")
    row_df, weekly_df, weekly_sales = data_prep.run_full_pipeline()
    print(f"Row-level dataset : {row_df.shape[0]:,} rows x {row_df.shape[1]} columns")
    print(f"Weekly feature table: {weekly_df.shape[0]} weeks")
    print(f"Weeks with known Sales: {weekly_sales['Total Sales'].notna().sum()} / {len(weekly_sales)}")

    raw = data_prep.load_raw_data()
    miss = eda.missingness_report(raw)
    if not miss.empty:
        print("\nMissing-value summary (raw file):")
        print(miss)

    row_df.to_csv(config.REPORTS_DIR / "row_level_clean.csv", index=False)
    weekly_df.to_csv(config.REPORTS_DIR / "weekly_feature_table.csv", index=False)
    weekly_sales.to_csv(config.REPORTS_DIR / "weekly_sales.csv", index=False)
    raw.to_csv(config.REPORTS_DIR / "raw_data.csv", index=False)

    # ------------------------------------------------------------------
    _section("2. EXPLORATORY ANALYSIS")
    kpis = eda.kpi_summary(row_df, weekly_sales)
    print("Executive KPIs:")
    for k, v in kpis.items():
        print(f"  {k:20s}: {v:,.0f}")

    print("\nEstimated Sales Contribution by Media Type:")
    print(eda.breakdown_by(row_df, "Media Type"))

    print("\nEstimated Sales Contribution by Channel:")
    print(eda.breakdown_by(row_df, "Channel"))

    print("\nTop 10 Creatives by Estimated Sales Contribution:")
    print(eda.top_creatives(row_df, n=10)[["Creative", "Estimated Sales Contribution"]])

    print("\nWeekly correlation matrix:")
    print(eda.correlation_matrix(weekly_df))

    # ------------------------------------------------------------------
    _section("3. SALES FORECASTING")
    result = forecasting.train_and_evaluate_all(weekly_df)
    print("Model leaderboard (holdout test weeks, sorted by RMSE):")
    print(result["leaderboard"])

    best_name = result["leaderboard"].index[0]
    best_model = result["models"][best_name]
    print(f"\nProduction model selected: {best_name}")

    joblib.dump(best_model, config.MODELS_DIR / "production_model.joblib")
    joblib.dump(result, config.MODELS_DIR / "model_result.joblib")
    for name, model in result["models"].items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, config.MODELS_DIR / f"{safe_name}.joblib")
    result["leaderboard"].to_csv(config.REPORTS_DIR / "model_leaderboard.csv")

    forecast = forecasting.recursive_forecast(
        weekly_df, best_model, result["feature_cols"], result["ratio_caps"], n_weeks=config.N_FORECAST_WEEKS
    )
    forecast = forecasting.attach_forecast_interval(forecast, result["leaderboard"].loc[best_name].to_dict())
    print(f"\nNext {config.N_FORECAST_WEEKS}-week forecast ({best_name}):")
    print(forecast)
    forecast.to_csv(config.REPORTS_DIR / "next_4_week_forecast.csv", index=False)

    # ------------------------------------------------------------------
    _section("4. EXPLAINABILITY")
    xgb_model = result["models"]["XGBoost"]
    X_train, y_train, train_df = result["train"]
    shap_values, _ = explainability.compute_shap_values(xgb_model, X_train)
    shap_importance = explainability.shap_global_importance(shap_values, result["feature_cols"])
    print("Top 10 SHAP feature importances (XGBoost, log-Sales space):")
    print(shap_importance.head(10))

    correlation = explainability.channel_engagement_correlation(weekly_df)
    print("\nWeekly metric correlation with Total Sales:")
    print(correlation)

    shap_importance.to_csv(config.REPORTS_DIR / "shap_feature_importance.csv")
    joblib.dump({"shap_values": shap_values, "shap_importance": shap_importance}, config.MODELS_DIR / "shap_data.joblib")

    # ------------------------------------------------------------------
    _section("5. AI-POWERED INSIGHTS")
    channel_sales = eda.breakdown_by(row_df, "Channel")
    efficiency = eda.efficiency_by_dimension(row_df, "Channel")
    insights = ai_insights.generate_insights(
        channel_sales, efficiency, result["leaderboard"], shap_importance, correlation
    )
    for section, bullets in insights.items():
        print(f"\n{section}:")
        for b in bullets:
            print(f"  - {b}")

    print("\nDone. Artifacts saved to:", config.OUTPUTS_DIR)

    # ------------------------------------------------------------------
    _section("6. GENERATING DEFAULT PLOTS")
    print("Generating default interactive plots and saving to outputs/figures/ ...")
    
    # 1. Weekly Sales Trend
    fig_sales_trend = charts.line_trend(weekly_sales.dropna(subset=["Total Sales"]), config.DATE_COL, "Total Sales", y_title="Sales ($)")
    fig_sales_trend.write_html(config.FIGURES_DIR / "weekly_sales_trend.html")
    
    # 2. Estimated Sales Contribution by Media Type
    mt = eda.breakdown_by(row_df, "Media Type")
    fig_media_type = charts.bar_breakdown(mt, "Media Type", "Estimated Sales Contribution")
    fig_media_type.write_html(config.FIGURES_DIR / "sales_by_media_type.html")
    
    # 3. Correlation Heatmap
    corr = eda.correlation_matrix(weekly_df, ["Spend", "Impressions", "Clicks", "Engagements", "Total Sales"])
    fig_corr = charts.correlation_heatmap(corr, "Spend / Impressions / Clicks / Engagements / Sales")
    fig_corr.write_html(config.FIGURES_DIR / "correlation_heatmap.html")
    
    # 4. Model Leaderboard
    fig_leaderboard = charts.leaderboard_bar(result["leaderboard"], "RMSE", "Holdout RMSE by Model (lower is better)")
    fig_leaderboard.write_html(config.FIGURES_DIR / "model_leaderboard.html")
    
    # 5. Forecast Chart
    history = weekly_sales.dropna(subset=["Total Sales"]).tail(26)
    fig_forecast = charts.forecast_chart(history, forecast, f"Actual vs. Forecasted Weekly Sales (next {config.N_FORECAST_WEEKS} weeks)")
    fig_forecast.write_html(config.FIGURES_DIR / "forecast_chart.html")
    
    # 6. Spend vs Sales Scatter
    fig_spend_sales = charts.spend_vs_sales_scatter(weekly_df, "How Spend Affects Sales")
    fig_spend_sales.write_html(config.FIGURES_DIR / "spend_vs_sales.html")
    
    print("Plots generated successfully.")


if __name__ == "__main__":
    main()
