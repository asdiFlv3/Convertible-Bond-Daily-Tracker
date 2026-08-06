"""Backtest summaries and calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest_config import BacktestConfig
from .backtest_engine import (
    BASELINE_COLUMNS_BY_ENGINE,
    FORECAST_COLUMNS_BY_ENGINE,
)


def summarize_backtest(
    backtest: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Compare models on one common forecast sample within each engine."""

    calendar_dates = pd.Index(
        pd.to_datetime(backtest["date"]).dropna().sort_values().unique()
    )
    if len(calendar_dates) < 2:
        raise ValueError("backtest requires at least two distinct dates")
    calendar_split = int(
        round(len(calendar_dates) * (1 - config.validation_fraction))
    )
    calendar_split = min(max(calendar_split, 1), len(calendar_dates) - 1)
    validation_start_date = pd.Timestamp(calendar_dates[calendar_split])

    # The split is global rather than per bond, so every model is judged over
    # the same market regime even when individual histories start later.
    rows: list[dict[str, object]] = []
    for code, group in backtest.groupby("bond_code", sort=False):
        ordered = group.sort_values("date")
        selection_period = ordered.loc[
            ordered["date"].lt(validation_start_date)
        ]
        selection_style = (
            selection_period["bond_style"].iloc[-1]
            if not selection_period.empty
            else np.nan
        )
        for segment, subset in (
            ("calibration", selection_period),
            (
                "validation",
                ordered.loc[ordered["date"].ge(validation_start_date)],
            ),
        ):
            market = subset["market_price"]
            usable_market = market.notna() & market.ne(0)
            for engine, forecast_models in FORECAST_COLUMNS_BY_ENGINE.items():
                common_forecast = (
                    usable_market
                    & subset[list(forecast_models)].notna().all(axis=1)
                )
                # Baseline and all IV variants share this intersection. 
                # This prevents a sparse forecast from looking better by skipping dates that are difficult to price.
                model_columns = (
                    BASELINE_COLUMNS_BY_ENGINE[engine],
                    *forecast_models,
                )
                for model in model_columns:
                    available = subset[model].notna() & usable_market
                    comparison = common_forecast & subset[model].notna()
                    if not comparison.any():
                        mape = np.nan
                        mean_gap_pct = np.nan
                        median_gap_pct = np.nan
                        gap_pct_std = np.nan
                        gap_pct_change_mae = np.nan
                    else:
                        gap_pct = (
                            subset.loc[comparison, model]
                            - market.loc[comparison]
                        ) / market.loc[comparison]
                        mape = gap_pct.abs().mean()
                        mean_gap_pct = gap_pct.mean()
                        median_gap_pct = gap_pct.median()
                        gap_pct_std = gap_pct.std(ddof=1)
                        comparison_positions = pd.Series(
                            np.flatnonzero(comparison.to_numpy()),
                            index=gap_pct.index,
                        )
                        consecutive = comparison_positions.diff().eq(1)
                        gap_pct_change = gap_pct.diff().abs()
                        gap_pct_change_mae = gap_pct_change.loc[
                            consecutive
                        ].mean()
                        gap_pct_change_pairs = int(consecutive.sum())
                    if not comparison.any():
                        gap_pct_change_pairs = 0
                    rows.append(
                        {
                            "bond_code": code,
                            "selection_style": selection_style,
                            "engine": engine,
                            "segment": segment,
                            "segment_start_date": (
                                subset["date"].min()
                                if not subset.empty
                                else pd.NaT
                            ),
                            "segment_end_date": (
                                subset["date"].max()
                                if not subset.empty
                                else pd.NaT
                            ),
                            "validation_start_date": validation_start_date,
                            "model": model,
                            "total_days": len(subset),
                            "priced_days": int(available.sum()),
                            "comparison_days": int(comparison.sum()),
                            "raw_coverage": (
                                available.mean()
                                if len(available)
                                else np.nan
                            ),
                            "comparison_coverage": (
                                comparison.mean()
                                if len(comparison)
                                else np.nan
                            ),
                            "mape": mape,
                            "mean_gap_pct": mean_gap_pct,
                            "median_gap_pct": median_gap_pct,
                            "gap_pct_std": gap_pct_std,
                            "gap_pct_change_mae": gap_pct_change_mae,
                            "gap_pct_change_pairs": gap_pct_change_pairs,
                        }
                    )
    return pd.DataFrame(rows)

def summarize_by_style(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate errors by style frozen at the validation cutoff."""

    return (
        summary.groupby(
            ["selection_style", "engine", "segment", "model"],
            sort=False,
            dropna=False,
        )
        .agg(
            bond_count=("bond_code", "nunique"),
            earliest_segment_start=("segment_start_date", "min"),
            latest_segment_end=("segment_end_date", "max"),
            total_comparison_days=("comparison_days", "sum"),
            mean_raw_coverage=("raw_coverage", "mean"),
            mean_comparison_coverage=("comparison_coverage", "mean"),
            mean_mape=("mape", "mean"),
            median_mape=("mape", "median"),
            mean_gap_pct=("mean_gap_pct", "mean"),
            mean_gap_pct_std=("gap_pct_std", "mean"),
            mean_gap_pct_change_mae=("gap_pct_change_mae", "mean"),
        )
        .reset_index()
    )


def calibration_diagnostics(backtest: pd.DataFrame) -> pd.DataFrame:
    """Summarize solver success without treating same-day fit as forecast."""

    calibration = backtest.loc[backtest["is_calibration_date"]].copy()
    rows: list[dict[str, object]] = []
    for code, group in calibration.groupby("bond_code", sort=False):
        for prefix in ("iv_call", "iv_no_call"):
            success = group[f"{prefix}_success"].fillna(False).astype(bool)
            implied = pd.to_numeric(
                group[f"{prefix}_implied_sigma"],
                errors="coerce",
            )
            ratio = implied / group["historical_sigma"]
            reasons = (
                group.loc[~success, f"{prefix}_reason"]
                .fillna("missing")
                .value_counts()
                .to_dict()
            )
            rows.append(
                {
                    "bond_code": code,
                    "engine": prefix,
                    "calibration_days": len(group),
                    "solution_days": int(success.sum()),
                    "solution_share": success.mean(),
                    "median_implied_sigma": implied.loc[success].median(),
                    "median_implied_to_historical": ratio.loc[success].median(),
                    "failure_reasons": str(reasons),
                }
            )
    return pd.DataFrame(rows)

