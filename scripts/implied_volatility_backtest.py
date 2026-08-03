"""Leakage-safe implied-volatility backtest for saved batch results.

Contemporaneous implied volatility is a diagnostic because it is inverted from
the same market price being explained.  This script calibrates it only on
scheduled dates and permits that observation to enter model pricing from the
next trading day onward.  The normal Wind-backed tracking pipeline and its
historical-volatility baseline remain unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crr_model import ManualModelTerms, price_convertible_with_terms
from implied_volatility import ImpliedVolatilityResult, solve_implied_volatility


@dataclass(frozen=True)
class BacktestConfig:
    """File locations, IV calibration cadence, and validation thresholds."""

    input_csv: Path = Path(
        "output/classic/batch/all_9/daily_tracking_all.csv"
    )
    summary_csv: Path = Path(
        "output/classic/batch/all_9/comparison_summary.csv"
    )
    output_dir: Path = Path("output/diagnostics/implied_volatility")
    calibration_stride: int = 5
    rolling_calibration_count: int = 4
    max_iv_age_trading_days: int = 10
    shrinkage_weight: float = 0.5
    validation_fraction: float = 0.4
    sigma_grid: tuple[float, ...] = (
        0.01,
        0.10,
        0.20,
        0.40,
        0.60,
        1.00,
        1.50,
        2.00,
        3.00,
    )
    solver_iterations: int = 24
    solver_price_tolerance: float = 0.01
    minimum_forecast_coverage: float = 0.80


FORECAST_COLUMNS_BY_ENGINE = {
    "with_call": (
        "price_iv_latest_calibration_with_call",
        "price_iv_median_recent_calibrations_with_call",
        "price_iv_shrunk_to_historical_with_call",
    ),
    "no_call": (
        "price_iv_latest_calibration_no_call",
        "price_iv_median_recent_calibrations_no_call",
        "price_iv_shrunk_to_historical_no_call",
    ),
}
FORECAST_COLUMNS = tuple(
    model
    for models in FORECAST_COLUMNS_BY_ENGINE.values()
    for model in models
)
BASELINE_COLUMNS_BY_ENGINE = {
    "with_call": "baseline_price_with_call",
    "no_call": "baseline_price_no_call",
}


def load_terms_snapshot(summary: pd.DataFrame) -> dict[str, ManualModelTerms]:
    """Reconstruct the exact per-bond assumptions saved by the batch run."""

    required = [
        "bond_code",
        "risk_free_rate",
        "dividend_yield",
        "credit_spread",
        "call_parity_trigger",
        "put_parity_trigger",
        "put_price",
        "tree_steps",
    ]
    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise KeyError(f"comparison summary missing terms columns: {missing}")
    if summary["bond_code"].isna().any():
        raise ValueError("comparison summary contains a missing bond code")
    normalized_codes = summary["bond_code"].astype(str)
    duplicated = normalized_codes.duplicated(keep=False)
    if duplicated.any():
        codes = normalized_codes.loc[duplicated].unique().tolist()
        raise ValueError(f"comparison summary contains duplicate bonds: {codes}")

    def finite_value(row: object, column: str) -> float:
        """Read one required finite scalar from a saved terms row."""

        value = float(getattr(row, column))
        if not np.isfinite(value):
            code = str(getattr(row, "bond_code"))
            raise ValueError(f"{code} has invalid {column}: {value}")
        return value

    has_blend_snapshot = {
        "debt_equity_blend_low",
        "debt_equity_blend_high",
    }.issubset(summary.columns)
    result: dict[str, ManualModelTerms] = {}
    for row in summary.itertuples(index=False):
        code = str(row.bond_code)
        blend_low = (
            finite_value(row, "debt_equity_blend_low")
            if has_blend_snapshot
            else 70.0
        )
        blend_high = (
            finite_value(row, "debt_equity_blend_high")
            if has_blend_snapshot
            else 130.0
        )
        tree_steps_value = finite_value(row, "tree_steps")
        tree_steps = int(tree_steps_value)
        if tree_steps <= 0 or tree_steps != tree_steps_value:
            raise ValueError(f"{code} has invalid tree_steps: {tree_steps_value}")
        result[code] = ManualModelTerms(
            risk_free_rate=finite_value(row, "risk_free_rate"),
            dividend_yield=finite_value(row, "dividend_yield"),
            credit_spread=finite_value(row, "credit_spread"),
            debt_equity_blend_low=blend_low,
            debt_equity_blend_high=blend_high,
            call_parity_trigger=finite_value(row, "call_parity_trigger"),
            put_parity_trigger=finite_value(row, "put_parity_trigger"),
            put_price=finite_value(row, "put_price"),
            tree_steps=tree_steps,
        )
        if blend_low <= 0 or blend_low >= blend_high:
            raise ValueError(
                f"{code} blend thresholds must satisfy 0 < low < high"
            )
        if result[code].put_price <= 0:
            raise ValueError(f"{code} put_price must be positive")
        if result[code].call_parity_trigger <= 0:
            raise ValueError(f"{code} call_parity_trigger must be positive")
        if result[code].put_parity_trigger <= 0:
            raise ValueError(f"{code} put_parity_trigger must be positive")
    return result


def _price(
    row: object,
    terms: ManualModelTerms,
    sigma: float,
    *,
    with_call: bool,
) -> float:
    """Reprice one saved row with a supplied volatility and engine variant."""

    return price_convertible_with_terms(
        stock_price=float(getattr(row, "stock_close")),
        conversion_price=float(getattr(row, "K")),
        sigma=float(sigma),
        maturity_years=float(getattr(row, "T")),
        maturity_redemption_price=float(
            getattr(row, "maturity_redemption_price")
        ),
        coupon_rate=float(getattr(row, "coupon_used")),
        conversion_wait_years=float(getattr(row, "t_conv_used")),
        put_wait_years=float(getattr(row, "t_put_used")),
        terms=terms,
        with_call=with_call,
    )


def _solve_row(
    row: object,
    terms: ManualModelTerms,
    config: BacktestConfig,
    *,
    with_call: bool,
) -> ImpliedVolatilityResult:
    """Invert one row's market price under the selected call treatment."""

    return solve_implied_volatility(
        lambda sigma: _price(
            row,
            terms,
            sigma,
            with_call=with_call,
        ),
        float(getattr(row, "bond_close")),
        sigma_grid=config.sigma_grid,
        max_iterations=config.solver_iterations,
        price_tolerance=config.solver_price_tolerance,
    )


def _solver_fields(
    prefix: str,
    result: ImpliedVolatilityResult,
) -> dict[str, object]:
    """Flatten a solver result into prefixed columns for CSV output."""

    return {
        f"{prefix}_{key}": value
        for key, value in asdict(result).items()
    }


def _forecast_sigmas(
    history: list[tuple[int, float]],
    position: int,
    historical_sigma: float,
    config: BacktestConfig,
) -> tuple[dict[str, float], float]:
    """Build leakage-safe forecasts from calibrations available before today."""

    recent = [
        (observed_position, sigma)
        for observed_position, sigma in history
        if position - observed_position <= config.max_iv_age_trading_days
    ]
    if not recent:
        return (
            {
                "latest_calibration": np.nan,
                "median_recent_calibrations": np.nan,
                "shrunk_to_historical": np.nan,
            },
            np.nan,
        )
    last_position, last = recent[-1]
    median = float(
        np.median(
            [
                sigma
                for _, sigma in recent[-config.rolling_calibration_count :]
            ]
        )
    )
    shrink = (
        config.shrinkage_weight * median
        + (1 - config.shrinkage_weight) * historical_sigma
    )
    return (
        {
            "latest_calibration": last,
            "median_recent_calibrations": median,
            "shrunk_to_historical": shrink,
        },
        float(position - last_position),
    )


def _required_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Return complete, unique bond/date rows in deterministic order."""

    required = [
        "date",
        "bond_code",
        "stock_code",
        "bond_close",
        "stock_close",
        "K",
        "coupon_used",
        "T",
        "t_conv_used",
        "t_put_used",
        "sigma_used",
        "maturity_redemption_price",
        "theoretical_price",
        "theoretical_price_no_call",
    ]
    missing = [column for column in required if column not in daily.columns]
    if missing:
        raise KeyError(f"daily tracking data missing columns: {missing}")
    result = daily.dropna(subset=required).copy()
    result["date"] = pd.to_datetime(result["date"])
    duplicated = result.duplicated(["bond_code", "date"], keep=False)
    if duplicated.any():
        examples = (
            result.loc[duplicated, ["bond_code", "date"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "daily tracking data contains duplicate bond/date rows: "
            f"{examples}"
        )
    return result.sort_values(["bond_code", "date"])


def _safe_forecast_price(
    row: object,
    terms: ManualModelTerms,
    sigma: float,
    *,
    with_call: bool,
) -> tuple[float, str]:
    """Keep one forecast failure from terminating the whole batch."""

    try:
        value = float(
            _price(
                row,
                terms,
                sigma,
                with_call=with_call,
            )
        )
        if not np.isfinite(value):
            return np.nan, "non_finite_forecast_price"
        return value, ""
    except Exception as exc:
        return np.nan, f"{type(exc).__name__}: {exc}"


def run_backtest(
    daily: pd.DataFrame,
    terms_by_code: dict[str, ManualModelTerms],
    config: BacktestConfig,
) -> pd.DataFrame:
    """Build daily forecasts; current-date IV never prices current date."""

    if config.calibration_stride <= 0:
        raise ValueError("calibration_stride must be positive")
    if config.rolling_calibration_count <= 0:
        raise ValueError("rolling_calibration_count must be positive")
    if not 0 <= config.shrinkage_weight <= 1:
        raise ValueError("shrinkage_weight must be between zero and one")
    if config.max_iv_age_trading_days <= 0:
        raise ValueError("max_iv_age_trading_days must be positive")
    if not 0 < config.validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if not 0 <= config.minimum_forecast_coverage <= 1:
        raise ValueError(
            "minimum_forecast_coverage must be between zero and one"
        )

    valid = _required_daily(daily)
    output_rows: list[dict[str, object]] = []
    for code, group in valid.groupby("bond_code", sort=False):
        code = str(code)
        if code not in terms_by_code:
            raise KeyError(f"terms snapshot missing for {code}")
        terms = terms_by_code[code]
        call_history: list[tuple[int, float]] = []
        no_call_history: list[tuple[int, float]] = []

        for position, row in enumerate(group.itertuples(index=False)):
            historical_sigma = float(row.sigma_used)

            # Forecast first.  Today's calibration, if scheduled, is appended
            # only after these prices are calculated and can affect t+1 onward.
            call_forecast, call_age = _forecast_sigmas(
                call_history,
                position,
                historical_sigma,
                config,
            )
            no_call_forecast, no_call_age = _forecast_sigmas(
                no_call_history,
                position,
                historical_sigma,
                config,
            )
            row_output: dict[str, object] = {
                "date": row.date,
                "bond_code": code,
                "stock_code": row.stock_code,
                "market_price": row.bond_close,
                "historical_sigma": historical_sigma,
                "baseline_price_with_call": row.theoretical_price,
                "baseline_price_no_call": row.theoretical_price_no_call,
                "is_calibration_date": (
                    position % config.calibration_stride == 0
                ),
                "iv_call_age_trading_days": call_age,
                "iv_no_call_age_trading_days": no_call_age,
            }
            current_parity = 100 * float(row.stock_close) / float(row.K)
            if current_parity < terms.debt_equity_blend_low:
                row_output["bond_style"] = "debt"
            elif current_parity > terms.debt_equity_blend_high:
                row_output["bond_style"] = "equity"
            else:
                row_output["bond_style"] = "balanced"

            cache: dict[tuple[bool, float], tuple[float, str]] = {}
            for with_call, forecasts, suffix in (
                (True, call_forecast, "with_call"),
                (False, no_call_forecast, "no_call"),
            ):
                for method, sigma in forecasts.items():
                    row_output[f"sigma_iv_{method}_{suffix}"] = sigma
                    if not np.isfinite(sigma) or sigma <= 0:
                        price = np.nan
                        pricing_error = ""
                    else:
                        cache_key = (with_call, round(float(sigma), 12))
                        if cache_key not in cache:
                            cache[cache_key] = _safe_forecast_price(
                                row,
                                terms,
                                sigma,
                                with_call=with_call,
                            )
                        price, pricing_error = cache[cache_key]
                    row_output[f"price_iv_{method}_{suffix}"] = price
                    row_output[
                        f"price_iv_{method}_{suffix}_error"
                    ] = pricing_error

            if position % config.calibration_stride == 0:
                call_result = _solve_row(
                    row,
                    terms,
                    config,
                    with_call=True,
                )
                no_call_result = _solve_row(
                    row,
                    terms,
                    config,
                    with_call=False,
                )
                row_output.update(_solver_fields("iv_call", call_result))
                row_output.update(
                    _solver_fields("iv_no_call", no_call_result)
                )
                if call_result.success:
                    call_history.append(
                        (position, call_result.implied_sigma)
                    )
                if no_call_result.success:
                    no_call_history.append(
                        (position, no_call_result.implied_sigma)
                    )
            output_rows.append(row_output)

    return pd.DataFrame(output_rows)


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


def _matched_baseline(model: str) -> str:
    """Return the historical-volatility baseline for a forecast column."""

    if model.endswith("_with_call"):
        return "baseline_price_with_call"
    if model.endswith("_no_call"):
        return "baseline_price_no_call"
    raise ValueError(f"cannot match a baseline to model {model!r}")


def select_models(
    summary: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Select on calibration dates, then evaluate untouched validation dates."""

    forecast_models = set(FORECAST_COLUMNS)
    calibration_all = summary.loc[
        summary["segment"].eq("calibration")
    ].copy()
    validation = summary.loc[summary["segment"].eq("validation")].copy()
    candidates = calibration_all.loc[
        calibration_all["model"].isin(forecast_models)
        & calibration_all["mape"].notna()
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates["baseline_model"] = candidates["model"].map(
        _matched_baseline
    )
    baseline_calibration = calibration_all.loc[
        calibration_all["model"].isin(
            {"baseline_price_with_call", "baseline_price_no_call"}
        ),
        [
            "bond_code",
            "engine",
            "model",
            "mape",
            "mean_gap_pct",
            "gap_pct_std",
            "gap_pct_change_mae",
        ],
    ].rename(
        columns={
            "model": "baseline_model",
            "mape": "calibration_baseline_mape",
            "mean_gap_pct": "calibration_baseline_mean_gap_pct",
            "gap_pct_std": "calibration_baseline_gap_pct_std",
            "gap_pct_change_mae": (
                "calibration_baseline_gap_pct_change_mae"
            ),
        }
    )
    candidates = candidates.merge(
        baseline_calibration,
        on=["bond_code", "engine", "baseline_model"],
        how="left",
        validate="many_to_one",
    )
    candidates["calibration_mape_improvement"] = (
        candidates["calibration_baseline_mape"] - candidates["mape"]
    )

    validation_lookup = validation.set_index(
        ["bond_code", "engine", "model"]
    )
    result_rows: list[dict[str, object]] = []
    engines = summary[["bond_code", "engine"]].drop_duplicates()
    for code, engine in engines.itertuples(index=False, name=None):
        bond_candidates = candidates.loc[
            candidates["bond_code"].eq(code)
            & candidates["engine"].eq(engine)
        ]
        eligible = bond_candidates.loc[
            bond_candidates["raw_coverage"].ge(
                config.minimum_forecast_coverage
            )
            & bond_candidates["comparison_coverage"].ge(
                config.minimum_forecast_coverage
            )
        ]
        if eligible.empty:
            style_values = bond_candidates["selection_style"].dropna()
            result_rows.append(
                {
                    "bond_code": code,
                    "engine": engine,
                    "selection_style": (
                        style_values.iloc[0]
                        if not style_values.empty
                        else np.nan
                    ),
                    "selection_status": "no_forecast_meets_coverage",
                    "selection_passed": False,
                    "minimum_forecast_coverage": (
                        config.minimum_forecast_coverage
                    ),
                    "maximum_calibration_comparison_coverage": (
                        bond_candidates["comparison_coverage"].max()
                    ),
                }
            )
            continue

        improving = eligible.loc[
            eligible["calibration_mape_improvement"].gt(0)
        ]
        selection_passed = not improving.empty
        pool = improving if selection_passed else eligible
        best = pool.sort_values(["mape", "model"], kind="stable").iloc[0]
        selected_model = str(best["model"])
        baseline_model = str(best["baseline_model"])
        row: dict[str, object] = {
            "bond_code": code,
            "engine": engine,
            "selection_style": best["selection_style"],
            "selection_status": (
                "selected_on_calibration"
                if selection_passed
                else "best_candidate_does_not_beat_baseline"
            ),
            "selection_passed": selection_passed,
            "minimum_forecast_coverage": (
                config.minimum_forecast_coverage
            ),
            "selected_model": selected_model,
            "baseline_model": baseline_model,
            "calibration_mape": best["mape"],
            "calibration_start_date": best["segment_start_date"],
            "calibration_end_date": best["segment_end_date"],
            "calibration_baseline_mape": best[
                "calibration_baseline_mape"
            ],
            "calibration_baseline_mean_gap_pct": best[
                "calibration_baseline_mean_gap_pct"
            ],
            "calibration_baseline_gap_pct_std": best[
                "calibration_baseline_gap_pct_std"
            ],
            "calibration_baseline_gap_pct_change_mae": best[
                "calibration_baseline_gap_pct_change_mae"
            ],
            "calibration_mape_improvement": best[
                "calibration_mape_improvement"
            ],
            "calibration_relative_mape_improvement": (
                best["calibration_mape_improvement"]
                / best["calibration_baseline_mape"]
                if best["calibration_baseline_mape"] > 0
                else np.nan
            ),
            "calibration_raw_coverage": best["raw_coverage"],
            "calibration_comparison_coverage": best[
                "comparison_coverage"
            ],
            "calibration_mean_gap_pct": best["mean_gap_pct"],
            "calibration_gap_pct_std": best["gap_pct_std"],
            "calibration_gap_pct_change_mae": best[
                "gap_pct_change_mae"
            ],
        }

        selected_key = (code, engine, selected_model)
        baseline_key = (code, engine, baseline_model)
        selected_validation = (
            validation_lookup.loc[selected_key]
            if selected_key in validation_lookup.index
            else None
        )
        baseline_validation = (
            validation_lookup.loc[baseline_key]
            if baseline_key in validation_lookup.index
            else None
        )
        if selected_validation is not None:
            row.update(
                {
                    "validation_mape": selected_validation["mape"],
                    "validation_start_date": selected_validation[
                        "segment_start_date"
                    ],
                    "validation_end_date": selected_validation[
                        "segment_end_date"
                    ],
                    "validation_mean_gap_pct": selected_validation[
                        "mean_gap_pct"
                    ],
                    "validation_gap_pct_std": selected_validation[
                        "gap_pct_std"
                    ],
                    "validation_gap_pct_change_mae": selected_validation[
                        "gap_pct_change_mae"
                    ],
                    "validation_raw_coverage": selected_validation[
                        "raw_coverage"
                    ],
                    "validation_comparison_coverage": selected_validation[
                        "comparison_coverage"
                    ],
                }
            )
        if baseline_validation is not None:
            row.update(
                {
                    "validation_baseline_mape": baseline_validation["mape"],
                    "validation_baseline_mean_gap_pct": (
                        baseline_validation["mean_gap_pct"]
                    ),
                    "validation_baseline_gap_pct_std": baseline_validation[
                        "gap_pct_std"
                    ],
                    "validation_baseline_gap_pct_change_mae": (
                        baseline_validation["gap_pct_change_mae"]
                    ),
                }
            )

        validation_improvement = (
            row.get("validation_baseline_mape", np.nan)
            - row.get("validation_mape", np.nan)
        )
        row["validation_mape_improvement"] = validation_improvement
        validation_baseline_mape = row.get(
            "validation_baseline_mape",
            np.nan,
        )
        row["validation_relative_mape_improvement"] = (
            validation_improvement / validation_baseline_mape
            if np.isfinite(validation_baseline_mape)
            and validation_baseline_mape > 0
            else np.nan
        )
        row["validation_improves_baseline"] = (
            bool(validation_improvement > 0)
            if np.isfinite(validation_improvement)
            else pd.NA
        )
        validation_coverage = row.get(
            "validation_comparison_coverage",
            np.nan,
        )
        row["validation_meets_coverage"] = (
            bool(validation_coverage >= config.minimum_forecast_coverage)
            if np.isfinite(validation_coverage)
            else pd.NA
        )
        row["validation_passed"] = (
            bool(
                validation_improvement > 0
                and validation_coverage >= config.minimum_forecast_coverage
            )
            if np.isfinite(validation_improvement)
            and np.isfinite(validation_coverage)
            else pd.NA
        )
        row["validation_abs_bias_improvement"] = (
            abs(row.get("validation_baseline_mean_gap_pct", np.nan))
            - abs(row.get("validation_mean_gap_pct", np.nan))
        )
        row["validation_gap_std_improvement"] = (
            row.get("validation_baseline_gap_pct_std", np.nan)
            - row.get("validation_gap_pct_std", np.nan)
        )
        row["validation_gap_change_mae_improvement"] = (
            row.get("validation_baseline_gap_pct_change_mae", np.nan)
            - row.get("validation_gap_pct_change_mae", np.nan)
        )
        result_rows.append(row)
    return pd.DataFrame(result_rows)


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


def main() -> None:
    """Run the default offline IV backtest and persist reproducible outputs."""

    config = BacktestConfig()
    daily = pd.read_csv(config.input_csv)
    bond_metadata = pd.read_csv(config.summary_csv)
    terms_by_code = load_terms_snapshot(bond_metadata)
    backtest = run_backtest(daily, terms_by_code, config)
    summary = summarize_backtest(backtest, config)
    selected = select_models(summary, config)
    style_summary = summarize_by_style(summary)
    calibration = calibration_diagnostics(backtest)
    config_values = asdict(config)
    config_values["input_csv"] = str(config.input_csv)
    config_values["summary_csv"] = str(config.summary_csv)
    config_values["output_dir"] = str(config.output_dir)
    config_values["sigma_grid"] = ",".join(
        str(value) for value in config.sigma_grid
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(
        config.output_dir / "implied_volatility_backtest_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        config.output_dir / "implied_volatility_backtest_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.to_csv(
        config.output_dir / "implied_volatility_selected_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    style_summary.to_csv(
        config.output_dir / "implied_volatility_summary_by_style.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibration.to_csv(
        config.output_dir / "implied_volatility_calibration_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([config_values]).to_csv(
        config.output_dir / "implied_volatility_backtest_config.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Calibration diagnostics:")
    print(calibration.to_string(index=False))
    print("\nCalibration-selected models evaluated on validation dates:")
    print(selected.to_string(index=False))
    print(f"\nSaved outputs to {config.output_dir}")


if __name__ == "__main__":
    main()
