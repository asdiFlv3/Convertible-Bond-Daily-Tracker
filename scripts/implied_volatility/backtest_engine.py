"""Leakage-safe daily implied-volatility calibration and forecasting."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from crr_model import ManualModelTerms, price_convertible_with_terms
from implied_volatility.solver import ImpliedVolatilityResult, solve_implied_volatility
from .backtest_config import BacktestConfig


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

