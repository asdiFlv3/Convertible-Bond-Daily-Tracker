"""Offline diagnostics for persistent convertible-bond pricing gaps.

The analysis reuses a saved ``daily_tracking_all.csv`` and therefore does not
open Wind.  It tests inexpensive explanations before a path-dependent reset
model is introduced:

* no-call volatility sensitivity;
* no-call credit-spread sensitivity;
* no-call implied volatility;
* proximity to generic downward-reset trigger bands.

Run from the repository root::

    python scripts/gap_diagnostics.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from crr_model import ManualModelTerms, price_convertible_with_terms
from implied_volatility import ImpliedVolatilityResult, solve_implied_volatility
from implied_volatility_backtest import load_terms_snapshot


@dataclass(frozen=True)
class DiagnosticConfig:
    input_csv: Path = Path(
        "output/classic/batch/all_9/daily_tracking_all.csv"
    )
    summary_csv: Path = Path(
        "output/classic/batch/all_9/comparison_summary.csv"
    )
    output_dir: Path = Path("output/diagnostics/gap")
    volatility_multipliers: tuple[float, ...] = (
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
    )
    credit_spreads: tuple[float, ...] = (0.0, 0.005, 0.01, 0.015, 0.02)
    sensitivity_stride: int = 20
    implied_volatility_grid: tuple[float, ...] = (
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
    implied_volatility_iterations: int = 24
    implied_volatility_price_tolerance: float = 0.01
    implied_volatility_stride: int = 20
    reset_lookback_days: int = 30
    reset_price_ratios: tuple[float, ...] = (0.80, 0.85, 0.90)


def _no_call_price(
    row: object,
    terms: ManualModelTerms,
    *,
    sigma: float,
    diagnostics: dict[str, float | bool] | None = None,
) -> float:
    """Reprice one saved daily row with no reachable call threshold."""

    return price_convertible_with_terms(
        stock_price=float(getattr(row, "stock_close")),
        conversion_price=float(getattr(row, "K")),
        sigma=sigma,
        maturity_years=float(getattr(row, "T")),
        maturity_redemption_price=float(
            getattr(row, "maturity_redemption_price")
        ),
        coupon_rate=float(getattr(row, "coupon_used")),
        conversion_wait_years=float(getattr(row, "t_conv_used")),
        put_wait_years=float(getattr(row, "t_put_used")),
        terms=terms,
        with_call=False,
        diagnostics=diagnostics,
    )


def _implied_no_call_volatility(
    row: object,
    terms: ManualModelTerms,
    config: DiagnosticConfig,
) -> ImpliedVolatilityResult:
    """Invert the no-call model and preserve numerical failure diagnostics."""

    return solve_implied_volatility(
        lambda sigma: _no_call_price(row, terms, sigma=sigma),
        float(getattr(row, "bond_close")),
        sigma_grid=config.implied_volatility_grid,
        max_iterations=config.implied_volatility_iterations,
        price_tolerance=config.implied_volatility_price_tolerance,
    )


def _valid_daily(daily: pd.DataFrame) -> pd.DataFrame:
    required = [
        "bond_code",
        "date",
        "bond_close",
        "stock_close",
        "K",
        "sigma_used",
        "coupon_used",
        "T",
        "t_conv_used",
        "t_put_used",
        "maturity_redemption_price",
    ]
    missing = [column for column in required if column not in daily.columns]
    if missing:
        raise KeyError(f"daily tracking data missing columns: {missing}")
    result = daily.dropna(subset=required).copy()
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values(["bond_code", "date"])


def run_diagnostics(
    daily: pd.DataFrame,
    terms_by_code: dict[str, ManualModelTerms],
    config: DiagnosticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return sensitivity summary, implied-volatility rows and reset metrics."""

    valid = _valid_daily(daily)
    sensitivity_rows: list[dict[str, object]] = []
    implied_rows: list[dict[str, object]] = []
    structure_rows: list[dict[str, object]] = []

    for code, group in valid.groupby("bond_code", sort=False):
        code = str(code)
        if code not in terms_by_code:
            raise KeyError(f"terms snapshot missing for {code}")
        base = terms_by_code[code]
        records = list(group.itertuples(index=False))
        sensitivity_records = records[:: config.sensitivity_stride]
        sensitivity_market = group["bond_close"].to_numpy(dtype=float)[
            :: config.sensitivity_stride
        ]

        for row in sensitivity_records:
            structure: dict[str, float | bool] = {}
            price = _no_call_price(
                row,
                base,
                sigma=float(row.sigma_used),
                diagnostics=structure,
            )
            structure_rows.append(
                {
                    "date": row.date,
                    "bond_code": code,
                    "stock_code": row.stock_code,
                    "market_price": row.bond_close,
                    "no_call_price": price,
                    **structure,
                }
            )

        for multiplier in config.volatility_multipliers:
            prices = np.array(
                [
                    _no_call_price(
                        row,
                        base,
                        sigma=float(row.sigma_used) * multiplier,
                    )
                    for row in sensitivity_records
                ]
            )
            sensitivity_rows.append(
                {
                    "bond_code": code,
                    "parameter": "volatility_multiplier",
                    "value": multiplier,
                    "mape": np.mean(
                        np.abs(prices - sensitivity_market)
                        / sensitivity_market
                    ),
                    "mean_gap_pct": np.mean(
                        (prices - sensitivity_market) / sensitivity_market
                    ),
                }
            )

        for spread in config.credit_spreads:
            terms = replace(base, credit_spread=spread)
            prices = np.array(
                [
                    _no_call_price(
                        row,
                        terms,
                        sigma=float(row.sigma_used),
                    )
                    for row in sensitivity_records
                ]
            )
            sensitivity_rows.append(
                {
                    "bond_code": code,
                    "parameter": "credit_spread",
                    "value": spread,
                    "mape": np.mean(
                        np.abs(prices - sensitivity_market)
                        / sensitivity_market
                    ),
                    "mean_gap_pct": np.mean(
                        (prices - sensitivity_market) / sensitivity_market
                    ),
                }
            )

        # Implied-volatility inversion is the only iterative diagnostic.  A
        # weekly-like stride keeps the offline run light while preserving the
        # full daily sample for the cheaper sensitivity grids.
        for row in records[:: config.implied_volatility_stride]:
            implied_result = _implied_no_call_volatility(row, base, config)
            implied = implied_result.implied_sigma
            historical = float(row.sigma_used)
            implied_rows.append(
                {
                    "date": row.date,
                    "bond_code": code,
                    "stock_code": row.stock_code,
                    "historical_sigma": historical,
                    "implied_no_call_sigma": implied,
                    "implied_success": implied_result.success,
                    "implied_failure_reason": (
                        "" if implied_result.success else implied_result.reason
                    ),
                    "implied_price_residual": implied_result.residual,
                    "implied_to_historical": (
                        implied / historical if np.isfinite(implied) else np.nan
                    ),
                }
            )

    reset = valid[
        ["date", "bond_code", "stock_code", "stock_close", "K", "gap_pct"]
    ].copy()
    reset["stock_to_conversion_price"] = reset["stock_close"] / reset["K"]
    for ratio in config.reset_price_ratios:
        label = int(round(ratio * 100))
        hit = reset["stock_close"].lt(ratio * reset["K"]).astype(int)
        reset[f"reset_hit_count_{label}_{config.reset_lookback_days}d"] = (
            hit.groupby(reset["bond_code"])
            .rolling(config.reset_lookback_days, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    return (
        pd.DataFrame(sensitivity_rows),
        pd.DataFrame(implied_rows),
        reset,
        pd.DataFrame(structure_rows),
    )


def summarize_reset_proximity(reset: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether model underpricing grows near reset trigger bands."""

    rows: list[dict[str, object]] = []
    count_columns = [
        column
        for column in reset.columns
        if column.startswith("reset_hit_count_")
    ]
    for code, group in reset.groupby("bond_code", sort=False):
        underpricing = -group["gap_pct"]
        for column in count_columns:
            count = group[column]
            correlation = (
                count.corr(underpricing)
                if count.nunique() > 1 and underpricing.nunique() > 1
                else np.nan
            )
            rows.append(
                {
                    "bond_code": code,
                    "trigger_metric": column,
                    "days_with_any_hit_share": count.gt(0).mean(),
                    "maximum_hit_count": count.max(),
                    "underpricing_correlation": correlation,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    config = DiagnosticConfig()
    daily = pd.read_csv(config.input_csv)
    summary = pd.read_csv(config.summary_csv)
    terms_by_code = load_terms_snapshot(summary)
    sensitivity, implied, reset, structure = run_diagnostics(
        daily,
        terms_by_code,
        config,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(
        config.output_dir / "parameter_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    implied.to_csv(
        config.output_dir / "implied_no_call_volatility.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reset.to_csv(
        config.output_dir / "reset_trigger_proximity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summarize_reset_proximity(reset).to_csv(
        config.output_dir / "reset_trigger_proximity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    structure.to_csv(
        config.output_dir / "no_call_conversion_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    implied_summary = (
        implied.groupby("bond_code", as_index=False)
        .agg(
            valid_days=("historical_sigma", "size"),
            implied_solution_days=("implied_no_call_sigma", "count"),
            median_implied_sigma=("implied_no_call_sigma", "median"),
            median_implied_to_historical=("implied_to_historical", "median"),
        )
    )
    implied_summary["solution_share"] = (
        implied_summary["implied_solution_days"]
        / implied_summary["valid_days"]
    )
    implied_summary.to_csv(
        config.output_dir / "implied_no_call_volatility_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(implied_summary.to_string(index=False))
    print(f"Saved diagnostics to {config.output_dir}")


if __name__ == "__main__":
    main()
