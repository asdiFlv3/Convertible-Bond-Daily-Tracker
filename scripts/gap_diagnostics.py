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

from crr_model import FACE_VALUE, ManualModelTerms, crr_convertible_basic


@dataclass(frozen=True)
class DiagnosticConfig:
    input_csv: Path = Path(
        "output/classic/batch/all_9/daily_tracking_all.csv"
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
    implied_volatility_lower: float = 0.01
    implied_volatility_upper: float = 3.0
    implied_volatility_iterations: int = 15
    implied_volatility_stride: int = 20
    reset_lookback_days: int = 30
    reset_price_ratios: tuple[float, ...] = (0.80, 0.85, 0.90)


DEFAULT_PUT_PRICES = {
    "123117.SZ": 103.0,
    "123257.SZ": 108.0,
    "118058.SH": 110.0,
    "123258.SZ": 113.0,
    "111022.SH": 113.0,
    "123255.SZ": 110.0,
    "118056.SH": 112.0,
    "123263.SZ": 110.0,
    "118062.SH": 112.0,
}


def _base_terms(code: str) -> ManualModelTerms:
    """Return the assumptions used by the current batch example."""

    return ManualModelTerms(
        risk_free_rate=0.02,
        dividend_yield=0.0,
        credit_spread=0.01,
        debt_equity_blend_low=70.0,
        debt_equity_blend_high=130.0,
        call_parity_trigger=130.0,
        put_parity_trigger=70.0,
        put_price=DEFAULT_PUT_PRICES[code],
        tree_steps=200,
    )


def _no_call_price(
    row: object,
    terms: ManualModelTerms,
    *,
    sigma: float,
    diagnostics: dict[str, float | bool] | None = None,
) -> float:
    """Reprice one saved daily row with no reachable call threshold."""

    return crr_convertible_basic(
        stock_price=float(getattr(row, "stock_close")),
        conversion_price=float(getattr(row, "K")),
        sigma=sigma,
        risk_free_rate=terms.risk_free_rate,
        maturity_years=float(getattr(row, "T")),
        dividend_yield=terms.dividend_yield,
        maturity_redemption_price=float(
            getattr(row, "maturity_redemption_price")
        ),
        coupon_rate=float(getattr(row, "coupon_used")),
        credit_spread=terms.credit_spread,
        blend_low=terms.debt_equity_blend_low,
        blend_high=terms.debt_equity_blend_high,
        conversion_wait_years=float(getattr(row, "t_conv_used")),
        put_wait_years=float(getattr(row, "t_put_used")),
        steps=terms.tree_steps,
        call_parity_trigger=float("inf"),
        put_parity_trigger=terms.put_parity_trigger,
        face_value=FACE_VALUE,
        put_price=terms.put_price,
        diagnostics=diagnostics,
    )


def _implied_no_call_volatility(
    row: object,
    terms: ManualModelTerms,
    config: DiagnosticConfig,
) -> float:
    """Invert the no-call model, returning NaN when price is out of range."""

    target = float(getattr(row, "bond_close"))
    low = config.implied_volatility_lower
    high = config.implied_volatility_upper
    low_price = _no_call_price(row, terms, sigma=low)
    high_price = _no_call_price(row, terms, sigma=high)
    if not low_price <= target <= high_price:
        return np.nan
    for _ in range(config.implied_volatility_iterations):
        middle = (low + high) / 2
        middle_price = _no_call_price(row, terms, sigma=middle)
        if middle_price < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


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
    config: DiagnosticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return sensitivity summary, implied-volatility rows and reset metrics."""

    valid = _valid_daily(daily)
    sensitivity_rows: list[dict[str, object]] = []
    implied_rows: list[dict[str, object]] = []
    structure_rows: list[dict[str, object]] = []

    for code, group in valid.groupby("bond_code", sort=False):
        code = str(code)
        if code not in DEFAULT_PUT_PRICES:
            raise KeyError(f"put price is not configured for {code}")
        base = _base_terms(code)
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
            implied = _implied_no_call_volatility(row, base, config)
            historical = float(row.sigma_used)
            implied_rows.append(
                {
                    "date": row.date,
                    "bond_code": code,
                    "stock_code": row.stock_code,
                    "historical_sigma": historical,
                    "implied_no_call_sigma": implied,
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
    sensitivity, implied, reset, structure = run_diagnostics(daily, config)
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
