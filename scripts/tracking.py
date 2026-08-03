"""Daily CRR tracking, diagnostics, and plotting.

This module joins vendor-supplied inputs to the pure CRR model.  It does not
call WindPy directly; all Wind-specific behavior is delegated to
``wind_data.py``.  That boundary makes it easier to tell whether an error comes
from data retrieval, daily feature construction, or the pricing model.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crr_model import (
    FACE_VALUE,
    ManualModelTerms,
    parity,
    price_convertible_with_terms,
)
from wind_data import (
    TrackingConfig,
    WindFieldConfig,
    load_wind_daily_inputs,
)


def _as_float(value: object, field_name: str) -> float:
    """Narrow a pandas dynamic scalar to a validated Python ``float``.

    ``DataFrame.itertuples`` exposes each cell to static type checkers as the
    broad pandas ``Scalar`` union, which also contains strings, bytes and date
    types.  Numerical model code should not carry that ambiguity.  Conversion
    is therefore performed once at this boundary and failures retain the
    offending field name and value.

    ``cast(Any, value)`` is intentionally confined to this adapter: pandas
    values are checked by ``float`` at runtime, while the CRR layer receives
    only an actual Python float.
    """

    try:
        return float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field_name}无法转换为float：{value!r}"
        ) from exc


def build_daily_tracking_table(
    tracking: TrackingConfig,
    wind_fields: WindFieldConfig,
    manual_terms: ManualModelTerms,
) -> pd.DataFrame:
    """Build the notebook-equivalent daily valuation table.

    Each row represents information effective on that valuation date:

    * bond and stock closes belong to the same normalized trade date;
    * K is the conversion price already effective on that date;
    * coupon is the rate applicable to that date's accrual year;
    * volatility uses only stock returns up to and including that date;
    * exercise waiting times are recomputed from that valuation date.

    Pricing failures are captured in ``pricing_error`` per row.  A single bad
    date therefore remains inspectable without discarding the whole backtest.
    Data-loading and schema errors still fail fast because continuing with the
    wrong security facts would make every result unreliable.
    """

    daily, metadata = load_wind_daily_inputs(tracking, wind_fields)
    maturity = metadata["maturity_date"]
    conversion_start = metadata["conversion_start_date"]
    put_start = metadata["put_start_date"]

    # Calendar-day ACT/365 is retained from the notebook.  This is a modelling
    # convention; a different day-count basis can be introduced explicitly if
    # required by the bond or chosen risk-free curve.
    daily["T"] = (maturity - daily["date"]).dt.days / 365.0
    daily = daily.loc[daily["T"] > 0].copy()
    if daily.empty:
        raise ValueError(
            "所有共同交易日都被到期日过滤掉了；"
            f"parsed maturity={maturity!r}, "
            f"requested range={tracking.start} to {tracking.end}"
        )

    # At each historical valuation date, exercise is blocked until the actual
    # contractual start date.  Once the date has passed, the waiting time is
    # zero and the corresponding option can be exercised immediately.
    daily["t_conv_used"] = (
        (conversion_start - daily["date"]).dt.days / 365.0
    ).clip(lower=0.0)
    daily["t_put_used"] = (
        (put_start - daily["date"]).dt.days / 365.0
    ).clip(lower=0.0)
    daily["conversion_start_date"] = conversion_start
    daily["put_start_date"] = put_start
    daily["put_status"] = np.where(
        daily["date"] >= put_start,
        "in_put_period",
        "not_yet_in_put_period",
    )

    daily["sigma_used"] = daily[
        f"vol_{tracking.volatility_window}d"
    ]
    daily["parity"] = parity(
        daily["stock_close"],
        FACE_VALUE,
        daily["K"],
    )
    daily["premium_rate"] = (
        daily["bond_close"] / daily["parity"] - 1
    )

    model_prices: list[float] = []
    no_call_prices: list[float] = []
    no_call_diagnostics: list[dict[str, float | bool]] = []
    errors: list[str] = []

    for row in daily.itertuples(index=False):
        # ``itertuples`` is dynamically generated, so Pylance can only describe
        # each attribute as pandas ``Scalar``.  Convert all numerical inputs
        # before using NumPy ufuncs or passing values into the typed CRR model.
        try:
            sigma_used = _as_float(row.sigma_used, "sigma_used")
            conversion_price = _as_float(row.K, "K")
            coupon_used = _as_float(row.coupon_used, "coupon_used")
            stock_close = _as_float(row.stock_close, "stock_close")
            maturity_years = _as_float(row.T, "T")
            conversion_wait_years = _as_float(
                row.t_conv_used,
                "t_conv_used",
            )
            put_wait_years = _as_float(
                row.t_put_used,
                "t_put_used",
            )
        except ValueError as exc:
            model_prices.append(np.nan)
            no_call_prices.append(np.nan)
            no_call_diagnostics.append({})
            errors.append(f"numeric_conversion_error: {exc}")
            continue

        # Early rows may legitimately lack a full rolling-volatility window.
        # They stay in the output with a diagnostic instead of being silently
        # dropped, which makes the usable start date visible to the researcher.
        if not np.isfinite(sigma_used) or sigma_used <= 0:
            model_prices.append(np.nan)
            no_call_prices.append(np.nan)
            no_call_diagnostics.append({})
            errors.append("insufficient_volatility_history")
            continue
        if not np.isfinite(conversion_price) or conversion_price <= 0:
            model_prices.append(np.nan)
            no_call_prices.append(np.nan)
            no_call_diagnostics.append({})
            errors.append("invalid_historical_conversion_price")
            continue
        if not np.isfinite(coupon_used) or coupon_used < 0:
            model_prices.append(np.nan)
            no_call_prices.append(np.nan)
            no_call_diagnostics.append({})
            errors.append("invalid_historical_coupon_rate")
            continue

        try:
            maturity_redemption_price = _as_float(
                metadata["maturity_redemption_price"],
                "maturity_redemption_price",
            )

            def price_with_call_status(
                with_call: bool,
                diagnostics: dict[str, float | bool] | None = None,
            ) -> float:
                """Price this already-normalized row with one call threshold."""

                return price_convertible_with_terms(
                    stock_price=stock_close,
                    conversion_price=conversion_price,
                    sigma=sigma_used,
                    maturity_years=maturity_years,
                    maturity_redemption_price=maturity_redemption_price,
                    coupon_rate=coupon_used,
                    conversion_wait_years=conversion_wait_years,
                    put_wait_years=put_wait_years,
                    terms=manual_terms,
                    with_call=with_call,
                    diagnostics=diagnostics,
                )

            price = price_with_call_status(True)

            # Re-price the same row with an unreachable call threshold.  The
            # difference isolates the impact of the model's call cutoff while
            # holding all market data and other assumptions constant.
            diagnostic_row: dict[str, float | bool] = {}
            price_no_call = price_with_call_status(
                False,
                diagnostic_row,
            )
            model_prices.append(price)
            no_call_prices.append(price_no_call)
            no_call_diagnostics.append(diagnostic_row)
            errors.append("")
        except Exception as exc:
            model_prices.append(np.nan)
            no_call_prices.append(np.nan)
            no_call_diagnostics.append({})
            errors.append(f"{type(exc).__name__}: {exc}")

    daily["theoretical_price"] = model_prices
    daily["theoretical_price_no_call"] = no_call_prices
    diagnostic_frame = pd.DataFrame(no_call_diagnostics, index=daily.index)
    for column in (
        "root_continuation",
        "root_parity",
        "root_continuation_minus_parity",
        "root_conversion_eligible",
        "root_conversion_optimal",
        "conversion_node_share",
        "earliest_conversion_years",
    ):
        daily[f"no_call_{column}"] = diagnostic_frame.get(column)
    daily["call_impact"] = (
        daily["theoretical_price_no_call"]
        - daily["theoretical_price"]
    )
    daily["pricing_error"] = errors

    # A positive gap means the simplified model values the bond above the
    # observed close; it does not by itself imply an immediately tradeable
    # arbitrage because liquidity, taxes, borrow and omitted clauses matter.
    daily["gap"] = daily["theoretical_price"] - daily["bond_close"]
    daily["gap_pct"] = daily["gap"] / daily["bond_close"]
    daily["abs_gap"] = daily["gap"].abs()
    daily["abs_gap_pct"] = daily["gap_pct"].abs()
    daily["market_change"] = daily["bond_close"].diff()
    daily["model_change"] = daily["theoretical_price"].diff()
    daily["rolling_20d_mean_gap_pct"] = (
        daily["gap_pct"].rolling(20).mean()
    )
    daily["rolling_20d_mape"] = (
        daily["abs_gap_pct"].rolling(20).mean()
    )

    # Repeating security metadata makes exported CSV rows self-describing and
    # prevents a result file from becoming detached from its original query.
    daily["bond_code"] = metadata["bond_code"]
    daily["stock_code"] = metadata["stock_code"]
    daily["maturity_date"] = maturity
    daily["maturity_redemption_price"] = metadata[
        "maturity_redemption_price"
    ]
    return daily.reset_index(drop=True)


def summarize_tracking_error(daily: pd.DataFrame) -> pd.Series:
    """Summarize model-minus-market errors over successfully priced dates."""

    valid = daily.dropna(subset=["gap", "gap_pct"])
    if valid.empty:
        error_counts = (
            daily["pricing_error"]
            .fillna("<missing>")
            .replace("", "<no recorded error>")
            .value_counts(dropna=False)
            .to_string()
        )
        sigma_count = int(
            pd.to_numeric(daily["sigma_used"], errors="coerce")
            .gt(0)
            .sum()
        )
        conversion_price_count = int(
            pd.to_numeric(daily["K"], errors="coerce")
            .gt(0)
            .sum()
        )
        coupon_count = int(
            pd.to_numeric(daily["coupon_used"], errors="coerce")
            .ge(0)
            .sum()
        )
        raise ValueError(
            "没有可汇总的有效定价结果。\n"
            f"Total rows: {len(daily)}\n"
            f"Rows with positive sigma: {sigma_count}\n"
            f"Rows with positive conversion price: {conversion_price_count}\n"
            f"Rows with non-negative coupon: {coupon_count}\n"
            "Pricing error counts:\n"
            f"{error_counts}"
        )
    return pd.Series(
        {
            "有效交易日": len(valid),
            "平均价差": valid["gap"].mean(),
            "平均相对价差": valid["gap_pct"].mean(),
            "MAE": valid["abs_gap"].mean(),
            "MAPE": valid["abs_gap_pct"].mean(),
            "RMSE": np.sqrt(np.mean(valid["gap"] ** 2)),
            "相对价差中位数": valid["gap_pct"].median(),
            "相对价差标准差": valid["gap_pct"].std(ddof=1),
            "相对价差5%分位": valid["gap_pct"].quantile(0.05),
            "相对价差95%分位": valid["gap_pct"].quantile(0.95),
            "平均强赎影响": valid["call_impact"].mean(),
        }
    )


def plot_tracking_table(
    daily: pd.DataFrame,
) -> tuple[Any, Any]:
    """Plot market/model prices and relative valuation gaps."""

    valid = daily.dropna(subset=["theoretical_price"]).copy()
    if valid.empty:
        raise ValueError("没有可绘制的有效理论价格")

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
    )
    axes[0].plot(
        valid["date"],
        valid["bond_close"],
        label="Market bond close",
    )
    axes[0].plot(
        valid["date"],
        valid["theoretical_price"],
        label="CRR theoretical price",
    )
    axes[0].plot(
        valid["date"],
        valid["theoretical_price_no_call"],
        "--",
        label="CRR without call cutoff",
    )
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        valid["date"],
        valid["gap_pct"] * 100,
        label="Daily gap %",
    )
    axes[1].plot(
        valid["date"],
        valid["rolling_20d_mean_gap_pct"] * 100,
        label="20D mean gap %",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Gap (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    return figure, axes
