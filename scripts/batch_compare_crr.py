"""Batch Wind-backed CRR tracking and cross-bond comparison.

This module reuses the existing single-bond pipeline instead of duplicating
market-data or pricing logic.  One Wind session is opened for the full batch,
each bond is isolated behind its own error boundary, and successful results are
combined into long-form daily data and one-row-per-bond comparison summaries.

Run from the project directory::

    uv run python scripts/batch_compare_crr.py

The Wind terminal must already be open and logged in.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from batch_plots import save_batch_charts
from crr_model import ManualModelTerms
from tracking import build_daily_tracking_table
from wind_data import (
    TrackingConfig,
    WindFieldConfig,
    assert_wind_fields_configured,
    start_wind,
)


@dataclass(frozen=True)
class BondSpec:
    """Per-bond inputs that Wind does not currently provide reliably.

    ``bond_code`` must include its Wind market suffix.  Contractual trigger and
    put-price values should be checked against each bond's prospectus.

    Optional credit-spread and dividend-yield overrides are useful when one
    bond needs a different modelling assumption.  ``None`` means use the
    shared value from ``base_terms``.
    """

    bond_code: str
    call_parity_trigger: float
    put_parity_trigger: float
    put_price: float
    credit_spread: float | None = None
    dividend_yield: float | None = None


@dataclass(frozen=True)
class BatchConfig:
    """Date, volatility, and output settings shared by the whole batch."""

    start: str
    end: str
    volatility_window: int = 60
    annualization_days: int = 252
    price_adjustment: str = "F"
    output_dir: Path = Path("output/batch")
    save_individual_tables: bool = True
    max_bonds_in_history_chart: int = 10


@dataclass
class BatchResult:
    """All in-memory outputs from one batch run."""

    daily: pd.DataFrame
    summary: pd.DataFrame
    common_date_summary: pd.DataFrame
    style_summary: pd.DataFrame
    failures: pd.DataFrame


# The builder is injectable so batch orchestration can be tested without a live
# Wind session.  Production callers should leave this at its default.
DailyBuilder = Callable[
    [TrackingConfig, WindFieldConfig, ManualModelTerms],
    pd.DataFrame,
]


def _validate_bond_specs(bond_specs: list[BondSpec]) -> None:
    """Reject ambiguous or invalid batch inputs before opening Wind."""

    if not bond_specs:
        raise ValueError("bond_specs不能为空")

    normalized_codes = [
        spec.bond_code.strip().upper()
        for spec in bond_specs
    ]
    duplicated = (
        pd.Series(normalized_codes)
        .loc[lambda values: values.duplicated(keep=False)]
        .unique()
        .tolist()
    )
    if duplicated:
        raise ValueError(f"bond_specs包含重复代码：{duplicated}")

    for spec in bond_specs:
        if "." not in spec.bond_code:
            raise ValueError(
                f"转债代码必须包含Wind市场后缀：{spec.bond_code!r}"
            )
        if spec.call_parity_trigger <= 0:
            raise ValueError(
                f"{spec.bond_code} call_parity_trigger必须为正数"
            )
        if spec.put_parity_trigger <= 0:
            raise ValueError(
                f"{spec.bond_code} put_parity_trigger必须为正数"
            )
        if spec.put_price <= 0:
            raise ValueError(f"{spec.bond_code} put_price必须为正数")


def _terms_for_bond(
    base_terms: ManualModelTerms,
    spec: BondSpec,
) -> ManualModelTerms:
    """Apply one bond's clauses and optional assumption overrides."""

    changes: dict[str, float] = {
        "call_parity_trigger": spec.call_parity_trigger,
        "put_parity_trigger": spec.put_parity_trigger,
        "put_price": spec.put_price,
    }
    if spec.credit_spread is not None:
        changes["credit_spread"] = spec.credit_spread
    if spec.dividend_yield is not None:
        changes["dividend_yield"] = spec.dividend_yield
    return replace(base_terms, **changes)


def _tracking_for_bond(
    batch: BatchConfig,
    bond_code: str,
) -> TrackingConfig:
    """Translate shared batch settings into the existing single-bond config."""

    return TrackingConfig(
        bond_code=bond_code.strip().upper(),
        start=batch.start,
        end=batch.end,
        volatility_window=batch.volatility_window,
        annualization_days=batch.annualization_days,
        price_adjustment=batch.price_adjustment,
        output_dir=batch.output_dir,
    )


def _summary_row(
    daily: pd.DataFrame,
    spec: BondSpec,
    terms: ManualModelTerms,
) -> dict[str, object]:
    """Build comparison metrics for one bond over the supplied dates."""

    valid = daily.dropna(
        subset=["gap", "gap_pct", "theoretical_price"],
    ).copy()
    if valid.empty:
        raise ValueError("没有有效理论价格可用于单券汇总")

    valid = valid.sort_values("date")
    latest = valid.iloc[-1]
    latest_parity = float(latest["parity"])
    if latest_parity < terms.debt_equity_blend_low:
        bond_style = "debt"
    elif latest_parity > terms.debt_equity_blend_high:
        bond_style = "equity"
    else:
        bond_style = "balanced"
    return {
        "bond_code": spec.bond_code.strip().upper(),
        "stock_code": latest.get("stock_code"),
        "bond_style": bond_style,
        "valid_days": int(len(valid)),
        "first_valid_date": valid["date"].iloc[0],
        "last_valid_date": latest["date"],
        "maturity_date": latest.get("maturity_date"),
        "remaining_years": latest.get("T"),
        "mean_gap": valid["gap"].mean(),
        "mean_gap_pct": valid["gap_pct"].mean(),
        "mae": valid["abs_gap"].mean(),
        "mape": valid["abs_gap_pct"].mean(),
        "rmse": float(np.sqrt(np.mean(valid["gap"] ** 2))),
        "median_gap_pct": valid["gap_pct"].median(),
        "gap_pct_std": valid["gap_pct"].std(ddof=1),
        "gap_pct_05": valid["gap_pct"].quantile(0.05),
        "gap_pct_95": valid["gap_pct"].quantile(0.95),
        "mean_call_impact": valid["call_impact"].mean(),
        "latest_market_price": latest["bond_close"],
        "latest_model_price": latest["theoretical_price"],
        "latest_gap": latest["gap"],
        "latest_gap_pct": latest["gap_pct"],
        "latest_parity": latest["parity"],
        "latest_premium_rate": latest["premium_rate"],
        "latest_sigma": latest["sigma_used"],
        "latest_conversion_price": latest["K"],
        "call_parity_trigger": terms.call_parity_trigger,
        "put_parity_trigger": terms.put_parity_trigger,
        "put_price": terms.put_price,
        "risk_free_rate": terms.risk_free_rate,
        "dividend_yield": terms.dividend_yield,
        "credit_spread": terms.credit_spread,
        "tree_steps": terms.tree_steps,
    }


def _add_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """Add separate rankings without inventing an opaque composite score."""

    result = summary.copy()
    if result.empty:
        return result

    # Lower MAPE and lower gap volatility indicate closer/more stable model fit.
    # Higher latest gap means model value exceeds market price by more.  This is
    # a diagnostic ranking, not an executable trading recommendation.
    result["rank_mape"] = result["mape"].rank(
        method="min",
        ascending=True,
    )
    result["rank_latest_gap_pct"] = result["latest_gap_pct"].rank(
        method="min",
        ascending=False,
    )
    result["rank_abs_latest_gap_pct"] = (
        result["latest_gap_pct"].abs().rank(
            method="min",
            ascending=True,
        )
    )
    result["rank_gap_stability"] = result["gap_pct_std"].rank(
        method="min",
        ascending=True,
    )
    return result.sort_values(
        ["rank_latest_gap_pct", "rank_mape"],
        kind="stable",
    ).reset_index(drop=True)


def _style_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Evaluate percentage model errors within bond-style groups."""

    if summary.empty:
        return pd.DataFrame()
    return (
        summary.groupby("bond_style", sort=False, dropna=False)
        .agg(
            bond_count=("bond_code", "nunique"),
            mean_mape=("mape", "mean"),
            median_mape=("mape", "median"),
            mean_gap_pct=("mean_gap_pct", "mean"),
            median_gap_pct=("median_gap_pct", "median"),
            mean_latest_gap_pct=("latest_gap_pct", "mean"),
            mean_gap_pct_std=("gap_pct_std", "mean"),
        )
        .reset_index()
    )


def _failure_row(code: str, exc: Exception) -> dict[str, object]:
    """Preserve structured Wind diagnostics in failures.csv."""

    return {
        "bond_code": code,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "wind_request_type": getattr(exc, "request_type", ""),
        "wind_request_code": getattr(exc, "wind_code", ""),
        "wind_fields": getattr(exc, "wind_fields", ""),
        "wind_error_code": getattr(exc, "wind_error_code", ""),
        "response_columns": getattr(exc, "response_columns", ""),
        "response_shape": getattr(exc, "response_shape", ""),
        "response_preview": getattr(exc, "response_preview", ""),
    }


def _common_date_summary(
    daily: pd.DataFrame,
    successful_specs: list[BondSpec],
    terms_by_code: dict[str, ManualModelTerms],
) -> pd.DataFrame:
    """Summarize every bond over the dates valid for all successful bonds."""

    if daily.empty or not successful_specs:
        return pd.DataFrame()

    successful_codes = [
        spec.bond_code.strip().upper()
        for spec in successful_specs
    ]
    valid = daily.dropna(
        subset=["gap", "gap_pct", "theoretical_price"],
    )
    coverage = valid.groupby("date")["bond_code"].nunique()
    common_dates = coverage.loc[
        coverage.eq(len(successful_codes))
    ].index
    if len(common_dates) == 0:
        return pd.DataFrame()

    common = valid.loc[valid["date"].isin(common_dates)]
    rows: list[dict[str, object]] = []
    spec_by_code = {
        spec.bond_code.strip().upper(): spec
        for spec in successful_specs
    }
    for code in successful_codes:
        bond_daily = common.loc[common["bond_code"].eq(code)]
        rows.append(
            _summary_row(
                bond_daily,
                spec_by_code[code],
                terms_by_code[code],
            )
        )
    result = _add_ranks(pd.DataFrame(rows))
    result.insert(2, "common_date_count", len(common_dates))
    return result


def run_batch_comparison(
    bond_specs: list[BondSpec],
    batch_config: BatchConfig,
    wind_fields: WindFieldConfig,
    base_terms: ManualModelTerms,
    *,
    start_session: bool = True,
    daily_builder: DailyBuilder = build_daily_tracking_table,
) -> BatchResult:
    """Run N bonds, retain successful outputs, and isolate failures.

    Parameters
    ----------
    start_session:
        Leave ``True`` in production.  Tests can pass ``False`` together with a
        fake ``daily_builder`` to exercise orchestration without Wind.
    daily_builder:
        Dependency-injection seam for tests; normal callers should not change
        it.
    """

    _validate_bond_specs(bond_specs)
    assert_wind_fields_configured(wind_fields)
    if start_session:
        start_wind()

    daily_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    successful_specs: list[BondSpec] = []
    terms_by_code: dict[str, ManualModelTerms] = {}

    for spec in bond_specs:
        code = spec.bond_code.strip().upper()
        tracking = _tracking_for_bond(batch_config, code)
        terms = _terms_for_bond(base_terms, spec)
        try:
            daily = daily_builder(tracking, wind_fields, terms).copy()
            if daily.empty:
                raise ValueError("日度跟踪结果为空")
            # Use the requested code as the canonical batch key even if a
            # vendor-returned column has inconsistent casing.
            daily["bond_code"] = code
            summary_rows.append(_summary_row(daily, spec, terms))
            daily_tables.append(daily)
            successful_specs.append(spec)
            terms_by_code[code] = terms
        except Exception as exc:
            failures.append(_failure_row(code, exc))

    daily_all = (
        pd.concat(daily_tables, ignore_index=True)
        if daily_tables
        else pd.DataFrame()
    )
    if not daily_all.empty:
        daily_all = daily_all.sort_values(
            ["bond_code", "date"],
            kind="stable",
        ).reset_index(drop=True)

    summary = _add_ranks(pd.DataFrame(summary_rows))
    common_summary = _common_date_summary(
        daily_all,
        successful_specs,
        terms_by_code,
    )
    style_summary = _style_summary(summary)
    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "bond_code",
            "error_type",
            "error_message",
            "wind_request_type",
            "wind_request_code",
            "wind_fields",
            "wind_error_code",
            "response_columns",
            "response_shape",
            "response_preview",
        ],
    )
    return BatchResult(
        daily=daily_all,
        summary=summary,
        common_date_summary=common_summary,
        style_summary=style_summary,
        failures=failure_frame,
    )


def save_batch_result(
    result: BatchResult,
    batch_config: BatchConfig,
) -> None:
    """Write tables and comparison charts to ``batch_config.output_dir``."""

    output_dir = batch_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result.daily.to_csv(
        output_dir / "daily_tracking_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.summary.to_csv(
        output_dir / "comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.common_date_summary.to_csv(
        output_dir / "comparison_summary_common_dates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.style_summary.to_csv(
        output_dir / "comparison_summary_by_style.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.failures.to_csv(
        output_dir / "failures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if batch_config.save_individual_tables and not result.daily.empty:
        individual_dir = output_dir / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        for code, group in result.daily.groupby(
            "bond_code",
            sort=False,
        ):
            safe_code = str(code).replace(".", "_")
            group.to_csv(
                individual_dir / f"daily_tracking_{safe_code}.csv",
                index=False,
                encoding="utf-8-sig",
            )

    save_batch_charts(
        result.daily,
        result.summary,
        output_dir,
        batch_config.max_bonds_in_history_chart,
    )


def main() -> None:
    """Configure and execute one example batch."""

    wind_fields = WindFieldConfig(
        underlying_code="bclc",
        maturity_date="maturitdate",
        conversion_start_date=(
            "clause_conversion_2_swapsharestartdate"
        ),
        put_start_date=(
            "clause_putoption_conditionalputbackstartenddate"
        ),
        maturity_redemption_price="maturitycallprice",
        historical_conversion_price="convprice",
        historical_coupon_rate="couponrate3",
    )

    # Add or remove BondSpec entries to change N.  The three clause values are
    # intentionally explicit per bond so an unnoticed default cannot be applied
    # to a prospectus with different terms.
    bond_specs = [
        BondSpec(
            bond_code="123117.SZ",
            call_parity_trigger=130.0,
            put_parity_trigger=70.0,
            put_price=103.0,
        ),
         BondSpec(
             bond_code="123257.SZ",
             call_parity_trigger=130.0,
             put_parity_trigger=70.0,
             put_price=108.0,
         ),
        BondSpec(
             bond_code="118058.SH",
             call_parity_trigger=130.0,
             put_parity_trigger=70.0,
             put_price=110.0,
         ),
        #BondSpec(
             #bond_code="123258.SZ",
             #call_parity_trigger=130.0,
             #put_parity_trigger=70.0,
             #put_price=113.0,
         #),

        #BondSpec(
            #bond_code="111022.SH",
            #call_parity_trigger=130.0,
            #put_parity_trigger=70.0,
            #put_price=113.0,
        #),
        BondSpec(
            bond_code="123255.SZ",
            call_parity_trigger=130.0,
            put_parity_trigger=70.0,
            put_price=110.0,
        ),
        #BondSpec(
            #bond_code="118056.SH",
            #call_parity_trigger=130.0,
            #put_parity_trigger=70.0,
            #put_price=112.0,
        #),
        #BondSpec(
            #bond_code="123263.SZ",
            #call_parity_trigger=130.0,
            #put_parity_trigger=70.0,
            #put_price=110.0,
        #),
        #BondSpec(
        #    bond_code="118062.SH",
        #    call_parity_trigger=130.0,
        #    put_parity_trigger=70.0,
        #    put_price=112.0,
        #),
    ]

    batch_config = BatchConfig(
        start="2025-07-21",
        end="2026-07-21",
        volatility_window=60,
        output_dir=Path("output/batch"),
    )
    base_terms = ManualModelTerms(
        risk_free_rate=0.02,
        dividend_yield=0.0,
        credit_spread=0.01,
        debt_equity_blend_low=70.0,
        debt_equity_blend_high=130.0,
        # These required dataclass values are replaced by every BondSpec.
        call_parity_trigger=130.0,
        put_parity_trigger=70.0,
        put_price=103.0,
        tree_steps=200,
    )

    result = run_batch_comparison(
        bond_specs,
        batch_config,
        wind_fields,
        base_terms,
    )
    save_batch_result(result, batch_config)

    print("Batch comparison summary:")
    if result.summary.empty:
        print("No bonds completed successfully.")
    else:
        columns = [
            "bond_code",
            "valid_days",
            "mape",
            "latest_market_price",
            "latest_model_price",
            "latest_gap_pct",
            "rank_latest_gap_pct",
        ]
        print(result.summary[columns].to_string(index=False))

    print("\nFailures:")
    if result.failures.empty:
        print("None")
    else:
        print(result.failures.to_string(index=False))

    print(f"\nSaved batch outputs to: {batch_config.output_dir}")


if __name__ == "__main__":
    main()
