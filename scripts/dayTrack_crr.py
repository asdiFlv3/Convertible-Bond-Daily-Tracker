"""Command-line entry point for Wind-backed daily CRR tracking.

This file is intentionally small.  It answers the run-specific questions:

* Which convertible bond and date range should be analysed?
* Which generated Wind fields should be used?
* Which modelling assumptions and manually checked clauses apply?
* Where should the table and chart be saved?

Implementation details live in:

* ``wind_data.py`` for Wind connection, retrieval and normalization;
* ``crr_model.py`` for the pure numerical model;
* ``tracking.py`` for daily pricing, diagnostics and plotting.

Run from the project directory after opening and logging into Wind::

    uv run python scripts/dayTrack_crr.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from crr_model import ManualModelTerms
from tracking import (
    build_daily_tracking_table,
    plot_tracking_table,
    summarize_tracking_error,
)
from wind_data import (
    TrackingConfig,
    WindFieldConfig,
    assert_wind_fields_configured,
    start_wind,
)


def main() -> None:
    """Configure one tracking run, retrieve Wind data, and save results."""

    # TODO(Wind code generator): replace every placeholder in this object with
    # an exact generated Wind field name.  Do not replace the missing fields
    # with manual values: these security facts must continue to come from Wind.
    #
    # Keep the generated commands and field definitions in the project so that
    # the selected units and data vintages remain auditable.
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
        # ``couponrate3`` was verified to change with the historical accrual
        # year, unlike ``couponrate2``, which repeated the current coupon across
        # the requested history.
        historical_coupon_rate="couponrate3",
    )

    # Run-level settings.  The stock code is intentionally absent: it is read
    # from Wind using ``underlying_code`` to prevent bond/stock mismatches.
    tracking = TrackingConfig(
        bond_code="123117.SZ",
        start="2025-07-21",
        end="2026-07-21",
        volatility_window=60,
    )

    # These are model choices and simplified prospectus clauses, not fallback
    # values for missing Wind data.  Validate the three clause parameters for
    # each bond before treating the output as investable research.
    manual_terms = ManualModelTerms(
        risk_free_rate=0.02,
        dividend_yield=0.0,
        credit_spread=0.01,
        debt_equity_blend_low=70.0,
        debt_equity_blend_high=130.0,
        call_parity_trigger=130.0,
        put_parity_trigger=70.0,
        put_price=103.0,
        tree_steps=200,
    )

    # Validate placeholders before starting Wind so configuration errors fail
    # immediately without making an unnecessary terminal request.
    assert_wind_fields_configured(wind_fields)
    start_wind()

    daily = build_daily_tracking_table(
        tracking,
        wind_fields,
        manual_terms,
    )

    tracking.output_dir.mkdir(parents=True, exist_ok=True)
    stem = tracking.bond_code.replace(".", "_")
    csv_path = tracking.output_dir / f"daily_tracking_{stem}.csv"
    chart_path = tracking.output_dir / f"daily_tracking_{stem}.png"

    # UTF-8 with BOM keeps Chinese column names readable when the CSV is opened
    # directly in common Windows versions of Excel.  Save before summarizing so
    # row-level ``pricing_error`` diagnostics survive even when every valuation
    # failed and the aggregate report cannot be produced.
    daily.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = summarize_tracking_error(daily)
    figure, _ = plot_tracking_table(daily)
    figure.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(daily.tail(20).to_string(index=False))
    print("\nTracking summary")
    print(summary.to_string())
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
