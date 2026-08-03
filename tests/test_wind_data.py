"""Tests for point-in-time Wind data alignment and price-adjustment roles."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from wind_data import TrackingConfig, WindFieldConfig, load_wind_daily_inputs


class WindDailyInputsTests(unittest.TestCase):
    """Verify that valuation and volatility use correctly adjusted series."""

    def test_unadjusted_stock_prices_are_used_for_valuation(self) -> None:
        all_dates = pd.bdate_range("2023-11-01", "2024-01-31")
        valuation_dates = all_dates[all_dates >= pd.Timestamp("2024-01-02")]
        raw_stock = pd.DataFrame(
            {"CLOSE": np.linspace(10.0, 12.0, len(all_dates))},
            index=all_dates,
        )
        positions = np.arange(len(all_dates), dtype=float)
        adjusted_values = 100.0 * np.exp(
            positions * 0.001 + np.sin(positions / 3.0) * 0.01
        )
        adjusted_stock = pd.DataFrame(
            {"CLOSE": adjusted_values},
            index=all_dates,
        )
        bond_market = pd.DataFrame(
            {"CLOSE": np.full(len(valuation_dates), 110.0)},
            index=valuation_dates,
        )
        terms = pd.DataFrame(
            {
                "CONVERSION_PRICE": np.full(len(all_dates), 20.0),
                "COUPON_RATE": np.full(len(all_dates), 1.5),
            },
            index=all_dates,
        )
        metadata = {
            "bond_code": "123456.SZ",
            "stock_code": "000001.SZ",
            "maturity_date": pd.Timestamp("2027-01-01"),
            "conversion_start_date": pd.Timestamp("2023-01-01"),
            "put_start_date": pd.Timestamp("2026-01-01"),
            "maturity_redemption_price": 110.0,
        }
        tracking = TrackingConfig(
            bond_code="123456.SZ",
            start="2024-01-02",
            end="2024-01-31",
            volatility_window=20,
            price_adjustment="F",
        )
        fields = WindFieldConfig(
            underlying_code="UNDERLYING",
            maturity_date="MATURITY",
            conversion_start_date="CONVERSION_START",
            put_start_date="PUT_START",
            maturity_redemption_price="REDEMPTION",
            historical_conversion_price="CONVERSION_PRICE",
            historical_coupon_rate="COUPON_RATE",
        )

        with (
            patch("wind_data._load_snapshot_metadata", return_value=metadata),
            patch(
                "wind_data._wind_wsd",
                side_effect=[
                    bond_market,
                    raw_stock,
                    adjusted_stock,
                    terms,
                ],
            ) as wind_wsd,
        ):
            daily, _ = load_wind_daily_inputs(tracking, fields)

        first = daily.iloc[0]
        first_date = pd.Timestamp(first["date"])
        expected_raw = raw_stock.loc[first_date, "CLOSE"]
        expected_adjusted = adjusted_stock.loc[first_date, "CLOSE"]
        adjusted_returns = np.log(
            adjusted_stock["CLOSE"] / adjusted_stock["CLOSE"].shift(1)
        )
        expected_volatility = (
            adjusted_returns.rolling(20, min_periods=20).std(ddof=1)
            * np.sqrt(252)
        ).loc[first_date]

        self.assertAlmostEqual(first["stock_close"], expected_raw)
        self.assertAlmostEqual(
            first["stock_close_adjusted"],
            expected_adjusted,
        )
        self.assertAlmostEqual(first["vol_20d"], expected_volatility)
        self.assertNotAlmostEqual(
            first["stock_close"],
            first["stock_close_adjusted"],
        )
        raw_options = wind_wsd.call_args_list[1].args[4]
        adjusted_options = wind_wsd.call_args_list[2].args[4]
        self.assertNotIn("PriceAdj", raw_options)
        self.assertIn("PriceAdj=F", adjusted_options)


if __name__ == "__main__":
    unittest.main()
