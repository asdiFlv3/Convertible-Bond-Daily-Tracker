"""Regression tests for leakage-safe lagged-gap correction benchmarks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gap_correction_backtest as gap_backtest


def _daily(
    code: str = "A",
    *,
    days: int = 6,
    calibration_positions: tuple[int, ...] = (0, 2, 4),
) -> pd.DataFrame:
    """Build a compact saved-IV daily table for one synthetic bond."""

    position = np.arange(days, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=days, freq="B"),
            "bond_code": code,
            "market_price": 100.0 + position,
            "bond_style": "balanced",
            "is_calibration_date": [
                index in calibration_positions for index in range(days)
            ],
            "baseline_price_with_call": 88.0 + position,
            "baseline_price_no_call": 90.0 + position,
            "price_iv_latest_calibration_with_call": 99.0 + position,
            "price_iv_latest_calibration_no_call": 99.5 + position,
        }
    )


class GapCorrectionForecastTests(unittest.TestCase):
    """Verify sign, timing, smoothing, expiry, and history isolation."""

    def setUp(self) -> None:
        self.config = gap_backtest.GapCorrectionConfig(
            calibration_stride=2,
            max_gap_age_trading_days=4,
            ewma_alpha=0.5,
            minimum_forecast_coverage=0.5,
        )

    def test_negative_gap_raises_next_day_corrected_price(self) -> None:
        result = gap_backtest.add_gap_correction_forecasts(
            _daily(days=2, calibration_positions=(0,)),
            self.config,
        )

        column = gap_backtest.LATEST_GAP_PRICE_BY_ENGINE["no_call"]
        self.assertTrue(np.isnan(result.loc[0, column]))
        self.assertAlmostEqual(result.loc[0, "realized_gap_no_call"], -10.0)
        self.assertAlmostEqual(result.loc[1, column], 101.0)

    def test_current_calibration_only_changes_following_day(self) -> None:
        daily = _daily(days=4, calibration_positions=(0, 2))
        daily.loc[2, "market_price"] = 120.0
        result = gap_backtest.add_gap_correction_forecasts(
            daily,
            self.config,
        )

        latest = gap_backtest.LATEST_GAP_PRICE_BY_ENGINE["no_call"]
        ewma = gap_backtest.EWMA_GAP_PRICE_BY_ENGINE["no_call"]
        self.assertAlmostEqual(result.loc[2, latest], 102.0)
        self.assertAlmostEqual(result.loc[3, latest], 121.0)
        self.assertAlmostEqual(result.loc[3, ewma], 112.0)

    def test_adding_future_rows_does_not_change_existing_forecasts(self) -> None:
        daily = _daily(days=6)
        short = gap_backtest.add_gap_correction_forecasts(
            daily.iloc[:4],
            self.config,
        )
        full = gap_backtest.add_gap_correction_forecasts(
            daily,
            self.config,
        )
        columns = [
            *gap_backtest.LATEST_GAP_PRICE_BY_ENGINE.values(),
            *gap_backtest.EWMA_GAP_PRICE_BY_ENGINE.values(),
        ]

        pd.testing.assert_frame_equal(
            short[columns].reset_index(drop=True),
            full.loc[:3, columns].reset_index(drop=True),
        )

    def test_bond_histories_are_isolated(self) -> None:
        first = _daily("A")
        second = _daily("B")
        second["market_price"] += 50.0
        shuffled = pd.concat([first, second], ignore_index=True).sample(
            frac=1.0,
            random_state=11,
        )
        together = gap_backtest.add_gap_correction_forecasts(
            shuffled,
            self.config,
        ).sort_values(["bond_code", "date"])
        separate = pd.concat(
            [
                gap_backtest.add_gap_correction_forecasts(first, self.config),
                gap_backtest.add_gap_correction_forecasts(second, self.config),
            ],
            ignore_index=True,
        ).sort_values(["bond_code", "date"])
        columns = [
            "bond_code",
            "date",
            *gap_backtest.LATEST_GAP_PRICE_BY_ENGINE.values(),
            *gap_backtest.EWMA_GAP_PRICE_BY_ENGINE.values(),
        ]

        pd.testing.assert_frame_equal(
            together[columns].reset_index(drop=True),
            separate[columns].reset_index(drop=True),
        )

    def test_gap_state_expires_after_maximum_age(self) -> None:
        config = gap_backtest.GapCorrectionConfig(
            calibration_stride=100,
            max_gap_age_trading_days=2,
            ewma_alpha=0.5,
            minimum_forecast_coverage=0.0,
        )
        result = gap_backtest.add_gap_correction_forecasts(
            _daily(days=4, calibration_positions=(0,)),
            config,
        )
        latest = gap_backtest.LATEST_GAP_PRICE_BY_ENGINE["no_call"]

        self.assertTrue(np.isfinite(result.loc[1, latest]))
        self.assertTrue(np.isfinite(result.loc[2, latest]))
        self.assertTrue(np.isnan(result.loc[3, latest]))

    def test_duplicate_bond_dates_are_rejected(self) -> None:
        daily = _daily(days=3)
        duplicate = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate bond/date"):
            gap_backtest.add_gap_correction_forecasts(
                duplicate,
                self.config,
            )


class GapCorrectionSummaryTests(unittest.TestCase):
    """Verify common samples and the pre-qualified headline population."""

    def setUp(self) -> None:
        self.config = gap_backtest.GapCorrectionConfig(
            calibration_stride=2,
            max_gap_age_trading_days=4,
            ewma_alpha=0.5,
            minimum_forecast_coverage=0.5,
        )
        self.cutoff = pd.Timestamp("2026-01-07")

    def _selected(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "bond_code": "A",
                    "engine": "no_call",
                    "selection_style": "balanced",
                    "selection_passed": True,
                    "selected_model": (
                        "price_iv_latest_calibration_no_call"
                    ),
                    "validation_passed": True,
                    "validation_comparison_coverage": 1.0,
                },
                {
                    "bond_code": "B",
                    "engine": "no_call",
                    "selection_style": "balanced",
                    "selection_passed": True,
                    "selected_model": (
                        "price_iv_latest_calibration_no_call"
                    ),
                    "validation_passed": False,
                    "validation_comparison_coverage": 0.75,
                },
                {
                    "bond_code": "C",
                    "engine": "no_call",
                    "selection_style": "balanced",
                    "selection_passed": False,
                    "selected_model": np.nan,
                    "validation_passed": False,
                    "validation_comparison_coverage": 0.60,
                },
            ]
        )

    def test_head_to_head_uses_one_common_sample(self) -> None:
        backtest = gap_backtest.add_gap_correction_forecasts(
            _daily("A", days=8),
            self.config,
        )
        iv_column = "price_iv_latest_calibration_no_call"
        backtest.loc[5, iv_column] = np.nan
        selected = self._selected().iloc[[0]]

        summary = gap_backtest.summarize_head_to_head(
            backtest,
            selected,
            self.cutoff,
        )
        validation = summary.loc[summary["segment"].eq("validation")]

        self.assertEqual(set(validation["comparison_days"]), {3})
        self.assertEqual(validation["comparison_coverage"].nunique(), 1)

    def test_headline_keeps_validation_failure_but_not_selection_failure(
        self,
    ) -> None:
        daily = pd.concat(
            [_daily("A", days=8), _daily("B", days=8), _daily("C", days=8)],
            ignore_index=True,
        )
        backtest = gap_backtest.add_gap_correction_forecasts(
            daily,
            self.config,
        )
        head_to_head = gap_backtest.summarize_head_to_head(
            backtest,
            self._selected(),
            self.cutoff,
        )
        headline = gap_backtest.summarize_headline(head_to_head)

        self.assertEqual(set(head_to_head["bond_code"]), {"A", "B"})
        self.assertEqual(set(headline["eligible_bond_count"]), {2})
        self.assertEqual(
            set(headline["iv_validation_passed_bond_count"]),
            {1},
        )
        self.assertEqual(set(headline["metric_bond_count"]), {2})


if __name__ == "__main__":
    unittest.main()
