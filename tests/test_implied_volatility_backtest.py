"""Regression tests for leakage-safe implied-volatility backtesting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import implied_volatility_backtest as iv_backtest
from crr_model import ManualModelTerms
from implied_volatility import ImpliedVolatilityResult


def _terms() -> ManualModelTerms:
    """Return compact deterministic model terms shared by test fixtures."""

    return ManualModelTerms(
        risk_free_rate=0.02,
        dividend_yield=0.0,
        credit_spread=0.01,
        debt_equity_blend_low=70.0,
        debt_equity_blend_high=130.0,
        call_parity_trigger=130.0,
        put_parity_trigger=70.0,
        put_price=105.0,
        tree_steps=20,
    )


def _daily(code: str, start_price: float, days: int = 6) -> pd.DataFrame:
    """Build one complete synthetic bond history for backtest tests."""

    positions = np.arange(days, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=days, freq="B"),
            "bond_code": code,
            "stock_code": "000001.SZ",
            "bond_close": start_price + positions,
            "stock_close": 10.0,
            "K": 10.0,
            "coupon_used": 0.01,
            "T": 5.0 - positions / 252,
            "t_conv_used": 0.0,
            "t_put_used": 0.0,
            "sigma_used": 0.2,
            "maturity_redemption_price": 110.0,
            "theoretical_price": 90.0 + positions,
            "theoretical_price_no_call": 91.0 + positions,
        }
    )


def _result(sigma: float) -> ImpliedVolatilityResult:
    """Build a successful deterministic IV solver response."""

    return ImpliedVolatilityResult(
        implied_sigma=sigma,
        success=True,
        reason="test",
        iterations=1,
        target_price=100.0,
        model_price=100.0,
        residual=0.0,
        lower_sigma=sigma,
        upper_sigma=sigma,
        lower_price=100.0,
        upper_price=100.0,
        grid_min_price=90.0,
        grid_max_price=110.0,
        valid_grid_points=2,
        monotonic=True,
        bracket_count=1,
    )


def _fake_solve(
    row: object,
    terms: ManualModelTerms,
    config: iv_backtest.BacktestConfig,
    *,
    with_call: bool,
) -> ImpliedVolatilityResult:
    """Replace numerical inversion with a predictable row-dependent result."""

    del terms, config
    offset = 0.0 if with_call else 0.5
    return _result(float(getattr(row, "bond_close")) / 100 + offset)


def _fake_price(
    row: object,
    terms: ManualModelTerms,
    sigma: float,
    *,
    with_call: bool,
) -> float:
    """Replace CRR pricing with an engine-specific deterministic value."""

    del row, terms
    return float(sigma) + (10.0 if with_call else 20.0)


class ImpliedVolatilityBacktestTests(unittest.TestCase):
    """Verify chronology, isolation, coverage, and common-sample metrics."""

    def setUp(self) -> None:
        """Use short calibration windows so small fixtures exercise expiry."""

        self.config = iv_backtest.BacktestConfig(
            calibration_stride=2,
            rolling_calibration_count=2,
            max_iv_age_trading_days=4,
            validation_fraction=0.4,
            minimum_forecast_coverage=0.5,
        )

    def _run(
        self,
        daily: pd.DataFrame,
        terms_by_code: dict[str, ManualModelTerms],
    ) -> pd.DataFrame:
        """Run the backtest with deterministic solver and pricing doubles."""

        with (
            patch.object(iv_backtest, "_solve_row", side_effect=_fake_solve),
            patch.object(iv_backtest, "_price", side_effect=_fake_price),
        ):
            return iv_backtest.run_backtest(
                daily,
                terms_by_code,
                self.config,
            )

    def test_same_day_calibration_never_prices_same_day(self) -> None:
        result = self._run(_daily("A", 100.0), {"A": _terms()})
        sigma = result["sigma_iv_latest_calibration_with_call"]

        self.assertTrue(np.isnan(sigma.iloc[0]))
        self.assertAlmostEqual(sigma.iloc[1], 1.00)
        self.assertAlmostEqual(sigma.iloc[2], 1.00)
        self.assertAlmostEqual(sigma.iloc[3], 1.02)

    def test_adding_future_rows_does_not_change_prefix(self) -> None:
        daily = _daily("A", 100.0)
        short = self._run(daily.iloc[:4], {"A": _terms()})
        full = self._run(daily, {"A": _terms()})

        pd.testing.assert_frame_equal(
            short[list(iv_backtest.FORECAST_COLUMNS)].reset_index(drop=True),
            full.loc[:3, list(iv_backtest.FORECAST_COLUMNS)].reset_index(
                drop=True
            ),
        )

    def test_shuffled_multi_bond_input_keeps_histories_isolated(self) -> None:
        first = _daily("A", 100.0)
        second = _daily("B", 200.0)
        combined = pd.concat([first, second], ignore_index=True).sample(
            frac=1.0,
            random_state=7,
        )
        terms = {"A": _terms(), "B": _terms()}

        together = self._run(combined, terms).sort_values(
            ["bond_code", "date"]
        )
        separate = pd.concat(
            [self._run(first, {"A": terms["A"]}), self._run(second, {"B": terms["B"]})],
            ignore_index=True,
        ).sort_values(["bond_code", "date"])

        pd.testing.assert_frame_equal(
            together[["bond_code", "date", *iv_backtest.FORECAST_COLUMNS]].reset_index(
                drop=True
            ),
            separate[["bond_code", "date", *iv_backtest.FORECAST_COLUMNS]].reset_index(
                drop=True
            ),
        )

    def test_duplicate_bond_date_rows_are_rejected(self) -> None:
        daily = _daily("A", 100.0)
        duplicated = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate bond/date"):
            self._run(duplicated, {"A": _terms()})

    def test_forecast_pricing_error_is_recorded_without_stopping(self) -> None:
        daily = _daily("A", 100.0, days=3)
        with (
            patch.object(iv_backtest, "_solve_row", side_effect=_fake_solve),
            patch.object(
                iv_backtest,
                "_price",
                side_effect=ValueError("bad forecast point"),
            ),
        ):
            result = iv_backtest.run_backtest(
                daily,
                {"A": _terms()},
                self.config,
            )

        price_column = "price_iv_latest_calibration_with_call"
        self.assertEqual(len(result), 3)
        self.assertTrue(np.isnan(result.loc[1, price_column]))
        self.assertIn(
            "bad forecast point",
            result.loc[1, f"{price_column}_error"],
        )

    def test_implied_volatility_expires_after_configured_age(self) -> None:
        config = iv_backtest.BacktestConfig(
            calibration_stride=100,
            rolling_calibration_count=2,
            max_iv_age_trading_days=2,
            validation_fraction=0.4,
            minimum_forecast_coverage=0.0,
        )
        with (
            patch.object(iv_backtest, "_solve_row", side_effect=_fake_solve),
            patch.object(iv_backtest, "_price", side_effect=_fake_price),
        ):
            result = iv_backtest.run_backtest(
                _daily("A", 100.0, days=4),
                {"A": _terms()},
                config,
            )

        sigma = result["sigma_iv_latest_calibration_with_call"]
        self.assertAlmostEqual(sigma.iloc[1], 1.0)
        self.assertAlmostEqual(sigma.iloc[2], 1.0)
        self.assertTrue(np.isnan(sigma.iloc[3]))

    def test_summary_uses_one_common_forecast_sample(self) -> None:
        rows = 10
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=rows, freq="B"),
                "bond_code": "A",
                "market_price": 100.0,
                "baseline_price_with_call": 80.0,
                "baseline_price_no_call": 85.0,
                "bond_style": "balanced",
            }
        )
        for model in iv_backtest.FORECAST_COLUMNS:
            frame[model] = 90.0 if model.endswith("_with_call") else 92.0
        frame.loc[1, iv_backtest.FORECAST_COLUMNS[0]] = np.nan

        summary = iv_backtest.summarize_backtest(frame, self.config)
        calibration = summary.loc[summary["segment"].eq("calibration")]

        call_calibration = calibration.loc[
            calibration["engine"].eq("with_call")
        ]
        no_call_calibration = calibration.loc[
            calibration["engine"].eq("no_call")
        ]
        self.assertEqual(set(call_calibration["comparison_days"]), {5})
        self.assertEqual(set(no_call_calibration["comparison_days"]), {6})
        baseline = calibration.loc[
            calibration["model"].eq("baseline_price_with_call")
        ].iloc[0]
        self.assertAlmostEqual(baseline["mape"], 0.20)
        selected = iv_backtest.select_models(summary, self.config)
        selected = selected.loc[selected["engine"].eq("no_call")].iloc[0]
        self.assertTrue(selected["selection_passed"])
        self.assertGreater(selected["validation_mape_improvement"], 0)
        self.assertTrue(selected["validation_passed"])

    def test_all_bonds_share_one_calendar_validation_cutoff(self) -> None:
        first = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=10, freq="B"),
                "bond_code": "A",
            }
        )
        second = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-07", periods=6, freq="B"),
                "bond_code": "B",
            }
        )
        frame = pd.concat([first, second], ignore_index=True)
        frame["market_price"] = 100.0
        frame["baseline_price_with_call"] = 90.0
        frame["baseline_price_no_call"] = 92.0
        frame["bond_style"] = "balanced"
        for model in iv_backtest.FORECAST_COLUMNS:
            frame[model] = 95.0

        summary = iv_backtest.summarize_backtest(frame, self.config)

        self.assertEqual(summary["validation_start_date"].nunique(), 1)


class TermsSnapshotTests(unittest.TestCase):
    """Verify reconstruction and validation of saved per-bond assumptions."""

    def _summary(self) -> pd.DataFrame:
        """Return the minimum valid saved-terms snapshot."""

        return pd.DataFrame(
            {
                "bond_code": ["A"],
                "risk_free_rate": [0.02],
                "dividend_yield": [0.0],
                "credit_spread": [0.01],
                "call_parity_trigger": [130.0],
                "put_parity_trigger": [70.0],
                "put_price": [105.0],
                "tree_steps": [200],
            }
        )

    def test_old_snapshot_uses_documented_blend_defaults(self) -> None:
        terms = iv_backtest.load_terms_snapshot(self._summary())["A"]

        self.assertEqual(terms.debt_equity_blend_low, 70.0)
        self.assertEqual(terms.debt_equity_blend_high, 130.0)

    def test_duplicate_bonds_are_rejected(self) -> None:
        summary = pd.concat([self._summary(), self._summary()])

        with self.assertRaisesRegex(ValueError, "duplicate bonds"):
            iv_backtest.load_terms_snapshot(summary)

    def test_nan_in_present_blend_snapshot_is_rejected(self) -> None:
        summary = self._summary()
        summary["debt_equity_blend_low"] = np.nan
        summary["debt_equity_blend_high"] = 130.0

        with self.assertRaisesRegex(ValueError, "invalid debt_equity_blend_low"):
            iv_backtest.load_terms_snapshot(summary)


if __name__ == "__main__":
    unittest.main()
