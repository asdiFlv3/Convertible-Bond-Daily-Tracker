"""Regression tests for the public CRR terms adapter and input guards."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from crr_model import (
    FACE_VALUE,
    ManualModelTerms,
    crr_convertible_basic,
    price_convertible_with_terms,
)


class ConvertiblePricingAdapterTests(unittest.TestCase):
    """Exercise adapter parity, option precedence, and validation boundaries."""

    @staticmethod
    def _direct_price(**overrides: object) -> float:
        """Price a minimal one-step fixture with optional input overrides."""

        inputs: dict[str, object] = {
            "stock_price": 100.0,
            "conversion_price": 100.0,
            "sigma": 1.0,
            "risk_free_rate": 0.0,
            "maturity_years": 1.0,
            "dividend_yield": 0.2,
            "maturity_redemption_price": 1.0,
            "coupon_rate": 0.0,
            "credit_spread": 0.0,
            "blend_low": 70.0,
            "blend_high": 130.0,
            "conversion_wait_years": 0.0,
            "put_wait_years": 0.0,
            "steps": 1,
            "call_parity_trigger": float("inf"),
            "put_parity_trigger": 200.0,
            "face_value": FACE_VALUE,
            "put_price": 50.0,
        }
        inputs.update(overrides)
        return crr_convertible_basic(**inputs)  # type: ignore[arg-type]

    def test_adapter_matches_direct_crr_call_for_both_call_settings(self) -> None:
        terms = ManualModelTerms(
            risk_free_rate=0.02,
            dividend_yield=0.01,
            credit_spread=0.015,
            debt_equity_blend_low=75.0,
            debt_equity_blend_high=125.0,
            call_parity_trigger=130.0,
            put_parity_trigger=70.0,
            put_price=105.0,
            tree_steps=50,
        )
        daily_inputs = {
            "stock_price": 20.0,
            "conversion_price": 15.0,
            "sigma": 0.4,
            "maturity_years": 3.0,
            "maturity_redemption_price": 110.0,
            "coupon_rate": 0.01,
            "conversion_wait_years": 0.0,
            "put_wait_years": 1.0,
        }

        for with_call in (True, False):
            adapted = price_convertible_with_terms(
                **daily_inputs,
                terms=terms,
                with_call=with_call,
            )
            direct = crr_convertible_basic(
                stock_price=daily_inputs["stock_price"],
                conversion_price=daily_inputs["conversion_price"],
                sigma=daily_inputs["sigma"],
                risk_free_rate=terms.risk_free_rate,
                maturity_years=daily_inputs["maturity_years"],
                dividend_yield=terms.dividend_yield,
                maturity_redemption_price=daily_inputs[
                    "maturity_redemption_price"
                ],
                coupon_rate=daily_inputs["coupon_rate"],
                credit_spread=terms.credit_spread,
                blend_low=terms.debt_equity_blend_low,
                blend_high=terms.debt_equity_blend_high,
                conversion_wait_years=daily_inputs["conversion_wait_years"],
                put_wait_years=daily_inputs["put_wait_years"],
                steps=terms.tree_steps,
                call_parity_trigger=(
                    terms.call_parity_trigger if with_call else float("inf")
                ),
                put_parity_trigger=terms.put_parity_trigger,
                face_value=FACE_VALUE,
                put_price=terms.put_price,
            )

            self.assertAlmostEqual(adapted, direct, places=12)

    def test_put_floor_preserves_a_better_conversion_value(self) -> None:
        price = self._direct_price()

        self.assertAlmostEqual(price, 100.0, places=12)

    def test_steps_must_be_a_positive_integer(self) -> None:
        for invalid_steps in (0, -1, 0.5, True):
            with self.subTest(steps=invalid_steps):
                with self.assertRaisesRegex(
                    ValueError,
                    "steps must be a positive integer",
                ):
                    self._direct_price(steps=invalid_steps)

    def test_non_finite_positive_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sigma"):
            self._direct_price(sigma=float("nan"))


if __name__ == "__main__":
    unittest.main()
