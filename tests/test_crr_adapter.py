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


if __name__ == "__main__":
    unittest.main()
