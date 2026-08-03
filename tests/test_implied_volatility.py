from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from implied_volatility import solve_implied_volatility


class ImpliedVolatilitySolverTests(unittest.TestCase):
    def test_finds_unique_root(self) -> None:
        result = solve_implied_volatility(
            lambda sigma: 100 + 10 * sigma,
            105,
            sigma_grid=(0.1, 1.0),
            max_iterations=40,
            price_tolerance=1e-8,
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.implied_sigma, 0.5, places=7)
        self.assertLessEqual(abs(result.residual), 1e-8)

    def test_skips_invalid_grid_points(self) -> None:
        def price(sigma: float) -> float:
            if sigma < 0.2:
                raise ValueError("invalid CRR probability")
            return 100 + 10 * sigma

        result = solve_implied_volatility(
            price,
            105,
            sigma_grid=(0.01, 0.2, 1.0),
            price_tolerance=1e-8,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.valid_grid_points, 2)
        self.assertAlmostEqual(result.implied_sigma, 0.5, places=7)

    def test_reports_target_outside_price_range(self) -> None:
        result = solve_implied_volatility(
            lambda sigma: 100 + sigma,
            120,
            sigma_grid=(0.1, 1.0),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "target_outside_model_price_range")

    def test_rejects_non_monotonic_price_curve(self) -> None:
        result = solve_implied_volatility(
            lambda sigma: -(sigma - 1) ** 2 + 2,
            1.5,
            sigma_grid=(0.1, 1.0, 2.0),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "non_monotonic_price_curve")

    def test_narrow_sigma_interval_is_not_false_convergence(self) -> None:
        result = solve_implied_volatility(
            lambda sigma: 0.0 if sigma < 0.5 else 1.0,
            0.4,
            sigma_grid=(0.1, 1.0),
            max_iterations=100,
            price_tolerance=0.01,
            sigma_tolerance=1e-6,
        )

        self.assertFalse(result.success)
        self.assertEqual(
            result.reason,
            "sigma_interval_collapsed_without_price_match",
        )
        self.assertGreater(abs(result.residual), 0.01)


if __name__ == "__main__":
    unittest.main()
