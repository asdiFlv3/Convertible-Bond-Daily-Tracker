"""Tests for the offline implied-volatility plotting helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import implied_volatility_plots as plots


class ImpliedVolatilityPlotTests(unittest.TestCase):
    """Verify selection grouping and non-empty small-multiple layouts."""

    def test_validation_groups_use_frozen_selection_style(self) -> None:
        selected = pd.DataFrame(
            [
                {
                    "bond_code": "A",
                    "engine": "no_call",
                    "selection_style": "debt",
                },
                {
                    "bond_code": "B",
                    "engine": "no_call",
                    "selection_style": "balanced",
                },
                {
                    "bond_code": "C",
                    "engine": "call",
                    "selection_style": "balanced",
                },
            ]
        )

        self.assertEqual(
            plots._selection_style_codes(selected, "balanced"),
            ["B"],
        )

    def test_five_bond_layout_has_no_empty_panel(self) -> None:
        figure, axes = plots._small_multiple_figure(5)
        try:
            self.assertEqual(len(axes), 5)
            self.assertEqual(len(figure.axes), 5)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
