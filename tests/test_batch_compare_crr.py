"""Tests for offline batch orchestration and run publication."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import batch_compare_crr as batch
from crr_model import ManualModelTerms
from pipeline_paths import (
    PipelinePaths,
    build_run_manifest,
    load_current_run,
)
from wind_data import WindFieldConfig


def _terms() -> ManualModelTerms:
    return ManualModelTerms(
        risk_free_rate=0.02,
        dividend_yield=0.0,
        credit_spread=0.01,
        debt_equity_blend_low=70.0,
        debt_equity_blend_high=130.0,
        call_parity_trigger=130.0,
        put_parity_trigger=70.0,
        put_price=103.0,
        tree_steps=10,
    )


def _wind_fields() -> WindFieldConfig:
    return WindFieldConfig(
        underlying_code="bclc",
        maturity_date="maturitdate",
        conversion_start_date="clause_conversion_2_swapsharestartdate",
        put_start_date="clause_putoption_conditionalputbackstartenddate",
        maturity_redemption_price="maturitycallprice",
        historical_conversion_price="convprice",
        historical_coupon_rate="couponrate3",
    )


def _daily(code: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05")],
            "bond_code": [code],
            "stock_code": ["000001.SZ"],
            "maturity_date": [pd.Timestamp("2028-01-01")],
            "T": [2.0],
            "bond_close": [100.0],
            "theoretical_price": [101.0],
            "gap": [1.0],
            "gap_pct": [0.01],
            "abs_gap": [1.0],
            "abs_gap_pct": [0.01],
            "call_impact": [0.5],
            "parity": [100.0],
            "premium_rate": [0.0],
            "sigma_used": [0.25],
            "K": [10.0],
        }
    )


def _saved_result(codes: list[str]) -> batch.BatchResult:
    return batch.BatchResult(
        daily=pd.concat([_daily(code) for code in codes], ignore_index=True),
        summary=pd.DataFrame({"bond_code": codes}),
        common_date_summary=pd.DataFrame({"bond_code": codes}),
        style_summary=pd.DataFrame(
            {"bond_style": ["balanced"], "bond_count": [len(codes)]}
        ),
        failures=pd.DataFrame(columns=["bond_code"]),
    )


class BatchOrchestrationTests(unittest.TestCase):
    """Exercise the existing injection seam without a Wind session."""

    def test_one_bond_failure_does_not_discard_successful_bond(self) -> None:
        specs = [
            batch.BondSpec("123117.SZ", 130.0, 70.0, 103.0),
            batch.BondSpec("118058.SH", 130.0, 70.0, 103.0),
        ]

        def builder(config, fields, terms):
            del fields, terms
            if config.bond_code == "118058.SH":
                raise RuntimeError("synthetic Wind failure")
            return _daily(config.bond_code)

        result = batch.run_batch_comparison(
            specs,
            batch.BatchConfig("2026-01-01", "2026-01-10"),
            _wind_fields(),
            _terms(),
            start_session=False,
            daily_builder=builder,
        )

        self.assertEqual(set(result.daily["bond_code"]), {"123117.SZ"})
        self.assertEqual(set(result.failures["bond_code"]), {"118058.SH"})


class BatchPublicationTests(unittest.TestCase):
    """Verify per-code-set isolation and current-run publication."""

    def _publish(self, root: Path, codes: list[str]) -> PipelinePaths:
        paths = PipelinePaths.from_bond_codes(codes, root)
        result = _saved_result(codes)
        manifest = build_run_manifest(paths, codes, codes, [])
        config = batch.BatchConfig(
            "2026-01-01",
            "2026-01-10",
            output_dir=paths.batch_dir,
        )
        with patch.object(batch, "save_batch_charts"):
            batch.save_batch_result(result, config, manifest)
        return paths

    def test_switching_codes_publishes_new_run_without_overwriting_old(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            first = self._publish(root, ["123117.SZ", "118058.SH"])
            first_bytes = first.batch_daily_csv.read_bytes()
            second = self._publish(root, ["123117.SZ", "123255.SZ"])

            current = load_current_run(root)
            second_daily = pd.read_csv(second.batch_daily_csv)

            self.assertEqual(current.paths.run_id, second.run_id)
            self.assertEqual(
                set(second_daily["bond_code"]),
                {"123117.SZ", "123255.SZ"},
            )
            self.assertNotIn("118058.SH", set(second_daily["bond_code"]))
            self.assertEqual(first.batch_daily_csv.read_bytes(), first_bytes)
            self.assertTrue(first.batch_daily_csv.is_file())

    def test_manifest_is_not_published_when_batch_save_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            old = self._publish(root, ["123117.SZ"])
            new_codes = ["118058.SH"]
            new = PipelinePaths.from_bond_codes(new_codes, root)
            manifest = build_run_manifest(new, new_codes, new_codes, [])
            config = batch.BatchConfig(
                "2026-01-01",
                "2026-01-10",
                output_dir=new.batch_dir,
            )
            with (
                patch.object(
                    batch,
                    "save_batch_charts",
                    side_effect=RuntimeError("chart save failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "chart save failed"),
            ):
                batch.save_batch_result(_saved_result(new_codes), config, manifest)

            self.assertEqual(load_current_run(root).paths.run_id, old.run_id)
            self.assertFalse(new.run_manifest_path.exists())

    def test_batch_directory_must_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            codes = ["123117.SZ"]
            paths = PipelinePaths.from_bond_codes(codes, root)
            manifest = build_run_manifest(paths, codes, codes, [])
            wrong_config = batch.BatchConfig(
                "2026-01-01",
                "2026-01-10",
                output_dir=root / "wrong",
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                batch.save_batch_result(
                    _saved_result(codes),
                    wrong_config,
                    manifest,
                )


if __name__ == "__main__":
    unittest.main()
