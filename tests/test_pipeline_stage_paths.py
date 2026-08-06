"""Integration tests for manifest-driven offline stage paths."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gap_correction_backtest as gap
import implied_volatility_backtest as iv
import implied_volatility_plots as plots
import gap.stage as gap_stage
import implied_volatility.backtest_stage as iv_stage
from pipeline_paths import (
    PipelinePaths,
    RunManifest,
    build_run_manifest,
    save_run_manifest,
)


def _manifest(root: Path, codes: list[str]) -> RunManifest:
    paths = PipelinePaths.from_bond_codes(codes, root)
    return build_run_manifest(paths, codes, codes, [])


def _write_iv_inputs(manifest: RunManifest, daily_code: str) -> None:
    paths = manifest.paths
    paths.iv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bond_code": [daily_code]}).to_csv(
        paths.iv_daily_csv,
        index=False,
    )
    pd.DataFrame({"placeholder": [1]}).to_csv(
        paths.iv_summary_csv,
        index=False,
    )
    pd.DataFrame({"bond_code": [daily_code]}).to_csv(
        paths.iv_selected_csv,
        index=False,
    )
    pd.DataFrame(
        {
            "calibration_stride": [5],
            "max_iv_age_trading_days": [10],
            "minimum_forecast_coverage": [0.8],
            "output_dir": [str(paths.iv_dir)],
        }
    ).to_csv(paths.iv_config_csv, index=False)


class StageConfigTests(unittest.TestCase):
    """Ensure every stage derives all paths from the same run."""

    def test_configs_share_one_manifest_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest(
                Path(directory) / "output",
                ["123117.SZ", "118058.SH"],
            )
            iv_config = iv.BacktestConfig.from_manifest(manifest)
            gap_config = gap.GapCorrectionConfig.from_manifest(manifest)
            plot_config = plots.PlotConfig.from_manifest(manifest)
            paths = manifest.paths

            self.assertEqual(iv_config.input_csv, paths.batch_daily_csv)
            self.assertEqual(iv_config.summary_csv, paths.batch_summary_csv)
            self.assertEqual(iv_config.output_dir, paths.iv_dir)
            self.assertEqual(gap_config.iv_daily_csv, paths.iv_daily_csv)
            self.assertEqual(gap_config.iv_summary_csv, paths.iv_summary_csv)
            self.assertEqual(gap_config.iv_selected_csv, paths.iv_selected_csv)
            self.assertEqual(gap_config.iv_config_csv, paths.iv_config_csv)
            self.assertEqual(gap_config.output_dir, paths.gap_dir)
            self.assertEqual(
                plot_config.batch_summary_csv,
                paths.batch_summary_csv,
            )
            self.assertEqual(plot_config.output_dir, paths.figures_dir)

    def test_switching_current_run_switches_every_offline_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            first = _manifest(root, ["123117.SZ", "118058.SH"])
            second = _manifest(root, ["123117.SZ", "123255.SZ"])
            save_run_manifest(first)
            save_run_manifest(second)

            iv_config = iv.BacktestConfig.from_current_run(root)
            gap_config = gap.GapCorrectionConfig.from_current_run(root)
            plot_config = plots.PlotConfig.from_current_run(root)

            self.assertEqual(iv_config.input_csv, second.paths.batch_daily_csv)
            self.assertEqual(iv_config.output_dir, second.paths.iv_dir)
            self.assertEqual(gap_config.iv_daily_csv, second.paths.iv_daily_csv)
            self.assertEqual(gap_config.output_dir, second.paths.gap_dir)
            self.assertEqual(plot_config.output_dir, second.paths.figures_dir)
            self.assertNotEqual(iv_config.input_csv, first.paths.batch_daily_csv)


class ImpliedVolatilityStageTests(unittest.TestCase):
    """Verify IV rejects stale inputs and writes only inside its run."""

    def test_iv_rejects_batch_codes_from_another_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest(Path(directory) / "output", ["123117.SZ"])
            paths = manifest.paths
            paths.batch_dir.mkdir(parents=True)
            pd.DataFrame({"bond_code": ["118058.SH"]}).to_csv(
                paths.batch_daily_csv,
                index=False,
            )
            pd.DataFrame({"bond_code": ["123117.SZ"]}).to_csv(
                paths.batch_summary_csv,
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "do not match"):
                iv.run_iv_stage(
                    iv.BacktestConfig.from_manifest(manifest),
                    expected_bond_codes=manifest.successful_bond_codes,
                    run_id=paths.run_id,
                )

            self.assertFalse(paths.iv_dir.exists())

    def test_iv_reports_incomplete_batch_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest(Path(directory) / "output", ["123117.SZ"])

            with self.assertRaisesRegex(
                FileNotFoundError,
                f"Batch artifacts are incomplete for run {manifest.paths.run_id}",
            ):
                iv.run_iv_stage(
                    iv.BacktestConfig.from_manifest(manifest),
                    expected_bond_codes=manifest.successful_bond_codes,
                    run_id=manifest.paths.run_id,
                )

            self.assertFalse(manifest.paths.iv_dir.exists())

    def test_iv_stage_writes_all_artifacts_to_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest(Path(directory) / "output", ["123117.SZ"])
            paths = manifest.paths
            paths.batch_dir.mkdir(parents=True)
            pd.DataFrame({"bond_code": ["123117.SZ"]}).to_csv(
                paths.batch_daily_csv,
                index=False,
            )
            pd.DataFrame({"bond_code": ["123117.SZ"]}).to_csv(
                paths.batch_summary_csv,
                index=False,
            )
            frame = pd.DataFrame({"bond_code": ["123117.SZ"]})

            with (
                patch.object(iv_stage, "load_terms_snapshot", return_value={}),
                patch.object(iv_stage, "run_backtest", return_value=frame),
                patch.object(iv_stage, "summarize_backtest", return_value=frame),
                patch.object(iv_stage, "select_models", return_value=frame),
                patch.object(iv_stage, "summarize_by_style", return_value=frame),
                patch.object(
                    iv_stage,
                    "calibration_diagnostics",
                    return_value=frame,
                ),
                redirect_stdout(io.StringIO()),
            ):
                iv.run_iv_stage(
                    iv.BacktestConfig.from_manifest(manifest),
                    expected_bond_codes=manifest.successful_bond_codes,
                    run_id=paths.run_id,
                )

            expected = {
                paths.iv_daily_csv,
                paths.iv_summary_csv,
                paths.iv_selected_csv,
                paths.iv_dir / "implied_volatility_summary_by_style.csv",
                paths.iv_calibration_csv,
                paths.iv_config_csv,
            }
            self.assertTrue(all(path.is_file() for path in expected))
            saved_config = pd.read_csv(paths.iv_config_csv)
            self.assertEqual(saved_config.loc[0, "output_dir"], str(paths.iv_dir))
            self.assertEqual(
                saved_config.loc[0, "input_csv"],
                str(paths.batch_daily_csv),
            )


class GapStageTests(unittest.TestCase):
    """Verify gap/EWMA cannot consume or write across run boundaries."""

    def test_gap_rejects_iv_daily_codes_from_another_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest(Path(directory) / "output", ["123117.SZ"])
            _write_iv_inputs(manifest, "118058.SH")

            with self.assertRaisesRegex(ValueError, "do not match"):
                gap.run_gap_stage(
                    gap.GapCorrectionConfig.from_manifest(manifest),
                    expected_bond_codes=manifest.successful_bond_codes,
                    run_id=manifest.paths.run_id,
                )

            self.assertFalse(manifest.paths.gap_dir.exists())

    def test_gap_reports_incomplete_iv_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest(Path(directory) / "output", ["123117.SZ"])

            with self.assertRaisesRegex(
                FileNotFoundError,
                f"IV artifacts are incomplete for run {manifest.paths.run_id}",
            ):
                gap.run_gap_stage(
                    gap.GapCorrectionConfig.from_manifest(manifest),
                    expected_bond_codes=manifest.successful_bond_codes,
                    run_id=manifest.paths.run_id,
                )

            self.assertFalse(manifest.paths.gap_dir.exists())

    def test_gap_rejects_iv_config_recorded_for_another_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest(Path(directory) / "output", ["123117.SZ"])
            _write_iv_inputs(manifest, "123117.SZ")
            iv_config = pd.read_csv(manifest.paths.iv_config_csv)
            iv_config["output_dir"] = str(
                manifest.paths.output_root / "runs" / "OTHER" / "implied_volatility"
            )
            iv_config.to_csv(manifest.paths.iv_config_csv, index=False)

            with self.assertRaisesRegex(ValueError, "different run"):
                gap.run_gap_stage(
                    gap.GapCorrectionConfig.from_manifest(manifest),
                    expected_bond_codes=manifest.successful_bond_codes,
                    run_id=manifest.paths.run_id,
                )

    def test_gap_stage_writes_all_artifacts_to_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = _manifest(Path(directory) / "output", ["123117.SZ"])
            _write_iv_inputs(manifest, "123117.SZ")
            frame = pd.DataFrame({"bond_code": ["123117.SZ"]})

            with (
                patch.object(
                    gap_stage,
                    "validation_start_date",
                    return_value=pd.Timestamp("2026-01-05"),
                ),
                patch.object(
                    gap_stage,
                    "add_gap_correction_forecasts",
                    return_value=frame,
                ),
                patch.object(
                    gap_stage,
                    "summarize_gap_corrections",
                    return_value=frame,
                ),
                patch.object(
                    gap_stage,
                    "summarize_head_to_head",
                    return_value=frame,
                ),
                patch.object(
                    gap_stage,
                    "summarize_headline",
                    return_value=frame,
                ),
                redirect_stdout(io.StringIO()),
            ):
                gap.run_gap_stage(
                    gap.GapCorrectionConfig.from_manifest(manifest),
                    expected_bond_codes=manifest.successful_bond_codes,
                    run_id=manifest.paths.run_id,
                )

            expected = {
                manifest.paths.gap_dir / "gap_correction_backtest_daily.csv",
                manifest.paths.gap_dir / "gap_correction_backtest_summary.csv",
                manifest.paths.gap_head_to_head_csv,
                manifest.paths.gap_headline_csv,
                manifest.paths.gap_dir / "gap_correction_backtest_config.csv",
            }
            self.assertTrue(all(path.is_file() for path in expected))
            saved_config = pd.read_csv(
                manifest.paths.gap_dir / "gap_correction_backtest_config.csv"
            )
            self.assertEqual(
                saved_config.loc[0, "iv_daily_csv"],
                str(manifest.paths.iv_daily_csv),
            )
            self.assertEqual(
                saved_config.loc[0, "output_dir"],
                str(manifest.paths.gap_dir),
            )


if __name__ == "__main__":
    unittest.main()
