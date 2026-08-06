"""File-oriented runner for one manifest-selected IV backtest."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from pipeline_paths import load_current_run, require_artifacts, validate_exact_bond_codes
from .backtest_config import BacktestConfig
from .backtest_engine import run_backtest
from .backtest_reporting import (
    calibration_diagnostics,
    summarize_backtest,
    summarize_by_style,
)
from .model_selection import select_models
from .terms import load_terms_snapshot


def run_iv_stage(
    config: BacktestConfig,
    *,
    expected_bond_codes: tuple[str, ...] | None = None,
    run_id: str = "configured",
) -> None:
    """Run one configured IV stage and persist reproducible outputs."""

    if config.input_csv is None:
        raise ValueError("IV input_csv is not configured")
    if config.summary_csv is None:
        raise ValueError("IV summary_csv is not configured")
    if config.output_dir is None:
        raise ValueError("IV output_dir is not configured")
    require_artifacts(
        [config.input_csv, config.summary_csv],
        "Batch",
        run_id,
    )
    daily = pd.read_csv(config.input_csv)
    bond_metadata = pd.read_csv(config.summary_csv)
    # When invoked from current_run.json, 
    # membership checks prevent a partial or copied CSV from being combined with another run's terms snapshot.
    if expected_bond_codes is not None:
        if "bond_code" not in daily.columns:
            raise KeyError(
                f"{config.input_csv} missing required column: bond_code"
            )
        if "bond_code" not in bond_metadata.columns:
            raise KeyError(
                f"{config.summary_csv} missing required column: bond_code"
            )
        validate_exact_bond_codes(
            daily["bond_code"].dropna().astype(str).unique(),
            expected_bond_codes,
            str(config.input_csv),
        )
        validate_exact_bond_codes(
            bond_metadata["bond_code"].dropna().astype(str).unique(),
            expected_bond_codes,
            str(config.summary_csv),
        )
    terms_by_code = load_terms_snapshot(bond_metadata)
    # Compute first, then persist the complete result set. Downstream stages
    # require all of these artifacts and fail early if one is missing.
    backtest = run_backtest(daily, terms_by_code, config)
    summary = summarize_backtest(backtest, config)
    selected = select_models(summary, config)
    style_summary = summarize_by_style(summary)
    calibration = calibration_diagnostics(backtest)
    config_values = asdict(config)
    config_values["input_csv"] = str(config.input_csv)
    config_values["summary_csv"] = str(config.summary_csv)
    config_values["output_dir"] = str(config.output_dir)
    config_values["sigma_grid"] = ",".join(
        str(value) for value in config.sigma_grid
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(
        config.output_dir / "implied_volatility_backtest_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        config.output_dir / "implied_volatility_backtest_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.to_csv(
        config.output_dir / "implied_volatility_selected_models.csv",
        index=False,
        encoding="utf-8-sig",
    )
    style_summary.to_csv(
        config.output_dir / "implied_volatility_summary_by_style.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibration.to_csv(
        config.output_dir / "implied_volatility_calibration_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([config_values]).to_csv(
        config.output_dir / "implied_volatility_backtest_config.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Calibration diagnostics:")
    print(calibration.to_string(index=False))
    print("\nCalibration-selected models evaluated on validation dates:")
    print(selected.to_string(index=False))
    print(f"\nSaved outputs to {config.output_dir}")


def main() -> None:
    """Run IV for the latest usable batch selected by current_run.json."""

    manifest = load_current_run()
    run_iv_stage(
        BacktestConfig.from_manifest(manifest),
        expected_bond_codes=manifest.successful_bond_codes,
        run_id=manifest.paths.run_id,
    )
