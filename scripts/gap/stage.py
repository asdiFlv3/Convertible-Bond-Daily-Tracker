"""File-oriented runner for one manifest-selected gap/EWMA backtest."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from pipeline_paths import load_current_run, require_artifacts, validate_exact_bond_codes
from .config import GapCorrectionConfig
from .correction import _validate_config, add_gap_correction_forecasts, validate_matched_iv_rules
from .reporting import (
    summarize_gap_corrections,
    summarize_head_to_head,
    summarize_headline,
    validation_start_date,
)


def run_gap_stage(
    config: GapCorrectionConfig,
    *,
    expected_bond_codes: tuple[str, ...] | None = None,
    run_id: str = "configured",
) -> None:
    """Run one configured gap/EWMA stage and persist its outputs."""

    _validate_config(config)
    input_paths = [
        config.iv_daily_csv,
        config.iv_summary_csv,
        config.iv_selected_csv,
        config.iv_config_csv,
    ]
    if any(path is None for path in input_paths):
        raise ValueError("gap/EWMA IV input paths are not fully configured")
    if config.output_dir is None:
        raise ValueError("gap/EWMA output_dir is not configured")
    resolved_inputs = [Path(path) for path in input_paths if path is not None]
    require_artifacts(resolved_inputs, "IV", run_id)
    daily = pd.read_csv(config.iv_daily_csv)
    iv_summary = pd.read_csv(config.iv_summary_csv)
    selected = pd.read_csv(config.iv_selected_csv)
    iv_config = pd.read_csv(config.iv_config_csv)
    if expected_bond_codes is not None:
        if "bond_code" not in daily.columns:
            raise KeyError(
                f"{config.iv_daily_csv} missing required column: bond_code"
            )
        validate_exact_bond_codes(
            daily["bond_code"].dropna().astype(str).unique(),
            expected_bond_codes,
            str(config.iv_daily_csv),
        )
    if "output_dir" in iv_config.columns and len(iv_config) == 1:
        recorded_iv_dir = Path(str(iv_config.iloc[0]["output_dir"])).resolve()
        actual_iv_dir = Path(config.iv_daily_csv).parent.resolve()
        if recorded_iv_dir != actual_iv_dir:
            raise ValueError(
                "IV config belongs to a different run: "
                f"recorded={recorded_iv_dir}, current={actual_iv_dir}"
            )
    validate_matched_iv_rules(iv_config, config)

    cutoff = validation_start_date(iv_summary)
    backtest = add_gap_correction_forecasts(daily, config)
    gap_summary = summarize_gap_corrections(backtest, cutoff)
    head_to_head = summarize_head_to_head(backtest, selected, cutoff)
    headline = summarize_headline(head_to_head)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(
        config.output_dir / "gap_correction_backtest_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gap_summary.to_csv(
        config.output_dir / "gap_correction_backtest_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    head_to_head.to_csv(
        config.output_dir / "iv_vs_gap_head_to_head.csv",
        index=False,
        encoding="utf-8-sig",
    )
    headline.to_csv(
        config.output_dir / "iv_vs_gap_headline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    config_values = asdict(config)
    for key, value in list(config_values.items()):
        if isinstance(value, Path):
            config_values[key] = str(value)
    config_values["gap_definition"] = "baseline_price_minus_market_price"
    config_values["observation_rule"] = (
        "same_scheduled_dates_as_iv_then_use_from_t_plus_1"
    )
    pd.DataFrame([config_values]).to_csv(
        config.output_dir / "gap_correction_backtest_config.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("IV-versus-gap headline (all IV selection-qualified bonds):")
    print(headline.to_string(index=False))
    print(f"\nSaved outputs to {config.output_dir}")


def main() -> None:
    """Run gap/EWMA for the latest usable batch and IV artifacts."""

    manifest = load_current_run()
    run_gap_stage(
        GapCorrectionConfig.from_manifest(manifest),
        expected_bond_codes=manifest.successful_bond_codes,
        run_id=manifest.paths.run_id,
    )

