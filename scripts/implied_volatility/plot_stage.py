"""Compose the figure suite for one manifest-selected pipeline run."""

from __future__ import annotations

from pathlib import Path

from pipeline_paths import load_current_run
from .plot_common import PlotConfig, STYLE_ORDER, _load_inputs
from .plot_gap_comparison import (
    save_iv_vs_gap_mape_chart,
    save_iv_vs_gap_tradeoff_chart,
)
from .plot_prices import save_iv_group_price_chart
from .plot_summaries import (
    save_solver_success_chart,
    save_strategy_tradeoff_chart,
    save_validation_mape_chart,
)


def generate_plots(config: PlotConfig) -> list[Path]:
    """Generate the complete figure suite and return paths in display order."""

    (
        batch_codes,
        iv_daily,
        iv_summary,
        iv_selected,
        iv_calibration,
        gap_head_to_head,
        gap_headline,
    ) = _load_inputs(config)

    # Keep filenames and return order aligned with the intended review flow:
    # daily behavior, IV diagnostics, then IV-versus-gap comparison.
    output_paths: list[Path] = []
    for style in STYLE_ORDER:
        iv_path = config.output_dir / f"02_validation_market_vs_iv_{style}.png"
        save_iv_group_price_chart(
            iv_daily,
            iv_summary,
            iv_selected,
            batch_codes,
            style,
            iv_path,
        )
        output_paths.append(iv_path)

    mape_path = config.output_dir / "03_validation_mape_all_bonds.png"
    save_validation_mape_chart(iv_summary, iv_selected, mape_path)
    output_paths.append(mape_path)

    solver_path = config.output_dir / "04_iv_solver_success_by_engine.png"
    save_solver_success_chart(
        iv_calibration,
        batch_codes,
        solver_path,
    )
    output_paths.append(solver_path)

    tradeoff_path = config.output_dir / "05_accuracy_smoothness_tradeoff.png"
    save_strategy_tradeoff_chart(iv_summary, iv_selected, tradeoff_path)
    output_paths.append(tradeoff_path)

    gap_mape_path = config.output_dir / "06_validation_iv_vs_gap_mape.png"
    save_iv_vs_gap_mape_chart(gap_head_to_head, gap_mape_path)
    output_paths.append(gap_mape_path)

    gap_tradeoff_path = (
        config.output_dir / "07_iv_vs_gap_accuracy_smoothness.png"
    )
    save_iv_vs_gap_tradeoff_chart(gap_headline, gap_tradeoff_path)
    output_paths.append(gap_tradeoff_path)
    return output_paths


def main() -> None:
    """Generate figures for the latest usable pipeline run."""

    manifest = load_current_run()
    paths = generate_plots(PlotConfig.from_manifest(manifest))
    print("Saved diagnostic figures:")
    for path in paths:
        print(f"- {path}")
