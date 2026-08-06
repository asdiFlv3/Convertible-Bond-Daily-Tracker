"""Configuration and manifest path resolution for IV backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline_paths import DEFAULT_OUTPUT_ROOT, RunManifest, load_current_run


@dataclass(frozen=True)
class BacktestConfig:
    """File locations, IV calibration cadence, and validation thresholds."""

    # Paths are optional only to make programmatic construction convenient;
    # run_iv_stage validates that the three stage paths are present.
    input_csv: Path | None = None
    summary_csv: Path | None = None
    output_dir: Path | None = None

    # Information set used to create out-of-sample daily sigma forecasts.
    calibration_stride: int = 5
    rolling_calibration_count: int = 4
    max_iv_age_trading_days: int = 10
    shrinkage_weight: float = 0.5

    # Model selection is performed before this trailing calendar fraction.
    validation_fraction: float = 0.4

    # Root-finding controls. The grid establishes a reliable bracket before
    # bisection, so it is part of the saved reproducibility configuration.
    sigma_grid: tuple[float, ...] = (
        0.01,
        0.10,
        0.20,
        0.40,
        0.60,
        1.00,
        1.50,
        2.00,
        3.00,
    )
    solver_iterations: int = 24
    solver_price_tolerance: float = 0.01

    # Candidate forecasts below this common-sample coverage are ineligible.
    minimum_forecast_coverage: float = 0.80

    @classmethod
    def from_manifest(
        cls,
        manifest: RunManifest,
        **overrides: object,
    ) -> "BacktestConfig":
        """Resolve all stage paths from one validated batch manifest."""

        values: dict[str, object] = {
            "input_csv": manifest.paths.batch_daily_csv,
            "summary_csv": manifest.paths.batch_summary_csv,
            "output_dir": manifest.paths.iv_dir,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_current_run(
        cls,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        **overrides: object,
    ) -> "BacktestConfig":
        """Resolve paths from the latest usable Wind-backed batch."""

        return cls.from_manifest(load_current_run(output_root), **overrides)

