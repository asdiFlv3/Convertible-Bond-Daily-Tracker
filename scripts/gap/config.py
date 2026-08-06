"""Configuration and manifest path resolution for gap/EWMA backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipeline_paths import DEFAULT_OUTPUT_ROOT, RunManifest, load_current_run


@dataclass(frozen=True)
class GapCorrectionConfig:
    """Saved IV inputs and rules frozen before the gap benchmark is run."""

    # Gap is deliberately downstream of IV: it reuses IV's daily table,
    # selection decision, validation cutoff, and saved information rules.
    iv_daily_csv: Path | None = None
    iv_summary_csv: Path | None = None
    iv_selected_csv: Path | None = None
    iv_config_csv: Path | None = None
    output_dir: Path | None = None

    # These must match the IV run so neither method receives fresher data.
    calibration_stride: int = 5
    max_gap_age_trading_days: int = 10

    # alpha=1 makes EWMA identical to the latest observed calibration gap.
    ewma_alpha: float = 0.5
    minimum_forecast_coverage: float = 0.80

    @classmethod
    def from_manifest(
        cls,
        manifest: RunManifest,
        **overrides: object,
    ) -> "GapCorrectionConfig":
        """Resolve every IV input and the gap output from one run."""

        values: dict[str, object] = {
            "iv_daily_csv": manifest.paths.iv_daily_csv,
            "iv_summary_csv": manifest.paths.iv_summary_csv,
            "iv_selected_csv": manifest.paths.iv_selected_csv,
            "iv_config_csv": manifest.paths.iv_config_csv,
            "output_dir": manifest.paths.gap_dir,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_current_run(
        cls,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        **overrides: object,
    ) -> "GapCorrectionConfig":
        """Resolve paths from the latest usable Wind-backed batch."""

        return cls.from_manifest(load_current_run(output_root), **overrides)

