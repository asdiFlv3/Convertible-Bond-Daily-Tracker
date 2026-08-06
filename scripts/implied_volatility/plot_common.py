"""Plot configuration, input validation, and shared chart helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_paths import (
    DEFAULT_OUTPUT_ROOT,
    RunManifest,
    load_current_run,
    require_artifacts,
    validate_exact_bond_codes,
)


MARKET_COLOR = "#202124"
BASELINE_COLOR = "#7A7A7A"
IV_COLOR = "#0072B2"
GAP_LATEST_COLOR = "#CC79A7"
GAP_EWMA_COLOR = "#E69F00"
CALL_COLOR = "#D55E00"
NO_CALL_COLOR = "#009E73"
GRID_COLOR = "#D9DEE5"
STYLE_ORDER = ("debt", "balanced", "equity")
STYLE_LABELS = {
    "debt": "Debt-like",
    "balanced": "Balanced",
    "equity": "Equity-like",
}
BATCH_SAMPLE_MARKER = "\N{DAGGER}"


@dataclass(frozen=True)
class PlotConfig:
    """Locations of saved analysis tables and generated figure files."""

    batch_summary_csv: Path | None = None
    iv_daily_csv: Path | None = None
    iv_summary_csv: Path | None = None
    iv_selected_csv: Path | None = None
    iv_calibration_csv: Path | None = None
    gap_head_to_head_csv: Path | None = None
    gap_headline_csv: Path | None = None
    output_dir: Path | None = None

    @classmethod
    def from_manifest(
        cls,
        manifest: RunManifest,
        **overrides: object,
    ) -> "PlotConfig":
        """Resolve all plot inputs and outputs from one pipeline run."""

        values: dict[str, object] = {
            "batch_summary_csv": manifest.paths.batch_summary_csv,
            "iv_daily_csv": manifest.paths.iv_daily_csv,
            "iv_summary_csv": manifest.paths.iv_summary_csv,
            "iv_selected_csv": manifest.paths.iv_selected_csv,
            "iv_calibration_csv": manifest.paths.iv_calibration_csv,
            "gap_head_to_head_csv": manifest.paths.gap_head_to_head_csv,
            "gap_headline_csv": manifest.paths.gap_headline_csv,
            "output_dir": manifest.paths.figures_dir,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_current_run(
        cls,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        **overrides: object,
    ) -> "PlotConfig":
        """Resolve paths from the latest usable Wind-backed batch."""

        return cls.from_manifest(load_current_run(output_root), **overrides)


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    source: str,
) -> None:
    """Raise a source-aware error when a plotting input lacks columns."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{source} missing columns: {missing}")


def _is_true(value: object) -> bool:
    """Interpret boolean values that may have round-tripped through CSV."""

    return str(value).strip().lower() == "true"


def _style_axis(axis: plt.Axes) -> None:
    """Apply the common unobtrusive grid and spine styling to an axis."""

    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=9)


def _format_date_axis(axis: plt.Axes) -> None:
    """Use concise, automatically spaced calendar labels on an axis."""

    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Create the destination, save a consistent PNG, and release memory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=190,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _load_inputs(
    config: PlotConfig,
) -> tuple[
    set[str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load and validate every saved table needed by the figure suite."""

    input_paths = [
        config.batch_summary_csv,
        config.iv_daily_csv,
        config.iv_summary_csv,
        config.iv_selected_csv,
        config.iv_calibration_csv,
        config.gap_head_to_head_csv,
        config.gap_headline_csv,
    ]
    if any(path is None for path in input_paths):
        raise ValueError("plot input paths are not fully configured")
    if config.output_dir is None:
        raise ValueError("plot output_dir is not configured")
    require_artifacts(
        [Path(path) for path in input_paths if path is not None],
        "Plot input",
        config.output_dir.parent.name,
    )
    batch_summary = pd.read_csv(config.batch_summary_csv)
    iv_daily = pd.read_csv(config.iv_daily_csv)
    iv_summary = pd.read_csv(config.iv_summary_csv)
    iv_selected = pd.read_csv(config.iv_selected_csv)
    iv_calibration = pd.read_csv(config.iv_calibration_csv)
    gap_head_to_head = pd.read_csv(config.gap_head_to_head_csv)
    gap_headline = pd.read_csv(config.gap_headline_csv)

    _require_columns(
        batch_summary,
        ["bond_code"],
        str(config.batch_summary_csv),
    )
    _require_columns(
        iv_daily,
        [
            "date",
            "bond_code",
            "market_price",
            "baseline_price_no_call",
            "price_iv_latest_calibration_no_call",
        ],
        str(config.iv_daily_csv),
    )
    _require_columns(
        iv_summary,
        [
            "bond_code",
            "engine",
            "segment",
            "model",
            "mape",
            "mean_gap_pct",
            "gap_pct_change_mae",
            "comparison_coverage",
            "validation_start_date",
        ],
        str(config.iv_summary_csv),
    )
    _require_columns(
        iv_selected,
        [
            "bond_code",
            "engine",
            "selection_style",
            "selection_passed",
            "validation_passed",
            "calibration_end_date",
        ],
        str(config.iv_selected_csv),
    )
    _require_columns(
        iv_calibration,
        ["bond_code", "engine", "solution_share"],
        str(config.iv_calibration_csv),
    )
    _require_columns(
        gap_head_to_head,
        [
            "bond_code",
            "engine",
            "segment",
            "model",
            "mape",
            "comparison_coverage",
            "iv_validation_passed",
        ],
        str(config.gap_head_to_head_csv),
    )
    _require_columns(
        gap_headline,
        [
            "engine",
            "segment",
            "model",
            "eligible_bond_count",
            "iv_validation_passed_bond_count",
            "mean_mape",
            "mean_abs_mean_gap_pct",
            "mean_gap_pct_change_mae",
        ],
        str(config.gap_headline_csv),
    )

    iv_daily["date"] = pd.to_datetime(iv_daily["date"])
    batch_codes = set(batch_summary["bond_code"].astype(str))
    validate_exact_bond_codes(
        iv_daily["bond_code"].dropna().astype(str).unique(),
        batch_codes,
        str(config.iv_daily_csv),
    )
    return (
        batch_codes,
        iv_daily,
        iv_summary,
        iv_selected,
        iv_calibration,
        gap_head_to_head,
        gap_headline,
    )


def _subplot_shape(count: int) -> tuple[int, int]:
    """Choose a compact two-column grid for a number of bond panels."""

    if count <= 2:
        return 1, count
    return int(np.ceil(count / 2)), 2


def _small_multiple_figure(
    count: int,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Create equal-width panels, centering the fifth panel when needed."""

    if count == 5:
        figure = plt.figure(figsize=(15, 14.4))
        grid = figure.add_gridspec(3, 4)
        slots = (
            grid[0, 0:2],
            grid[0, 2:4],
            grid[1, 0:2],
            grid[1, 2:4],
            grid[2, 1:3],
        )
        return figure, [figure.add_subplot(slot) for slot in slots]

    rows, columns = _subplot_shape(count)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 4.8 * rows),
        squeeze=False,
    )
    flat_axes = list(axes.ravel())
    for axis in flat_axes[count:]:
        axis.set_visible(False)
    return figure, flat_axes[:count]


def _validation_lookup(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Index no-call validation metrics for bond/model lookup."""

    return summary.loc[
        summary["segment"].eq("validation")
        & summary["engine"].eq("no_call")
    ].set_index(["bond_code", "model"])


def _selection_lookup(selected: pd.DataFrame) -> pd.DataFrame:
    """Index no-call model-selection outcomes by bond code."""

    return selected.loc[selected["engine"].eq("no_call")].set_index(
        "bond_code"
    )


def _selection_style_codes(
    selected: pd.DataFrame,
    style: str,
) -> list[str]:
    """Return bonds grouped by style frozen at the selection cutoff."""

    return (
        selected.loc[
            selected["engine"].eq("no_call")
            & selected["selection_style"].eq(style),
            "bond_code",
        ]
        .astype(str)
        .sort_values()
        .tolist()
    )


def _qualification_note(row: pd.Series | None) -> str:
    """Explain selection or validation eligibility in chart-friendly text."""

    if row is None:
        return "no selection record"
    if not _is_true(row.get("selection_passed")):
        coverage = pd.to_numeric(
            row.get("maximum_calibration_comparison_coverage"),
            errors="coerce",
        )
        if np.isfinite(coverage):
            return f"selection coverage {coverage:.0%} < 80%"
        return "selection gate failed"
    if not _is_true(row.get("validation_passed")):
        coverage = pd.to_numeric(
            row.get("validation_comparison_coverage"),
            errors="coerce",
        )
        if np.isfinite(coverage):
            return f"validation coverage {coverage:.0%} < 80%"
        return "validation gate failed"
    return "passes validation"

