"""Generate mentor-ready charts from saved batch and IV backtest outputs.

The script is fully offline.  It preserves the original market-versus-CRR
comparison, but separates the nine bonds into debt-like, balanced and
equity-like groups so high-price equity-like bonds do not flatten the lower
price series.  Current implied-volatility results use only the common
validation window and keep coverage failures visible.

Run from the repository root::

    python scripts/implied_volatility_plots.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib


matplotlib.use("Agg", force=True)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MARKET_COLOR = "#202124"
BASELINE_COLOR = "#7A7A7A"
IV_COLOR = "#0072B2"
CALL_COLOR = "#D55E00"
NO_CALL_COLOR = "#009E73"
GRID_COLOR = "#D9DEE5"
STYLE_ORDER = ("debt", "balanced", "equity")
STYLE_LABELS = {
    "debt": "Debt-like",
    "balanced": "Balanced",
    "equity": "Equity-like",
}
ORIGINAL_SAMPLE_MARKER = "\N{DAGGER}"


@dataclass(frozen=True)
class PlotConfig:
    classic_daily_csv: Path = Path(
        "output/classic/batch/all_9/daily_tracking_all.csv"
    )
    classic_summary_csv: Path = Path(
        "output/classic/batch/all_9/comparison_summary.csv"
    )
    original_four_summary_csv: Path = Path(
        "output/classic/batch/comparison_summary.csv"
    )
    iv_daily_csv: Path = Path(
        "output/diagnostics/implied_volatility/"
        "implied_volatility_backtest_daily.csv"
    )
    iv_summary_csv: Path = Path(
        "output/diagnostics/implied_volatility/"
        "implied_volatility_backtest_summary.csv"
    )
    iv_selected_csv: Path = Path(
        "output/diagnostics/implied_volatility/"
        "implied_volatility_selected_models.csv"
    )
    iv_calibration_csv: Path = Path(
        "output/diagnostics/implied_volatility/"
        "implied_volatility_calibration_diagnostics.csv"
    )
    output_dir: Path = Path(
        "output/diagnostics/implied_volatility/figures"
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    source: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{source} missing columns: {missing}")


def _is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=9)


def _format_date_axis(axis: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _save_figure(figure: plt.Figure, path: Path) -> None:
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
    pd.DataFrame,
    pd.DataFrame,
    set[str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    classic_daily = pd.read_csv(config.classic_daily_csv)
    classic_summary = pd.read_csv(config.classic_summary_csv)
    original_summary = pd.read_csv(config.original_four_summary_csv)
    iv_daily = pd.read_csv(config.iv_daily_csv)
    iv_summary = pd.read_csv(config.iv_summary_csv)
    iv_selected = pd.read_csv(config.iv_selected_csv)
    iv_calibration = pd.read_csv(config.iv_calibration_csv)

    _require_columns(
        classic_daily,
        [
            "date",
            "bond_code",
            "bond_close",
            "theoretical_price",
        ],
        str(config.classic_daily_csv),
    )
    _require_columns(
        classic_summary,
        ["bond_code", "bond_style", "mape"],
        str(config.classic_summary_csv),
    )
    _require_columns(
        original_summary,
        ["bond_code"],
        str(config.original_four_summary_csv),
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

    classic_daily["date"] = pd.to_datetime(classic_daily["date"])
    iv_daily["date"] = pd.to_datetime(iv_daily["date"])
    original_codes = set(original_summary["bond_code"].astype(str))
    if len(original_codes) != 4:
        raise ValueError(
            "original four-bond summary must contain exactly four bonds; "
            f"found {sorted(original_codes)}"
        )
    return (
        classic_daily,
        classic_summary,
        original_codes,
        iv_daily,
        iv_summary,
        iv_selected,
        iv_calibration,
    )


def _style_codes(
    summary: pd.DataFrame,
    style: str,
) -> list[str]:
    return (
        summary.loc[summary["bond_style"].eq(style), "bond_code"]
        .astype(str)
        .sort_values()
        .tolist()
    )


def save_original_group_price_chart(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    original_codes: set[str],
    style: str,
    path: Path,
) -> None:
    """Plot one small multiple per bond for an original CRR style group."""

    codes = _style_codes(summary, style)
    if not codes:
        return
    valid = daily.loc[daily["bond_code"].isin(codes)].dropna(
        subset=["date", "bond_close", "theoretical_price"]
    )
    if valid.empty:
        return

    figure, flat_axes = _small_multiple_figure(len(codes))
    metrics = summary.set_index("bond_code")
    for axis, code in zip(flat_axes, codes, strict=False):
        group = valid.loc[valid["bond_code"].eq(code)].sort_values("date")
        if group.empty:
            continue
        marker = ORIGINAL_SAMPLE_MARKER if code in original_codes else ""
        axis.plot(
            group["date"],
            group["bond_close"],
            color=MARKET_COLOR,
            linewidth=2.0,
            label="Market",
        )
        axis.plot(
            group["date"],
            group["theoretical_price"],
            color=CALL_COLOR,
            linestyle="--",
            linewidth=2.0,
            label="Original CRR",
        )
        axis.fill_between(
            group["date"],
            group["bond_close"],
            group["theoretical_price"],
            color=CALL_COLOR,
            alpha=0.08,
        )
        mape = float(metrics.loc[code, "mape"])
        axis.set_title(
            f"{code}{marker} | original MAPE {mape:.1%}",
            fontsize=11,
            pad=9,
        )
        axis.set_xlabel("Date")
        axis.set_ylabel("Price")
        _style_axis(axis)
        _format_date_axis(axis)

    figure.suptitle(
        f"{STYLE_LABELS[style]} bonds: full-sample market vs original CRR "
        "(simplified call)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=MARKET_COLOR,
                linewidth=2.0,
                label="Market",
            ),
            Line2D(
                [0],
                [0],
                color=CALL_COLOR,
                linestyle="--",
                linewidth=2.0,
                label="Original CRR",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    style_date = valid["date"].max()
    footer = (
        "Each bond has its own linear y-axis; shading is the pricing gap. "
        f"Group uses conversion parity at full-sample end "
        f"({style_date:%Y-%m-%d}). {ORIGINAL_SAMPLE_MARKER} Original "
        "four-bond sample."
    )
    figure.text(
        0.01,
        0.005,
        footer,
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.91))
    _save_figure(figure, path)


def _subplot_shape(count: int) -> tuple[int, int]:
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
    return summary.loc[
        summary["segment"].eq("validation")
        & summary["engine"].eq("no_call")
    ].set_index(["bond_code", "model"])


def _selection_lookup(selected: pd.DataFrame) -> pd.DataFrame:
    return selected.loc[selected["engine"].eq("no_call")].set_index(
        "bond_code"
    )


def _selection_style_codes(
    selected: pd.DataFrame,
    style: str,
) -> list[str]:
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


def save_iv_group_price_chart(
    daily: pd.DataFrame,
    iv_summary: pd.DataFrame,
    selected: pd.DataFrame,
    original_codes: set[str],
    style: str,
    path: Path,
) -> None:
    """Plot validation market, no-call baseline and lagged IV by bond."""

    codes = _selection_style_codes(selected, style)
    if not codes:
        return
    validation_dates = pd.to_datetime(
        iv_summary["validation_start_date"].dropna().unique()
    )
    if len(validation_dates) != 1:
        raise ValueError(
            "expected one common validation start date; "
            f"found {list(validation_dates)}"
        )
    start = pd.Timestamp(validation_dates[0])
    style_dates = pd.to_datetime(
        selected.loc[
            selected["engine"].eq("no_call"),
            "calibration_end_date",
        ],
        errors="coerce",
    ).dropna().unique()
    if len(style_dates) != 1:
        raise ValueError(
            "expected one selection-period style date; "
            f"found {list(style_dates)}"
        )
    style_date = pd.Timestamp(style_dates[0])
    plot_data = daily.loc[
        daily["bond_code"].isin(codes) & daily["date"].ge(start)
    ]
    metrics = _validation_lookup(iv_summary)
    selection = _selection_lookup(selected)
    figure, flat_axes = _small_multiple_figure(len(codes))

    for axis, code in zip(flat_axes, codes, strict=False):
        group = plot_data.loc[plot_data["bond_code"].eq(code)].sort_values(
            "date"
        )
        axis.plot(
            group["date"],
            group["baseline_price_no_call"],
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=1.8,
            label="Historical-vol no-call baseline",
        )
        axis.plot(
            group["date"],
            group["price_iv_latest_calibration_no_call"],
            color=IV_COLOR,
            linewidth=2.0,
            alpha=0.9,
            label="Lagged latest no-call IV",
        )
        axis.plot(
            group["date"],
            group["market_price"],
            color=MARKET_COLOR,
            linewidth=2.1,
            zorder=3,
            label="Market",
        )
        baseline_mape = float(
            metrics.loc[(code, "baseline_price_no_call"), "mape"]
        )
        iv_mape = float(
            metrics.loc[
                (code, "price_iv_latest_calibration_no_call"),
                "mape",
            ]
        )
        selection_row = selection.loc[code] if code in selection.index else None
        marker = ORIGINAL_SAMPLE_MARKER if code in original_codes else ""
        axis.set_title(
            f"{code}{marker} | MAPE {baseline_mape:.1%} "
            f"\N{RIGHTWARDS ARROW} {iv_mape:.1%}\n"
            f"{_qualification_note(selection_row)}",
            fontsize=11,
            pad=9,
        )
        axis.set_xlabel("Date")
        axis.set_ylabel("Price")
        _style_axis(axis)
        _format_date_axis(axis)

    handles = [
        Line2D([0], [0], color=MARKET_COLOR, linewidth=2.1, label="Market"),
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=1.8,
            label="Historical-vol no-call baseline",
        ),
        Line2D(
            [0],
            [0],
            color=IV_COLOR,
            linewidth=2.0,
            label="Lagged latest no-call IV",
        ),
    ]
    figure.suptitle(
        f"{STYLE_LABELS[style]} bonds: validation price comparison",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.01,
        0.005,
        f"Validation starts {start:%Y-%m-%d}. "
        "IV is calibrated every fifth usable day and first used at t+1. "
        "Blue gaps mean no usable lagged IV forecast. "
        f"Style is frozen at {style_date:%Y-%m-%d}. "
        f"{ORIGINAL_SAMPLE_MARKER} Original four-bond sample.",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.91))
    _save_figure(figure, path)


def save_validation_mape_chart(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    path: Path,
) -> None:
    """Compare validation MAPE for the no-call baseline and latest IV."""

    validation = summary.loc[
        summary["segment"].eq("validation")
        & summary["engine"].eq("no_call")
        & summary["model"].isin(
            {
                "baseline_price_no_call",
                "price_iv_latest_calibration_no_call",
            }
        )
    ]
    pivot = validation.pivot(
        index="bond_code",
        columns="model",
        values="mape",
    ).dropna()
    pivot = pivot.sort_values("baseline_price_no_call", ascending=False)
    selection = _selection_lookup(selected)

    labels: list[str] = []
    qualified: list[bool] = []
    for code in pivot.index:
        row = selection.loc[code] if code in selection.index else None
        if row is None or not _is_true(row.get("selection_passed")):
            suffix = " *"
            qualified.append(False)
        elif not _is_true(row.get("validation_passed")):
            suffix = " **"
            qualified.append(False)
        else:
            suffix = ""
            qualified.append(True)
        labels.append(f"{code}{suffix}")

    positions = np.arange(len(pivot))
    bar_height = 0.36
    baseline = pivot["baseline_price_no_call"].to_numpy() * 100
    implied = (
        pivot["price_iv_latest_calibration_no_call"].to_numpy() * 100
    )
    baseline_colors = [
        "#A3A3A3" if passed else "#D1D5DB" for passed in qualified
    ]
    iv_colors = [IV_COLOR if passed else "#A7CFE5" for passed in qualified]
    figure, axis = plt.subplots(figsize=(12, 7))
    baseline_bars = axis.barh(
        positions + bar_height / 2,
        baseline,
        height=bar_height,
        color=baseline_colors,
        label="Historical-vol no-call baseline",
    )
    iv_bars = axis.barh(
        positions - bar_height / 2,
        implied,
        height=bar_height,
        color=iv_colors,
        label="Lagged latest no-call IV",
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Validation MAPE (%) \N{EM DASH} lower is better")
    axis.set_title(
        "Validation MAPE on jointly available dates "
        f"({sum(qualified)}/{len(qualified)} pass all gates)",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    axis.legend(
        handles=[
            Patch(
                facecolor="#A3A3A3",
                label="Historical-vol no-call baseline",
            ),
            Patch(
                facecolor=IV_COLOR,
                label="Lagged latest no-call IV",
            ),
        ],
        loc="lower right",
    )
    _style_axis(axis)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    axis.grid(axis="y", visible=False)
    axis.bar_label(baseline_bars, fmt="%.1f", padding=3, fontsize=8)
    axis.bar_label(iv_bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set_xlim(0, max(baseline.max(), implied.max()) * 1.16)
    figure.text(
        0.01,
        0.01,
        "* Fails the selection-period coverage gate.  "
        "** Passes selection but validation coverage is below 80%.  "
        "Failed-gate bonds are faded; each pair uses identical dates.\n"
        "Both bars use the no-call engine and differ only in volatility "
        "input; do not compare them directly with full-sample original-CRR "
        "MAPE.",
        fontsize=9,
        color="#4B5563",
    )
    figure.subplots_adjust(bottom=0.15)
    _save_figure(figure, path)


def save_solver_success_chart(
    calibration: pd.DataFrame,
    original_codes: set[str],
    path: Path,
) -> None:
    """Compare call and no-call IV solver success shares."""

    pivot = calibration.pivot(
        index="bond_code",
        columns="engine",
        values="solution_share",
    ).dropna()
    pivot = pivot.sort_values("iv_no_call", ascending=False)
    positions = np.arange(len(pivot))
    bar_height = 0.36
    call = pivot["iv_call"].to_numpy() * 100
    no_call = pivot["iv_no_call"].to_numpy() * 100
    labels = [
        f"{code}{ORIGINAL_SAMPLE_MARKER if code in original_codes else ''}"
        for code in pivot.index
    ]

    figure, axis = plt.subplots(figsize=(12, 7))
    call_bars = axis.barh(
        positions + bar_height / 2,
        call,
        height=bar_height,
        color=CALL_COLOR,
        label="With simplified call",
    )
    no_call_bars = axis.barh(
        positions - bar_height / 2,
        no_call,
        height=bar_height,
        color=NO_CALL_COLOR,
        label="No-call engine",
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 108)
    axis.set_xlabel("Calibration dates with a valid IV solution (%)")
    axis.set_title(
        "No-call IV inversion is materially more reliable",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    axis.legend(loc="lower right")
    _style_axis(axis)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    axis.grid(axis="y", visible=False)
    axis.bar_label(call_bars, fmt="%.0f", padding=3, fontsize=8)
    axis.bar_label(no_call_bars, fmt="%.0f", padding=3, fontsize=8)
    figure.text(
        0.01,
        0.01,
        f"{ORIGINAL_SAMPLE_MARKER} Original four-bond sample. "
        "With-call failures are mostly "
        "non-monotonic price curves or unreachable market prices.",
        fontsize=9,
        color="#4B5563",
    )
    figure.subplots_adjust(bottom=0.12)
    _save_figure(figure, path)


def save_strategy_tradeoff_chart(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    path: Path,
) -> None:
    """Show accuracy, residual bias and day-to-day smoothness trade-offs."""

    selected_no_call = selected.loc[selected["engine"].eq("no_call")]
    passed_codes = selected_no_call.loc[
        selected_no_call["validation_passed"].map(_is_true),
        "bond_code",
    ].tolist()
    if not passed_codes:
        return

    model_labels = {
        "baseline_price_no_call": "Historical vol",
        "price_iv_shrunk_to_historical_no_call": "50% shrinkage",
        "price_iv_median_recent_calibrations_no_call": "Recent IV median",
        "price_iv_latest_calibration_no_call": "Latest IV",
    }
    validation = summary.loc[
        summary["segment"].eq("validation")
        & summary["engine"].eq("no_call")
        & summary["bond_code"].isin(passed_codes)
        & summary["model"].isin(model_labels)
    ].copy()
    grouped = validation.groupby("model").agg(
        mean_mape=("mape", "mean"),
        mean_abs_bias=("mean_gap_pct", lambda values: values.abs().mean()),
        mean_gap_change=("gap_pct_change_mae", "mean"),
    )
    order = list(model_labels)
    grouped = grouped.reindex(order)
    display_labels = [model_labels[model] for model in order]
    colors = ["#A3A3A3", "#8EC5E8", "#3F8FC4", "#00689D"]
    metrics = (
        ("mean_mape", "Mean validation MAPE (%)"),
        ("mean_abs_bias", "Mean |average signed gap| (%)"),
        ("mean_gap_change", "Adjacent-day gap change MAE (%)"),
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    positions = np.arange(len(order))
    for axis, (column, title) in zip(axes, metrics, strict=True):
        values = grouped[column].to_numpy() * 100
        bars = axis.barh(positions, values, color=colors, height=0.62)
        axis.set_yticks(positions, display_labels)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=11, pad=10)
        axis.set_xlabel("Lower is better")
        _style_axis(axis)
        axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
        axis.grid(axis="y", visible=False)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5)
        axis.set_xlim(0, max(values) * 1.24)

    figure.suptitle(
        "Validation trade-off: IV reduces level bias but raises "
        "adjacent-day gap variation",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.01,
        0.01,
        f"Equal-weighted across {len(passed_codes)} bonds passing the "
        "validation-improvement and 80% coverage gates.\n"
        "Level bias = mean across bonds of |each bond's average signed gap|; "
        "it is not daily gap MAE.",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.93))
    _save_figure(figure, path)


def generate_plots(config: PlotConfig) -> list[Path]:
    (
        classic_daily,
        classic_summary,
        original_codes,
        iv_daily,
        iv_summary,
        iv_selected,
        iv_calibration,
    ) = _load_inputs(config)

    output_paths: list[Path] = []
    for style in STYLE_ORDER:
        original_path = (
            config.output_dir / f"01_original_market_vs_crr_{style}.png"
        )
        save_original_group_price_chart(
            classic_daily,
            classic_summary,
            original_codes,
            style,
            original_path,
        )
        output_paths.append(original_path)

        iv_path = config.output_dir / f"02_validation_market_vs_iv_{style}.png"
        save_iv_group_price_chart(
            iv_daily,
            iv_summary,
            iv_selected,
            original_codes,
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
        original_codes,
        solver_path,
    )
    output_paths.append(solver_path)

    tradeoff_path = config.output_dir / "05_accuracy_smoothness_tradeoff.png"
    save_strategy_tradeoff_chart(iv_summary, iv_selected, tradeoff_path)
    output_paths.append(tradeoff_path)
    return output_paths


def main() -> None:
    paths = generate_plots(PlotConfig())
    print("Saved implied-volatility figures:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
