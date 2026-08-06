"""Cross-bond IV validation and solver summary charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .plot_common import (
    BASELINE_COLOR,
    BATCH_SAMPLE_MARKER,
    CALL_COLOR,
    GRID_COLOR,
    IV_COLOR,
    NO_CALL_COLOR,
    _is_true,
    _save_figure,
    _selection_lookup,
    _style_axis,
)


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
    batch_codes: set[str],
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
        f"{code}{BATCH_SAMPLE_MARKER if code in batch_codes else ''}"
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
        f"{BATCH_SAMPLE_MARKER} Current batch sample. "
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

