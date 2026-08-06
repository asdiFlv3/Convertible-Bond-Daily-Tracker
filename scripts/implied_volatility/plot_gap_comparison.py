"""Charts comparing selected IV forecasts with gap/EWMA rules."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_common import (
    GAP_EWMA_COLOR,
    GAP_LATEST_COLOR,
    GRID_COLOR,
    IV_COLOR,
    _is_true,
    _save_figure,
    _style_axis,
)


def save_iv_vs_gap_mape_chart(
    head_to_head: pd.DataFrame,
    path: Path,
) -> None:
    """Compare baseline, selected IV, and gap rules on identical dates."""

    model_order = ("baseline", "selected_iv", "latest_gap", "ewma_gap")
    model_labels = {
        "baseline": "Historical vol",
        "selected_iv": "Selected lagged IV",
        "latest_gap": "Latest lagged gap",
        "ewma_gap": "EWMA lagged gap",
    }
    colors = {
        "baseline": "#A3A3A3",
        "selected_iv": IV_COLOR,
        "latest_gap": GAP_LATEST_COLOR,
        "ewma_gap": GAP_EWMA_COLOR,
    }
    validation = head_to_head.loc[
        head_to_head["engine"].eq("no_call")
        & head_to_head["segment"].eq("validation")
    ].copy()
    pivot = validation.pivot(
        index="bond_code",
        columns="model",
        values="mape",
    ).dropna(subset=list(model_order))
    if pivot.empty:
        return
    pivot = pivot.sort_values("baseline", ascending=False)
    status = validation.drop_duplicates("bond_code").set_index("bond_code")
    labels = [
        (
            f"{code}"
            if _is_true(status.loc[code, "iv_validation_passed"])
            else f"{code} **"
        )
        for code in pivot.index
    ]

    positions = np.arange(len(pivot))
    bar_height = 0.18
    offsets = (1.5, 0.5, -0.5, -1.5)
    figure, axis = plt.subplots(figsize=(13, 8))
    maximum = 0.0
    for model, offset in zip(model_order, offsets, strict=True):
        values = pivot[model].to_numpy() * 100
        maximum = max(maximum, float(np.nanmax(values)))
        bars = axis.barh(
            positions + offset * bar_height,
            values,
            height=bar_height,
            color=colors[model],
            label=model_labels[model],
        )
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5)

    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Validation MAPE (%) \N{EM DASH} lower is better")
    passed = validation.loc[
        validation["iv_validation_passed"].map(_is_true), "bond_code"
    ].nunique()
    axis.set_title(
        "Same-date validation: selected IV versus simple lagged gaps\n"
        f"All {len(pivot)} IV selection-qualified bonds; "
        f"{passed}/{len(pivot)} pass the IV validation gate",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    axis.legend(loc="lower right")
    _style_axis(axis)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    axis.grid(axis="y", visible=False)
    axis.set_xlim(0, maximum * 1.18)
    figure.text(
        0.01,
        0.01,
        "** IV passed selection but validation coverage is below 80%.  "
        "Each four-bar group uses exactly the same bond-days.\n"
        "Gap observations use the IV calibration schedule, start at t+1, "
        "and expire after 10 usable trading days.",
        fontsize=9,
        color="#4B5563",
    )
    figure.subplots_adjust(bottom=0.15)
    _save_figure(figure, path)


def save_iv_vs_gap_tradeoff_chart(
    headline: pd.DataFrame,
    path: Path,
) -> None:
    """Show accuracy and smoothness for the full pre-qualified sample."""

    model_order = ("baseline", "selected_iv", "latest_gap", "ewma_gap")
    model_labels = {
        "baseline": "Historical vol",
        "selected_iv": "Selected lagged IV",
        "latest_gap": "Latest lagged gap",
        "ewma_gap": "EWMA lagged gap",
    }
    colors = ["#A3A3A3", IV_COLOR, GAP_LATEST_COLOR, GAP_EWMA_COLOR]
    validation = headline.loc[
        headline["engine"].eq("no_call")
        & headline["segment"].eq("validation")
    ].set_index("model")
    if not set(model_order).issubset(validation.index):
        return
    validation = validation.reindex(model_order)
    metrics = (
        ("mean_mape", "Mean validation MAPE (%)"),
        (
            "mean_abs_mean_gap_pct",
            "Mean |average signed gap| (%)",
        ),
        (
            "mean_gap_pct_change_mae",
            "Adjacent-day gap change MAE (%)",
        ),
    )
    labels = [model_labels[model] for model in model_order]
    positions = np.arange(len(model_order))
    figure, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for axis, (column, title) in zip(axes, metrics, strict=True):
        values = validation[column].to_numpy(dtype=float) * 100
        bars = axis.barh(positions, values, color=colors, height=0.62)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=11, pad=10)
        axis.set_xlabel("Lower is better")
        _style_axis(axis)
        axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
        axis.grid(axis="y", visible=False)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5)
        axis.set_xlim(0, max(values) * 1.24)

    eligible = int(validation["eligible_bond_count"].iloc[0])
    passed = int(validation["iv_validation_passed_bond_count"].iloc[0])
    figure.suptitle(
        "Accuracy and smoothness: IV versus low-cost gap correction",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.01,
        0.01,
        f"Equal-weighted across all {eligible} IV selection-qualified bonds; "
        f"{passed}/{eligible} pass the IV validation gate. No validation "
        "failure is removed.\n"
        "All four methods use a per-bond common validation sample. "
        "Level bias is |each bond's average signed gap|, not daily gap MAE.",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.93))
    _save_figure(figure, path)

