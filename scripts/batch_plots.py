"""Visualizations for batch convertible-bond CRR comparisons.

The plotting layer accepts ordinary pandas DataFrames and paths.  It does not
import batch configuration/result classes, WindPy, or the pricing model.  This
keeps dependencies one-way and allows charts to be regenerated from saved CSV
files without rerunning Wind requests or CRR valuation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Batch plots are written to PNG files and never require an interactive window.
# Select the backend before importing pyplot so uv Python works without Tcl/Tk.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_latest_gap_chart(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """Save a horizontal chart of latest model-minus-market gaps."""

    if summary.empty:
        return
    plot_data = summary.sort_values("latest_gap_pct")
    figure, axis = plt.subplots(
        figsize=(10, max(4, 0.45 * len(plot_data))),
    )
    colors = np.where(
        plot_data["latest_gap_pct"].ge(0),
        "#2f7d32",
        "#b23a32",
    )
    axis.barh(
        plot_data["bond_code"],
        plot_data["latest_gap_pct"] * 100,
        color=colors,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Latest model-minus-market gap (%)")
    axis.set_ylabel("Bond")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_mape_chart(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """Save per-bond historical MAPE comparison."""

    if summary.empty:
        return
    plot_data = summary.sort_values("mape", ascending=False)
    figure, axis = plt.subplots(
        figsize=(10, max(4, 0.45 * len(plot_data))),
    )
    axis.barh(
        plot_data["bond_code"],
        plot_data["mape"] * 100,
        color="#3f6fa3",
    )
    axis.set_xlabel("Historical MAPE (%)")
    axis.set_ylabel("Bond")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_gap_history_chart(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    max_bonds: int,
) -> None:
    """Save gap histories for at most ``max_bonds`` selected bonds."""

    if daily.empty or summary.empty or max_bonds <= 0:
        return

    # Prefer bonds with the largest absolute latest discrepancy so a large
    # batch does not produce an unreadable chart with dozens of lines.
    selected_codes = (
        summary.assign(
            absolute_latest_gap=lambda frame: frame[
                "latest_gap_pct"
            ].abs()
        )
        .nlargest(max_bonds, "absolute_latest_gap")["bond_code"]
        .tolist()
    )
    plot_data = daily.loc[
        daily["bond_code"].isin(selected_codes)
    ]
    figure, axis = plt.subplots(figsize=(13, 7))
    for code, group in plot_data.groupby(
        "bond_code",
        sort=False,
    ):
        valid = group.dropna(subset=["gap_pct"])
        axis.plot(
            valid["date"],
            valid["gap_pct"] * 100,
            label=str(code),
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Date")
    axis.set_ylabel("Model-minus-market gap (%)")
    axis.legend(ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_price_comparison_chart(
    daily: pd.DataFrame,
    path: Path,
) -> None:
    """Plot market and CRR prices for every successful bond on one axis.

    Each bond receives one color.  Its model price uses a saturated solid line,
    while its market close uses the same color with a dashed, partially
    transparent line.  The pairing remains visible without doubling the number
    of unrelated colors in an already dense ``N * 2`` chart.
    """

    if daily.empty:
        return
    valid = daily.dropna(
        subset=["date", "bond_close", "theoretical_price"],
    )
    if valid.empty:
        return

    codes = valid["bond_code"].drop_duplicates().tolist()
    color_map = plt.get_cmap("tab20")
    figure, axis = plt.subplots(
        figsize=(15, max(7, 0.25 * len(codes) + 6)),
    )
    for index, code in enumerate(codes):
        group = (
            valid.loc[valid["bond_code"].eq(code)]
            .sort_values("date")
        )
        color = color_map(index % color_map.N)
        axis.plot(
            group["date"],
            group["bond_close"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.45,
            color=color,
            label=f"{code} market",
        )
        axis.plot(
            group["date"],
            group["theoretical_price"],
            linestyle="-",
            linewidth=2.0,
            alpha=1.0,
            color=color,
            label=f"{code} CRR",
        )

    axis.set_title("Market vs CRR theoretical prices")
    axis.set_xlabel("Date")
    axis.set_ylabel("Price")
    axis.grid(alpha=0.25)
    # For a large batch, place the 2N-entry legend outside the data region so
    # it does not cover the price paths.
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        ncol=1 if len(codes) <= 10 else 2,
        fontsize="small",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _save_individual_price_chart(
    daily: pd.DataFrame,
    code: str,
    path: Path,
) -> None:
    """Save one bond's market/model price comparison without no-call output."""

    valid = (
        daily.dropna(
            subset=["date", "bond_close", "theoretical_price"],
        )
        .sort_values("date")
    )
    if valid.empty:
        return

    figure, axis = plt.subplots(figsize=(13, 6))
    model_color = "#2369a1"
    axis.plot(
        valid["date"],
        valid["bond_close"],
        linestyle="--",
        linewidth=1.7,
        alpha=0.45,
        color=model_color,
        label="Market bond close",
    )
    axis.plot(
        valid["date"],
        valid["theoretical_price"],
        linestyle="-",
        linewidth=2.2,
        alpha=1.0,
        color=model_color,
        label="CRR theoretical price",
    )
    axis.set_title(f"{code}: market vs CRR theoretical price")
    axis.set_xlabel("Date")
    axis.set_ylabel("Price")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_batch_charts(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    max_bonds_in_history_chart: int,
) -> None:
    """Save all batch and per-bond charts.

    ``individual`` is created here independently of CSV settings because every
    successful bond should receive a price-comparison PNG even when individual
    table export is disabled.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_price_comparison_chart(
        daily,
        output_dir / "price_comparison_all.png",
    )
    _save_latest_gap_chart(
        summary,
        output_dir / "latest_gap_ranking.png",
    )
    _save_mape_chart(
        summary,
        output_dir / "mape_comparison.png",
    )
    _save_gap_history_chart(
        daily,
        summary,
        output_dir / "gap_pct_history.png",
        max_bonds_in_history_chart,
    )

    if daily.empty:
        return
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for code, group in daily.groupby("bond_code", sort=False):
        safe_code = str(code).replace(".", "_")
        _save_individual_price_chart(
            group,
            str(code),
            individual_dir / f"price_comparison_{safe_code}.png",
        )
