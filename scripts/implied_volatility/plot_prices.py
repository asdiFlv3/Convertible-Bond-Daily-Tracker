"""Per-bond IV validation price charts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

from .plot_common import (
    BASELINE_COLOR,
    BATCH_SAMPLE_MARKER,
    IV_COLOR,
    MARKET_COLOR,
    STYLE_LABELS,
    _format_date_axis,
    _qualification_note,
    _save_figure,
    _selection_lookup,
    _selection_style_codes,
    _small_multiple_figure,
    _style_axis,
    _validation_lookup,
)


def save_iv_group_price_chart(
    daily: pd.DataFrame,
    iv_summary: pd.DataFrame,
    selected: pd.DataFrame,
    batch_codes: set[str],
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
        marker = BATCH_SAMPLE_MARKER if code in batch_codes else ""
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
        f"{BATCH_SAMPLE_MARKER} Current batch sample.",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.91))
    _save_figure(figure, path)

