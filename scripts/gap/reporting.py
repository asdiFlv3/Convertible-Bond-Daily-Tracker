"""Validation summaries comparing IV, latest-gap, and EWMA forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .correction import (
    BASELINE_COLUMNS_BY_ENGINE,
    EWMA_GAP_PRICE_BY_ENGINE,
    HEAD_TO_HEAD_MODEL_ORDER,
    LATEST_GAP_PRICE_BY_ENGINE,
    _is_true,
    _require_columns,
)


def validation_start_date(iv_summary: pd.DataFrame) -> pd.Timestamp:
    """Read the single frozen validation cutoff from the IV summary."""

    _require_columns(
        iv_summary,
        ["validation_start_date"],
        "IV backtest summary",
    )
    dates = pd.to_datetime(
        iv_summary["validation_start_date"], errors="coerce"
    ).dropna().unique()
    if len(dates) != 1:
        raise ValueError(
            "expected one frozen validation start date; "
            f"found {list(dates)}"
        )
    return pd.Timestamp(dates[0])


def _metric_values(
    subset: pd.DataFrame,
    model_column: str,
    common_sample: pd.Series,
) -> dict[str, object]:
    """Calculate coverage, error level, and adjacent-day variation."""

    market = pd.to_numeric(subset["market_price"], errors="coerce")
    model = pd.to_numeric(subset[model_column], errors="coerce")
    usable_market = market.notna() & market.ne(0)
    available = model.notna() & usable_market
    comparison = common_sample & model.notna()
    if comparison.any():
        gap_pct = (model.loc[comparison] - market.loc[comparison]) / market.loc[
            comparison
        ]
        comparison_positions = pd.Series(
            np.flatnonzero(comparison.to_numpy()),
            index=gap_pct.index,
        )
        # Smoothness uses only adjacent source rows. Missing forecast days do
        # not get bridged into a misleading multi-day "daily" change.
        consecutive = comparison_positions.diff().eq(1)
        gap_change = gap_pct.diff().abs()
        change_mae = gap_change.loc[consecutive].mean()
        change_pairs = int(consecutive.sum())
        mape = gap_pct.abs().mean()
        mean_gap = gap_pct.mean()
        median_gap = gap_pct.median()
        gap_std = gap_pct.std(ddof=1)
    else:
        mape = np.nan
        mean_gap = np.nan
        median_gap = np.nan
        gap_std = np.nan
        change_mae = np.nan
        change_pairs = 0
    return {
        "total_days": len(subset),
        "priced_days": int(available.sum()),
        "comparison_days": int(comparison.sum()),
        "raw_coverage": available.mean() if len(available) else np.nan,
        "comparison_coverage": (
            comparison.mean() if len(comparison) else np.nan
        ),
        "mape": mape,
        "mean_gap_pct": mean_gap,
        "median_gap_pct": median_gap,
        "gap_pct_std": gap_std,
        "gap_pct_change_mae": change_mae,
        "gap_pct_change_pairs": change_pairs,
    }


def _segments(
    ordered: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[tuple[str, pd.DataFrame], ...]:
    """Return the frozen selection and validation calendar segments."""

    return (
        ("calibration", ordered.loc[ordered["date"].lt(cutoff)]),
        ("validation", ordered.loc[ordered["date"].ge(cutoff)]),
    )


def summarize_gap_corrections(
    backtest: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Evaluate baseline and gap corrections on a gap-only common sample."""

    rows: list[dict[str, object]] = []
    for code, group in backtest.groupby("bond_code", sort=False):
        ordered = group.sort_values("date")
        calibration = ordered.loc[ordered["date"].lt(cutoff)]
        selection_style = (
            calibration["bond_style"].iloc[-1]
            if not calibration.empty
            else np.nan
        )
        for segment, subset in _segments(ordered, cutoff):
            market = pd.to_numeric(
                subset["market_price"], errors="coerce"
            )
            for engine, baseline in BASELINE_COLUMNS_BY_ENGINE.items():
                model_columns = (
                    baseline,
                    LATEST_GAP_PRICE_BY_ENGINE[engine],
                    EWMA_GAP_PRICE_BY_ENGINE[engine],
                )
                common = (
                    market.notna()
                    & market.ne(0)
                    & subset[list(model_columns)].notna().all(axis=1)
                )
                for model in model_columns:
                    rows.append(
                        {
                            "bond_code": code,
                            "selection_style": selection_style,
                            "engine": engine,
                            "segment": segment,
                            "segment_start_date": (
                                subset["date"].min()
                                if not subset.empty
                                else pd.NaT
                            ),
                            "segment_end_date": (
                                subset["date"].max()
                                if not subset.empty
                                else pd.NaT
                            ),
                            "validation_start_date": cutoff,
                            "model": model,
                            **_metric_values(subset, model, common),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_head_to_head(
    backtest: pd.DataFrame,
    selected: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Compare selected IV and gap rules on exactly the same bond-days."""

    required_selected = [
        "bond_code",
        "engine",
        "selection_style",
        "selection_passed",
        "selected_model",
        "validation_passed",
        "validation_comparison_coverage",
    ]
    _require_columns(selected, required_selected, "IV selected models")
    duplicated = selected.duplicated(["bond_code", "engine"], keep=False)
    if duplicated.any():
        raise ValueError("IV selected models contain duplicate bond/engine rows")

    rows: list[dict[str, object]] = []
    eligible = selected.loc[selected["selection_passed"].map(_is_true)]
    # Qualification is frozen by IV calibration results. Gap performance never
    # decides which bonds enter this comparison.
    for selection in eligible.itertuples(index=False):
        code = str(selection.bond_code)
        engine = str(selection.engine)
        if engine not in BASELINE_COLUMNS_BY_ENGINE:
            raise ValueError(f"unknown selected engine {engine!r}")
        selected_iv = str(selection.selected_model)
        model_columns = {
            "baseline": BASELINE_COLUMNS_BY_ENGINE[engine],
            "selected_iv": selected_iv,
            "latest_gap": LATEST_GAP_PRICE_BY_ENGINE[engine],
            "ewma_gap": EWMA_GAP_PRICE_BY_ENGINE[engine],
        }
        group = backtest.loc[backtest["bond_code"].astype(str).eq(code)].copy()
        if group.empty:
            raise ValueError(f"gap backtest has no rows for selected bond {code}")
        _require_columns(
            group,
            list(model_columns.values()),
            f"gap backtest for {code}",
        )
        ordered = group.sort_values("date")
        for segment, subset in _segments(ordered, cutoff):
            market = pd.to_numeric(
                subset["market_price"], errors="coerce"
            )
            common = (
                market.notna()
                & market.ne(0)
                & subset[list(model_columns.values())].notna().all(axis=1)
            )
            for model in HEAD_TO_HEAD_MODEL_ORDER:
                source = model_columns[model]
                rows.append(
                    {
                        "bond_code": code,
                        "selection_style": selection.selection_style,
                        "engine": engine,
                        "segment": segment,
                        "segment_start_date": (
                            subset["date"].min()
                            if not subset.empty
                            else pd.NaT
                        ),
                        "segment_end_date": (
                            subset["date"].max()
                            if not subset.empty
                            else pd.NaT
                        ),
                        "validation_start_date": cutoff,
                        "model": model,
                        "source_model": source,
                        "selected_iv_model": selected_iv,
                        "iv_validation_passed": (
                            selection.validation_passed
                        ),
                        "iv_validation_comparison_coverage": (
                            selection.validation_comparison_coverage
                        ),
                        **_metric_values(subset, source, common),
                    }
                )
    return pd.DataFrame(rows)


def summarize_headline(head_to_head: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight the full set of IV-selection-qualified no-call bonds."""

    validation = head_to_head.loc[
        head_to_head["engine"].eq("no_call")
        & head_to_head["segment"].eq("validation")
    ]
    eligible_bonds = validation["bond_code"].nunique()
    validation_passed = validation.loc[
        validation["iv_validation_passed"].map(_is_true), "bond_code"
    ].nunique()
    rows: list[dict[str, object]] = []
    for model in HEAD_TO_HEAD_MODEL_ORDER:
        group = validation.loc[validation["model"].eq(model)]
        rows.append(
            {
                "sample": "all_iv_selection_passed_bonds",
                "engine": "no_call",
                "segment": "validation",
                "model": model,
                "eligible_bond_count": eligible_bonds,
                "metric_bond_count": int(group["mape"].notna().sum()),
                "iv_validation_passed_bond_count": validation_passed,
                "mean_comparison_coverage": group[
                    "comparison_coverage"
                ].mean(),
                "mean_mape": group["mape"].mean(),
                "mean_gap_pct": group["mean_gap_pct"].mean(),
                "mean_abs_mean_gap_pct": group["mean_gap_pct"].abs().mean(),
                "mean_gap_pct_std": group["gap_pct_std"].mean(),
                "mean_gap_pct_change_mae": group[
                    "gap_pct_change_mae"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)

