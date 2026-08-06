"""Calibration-period IV model selection and validation lookup."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest_config import BacktestConfig
from .backtest_engine import FORECAST_COLUMNS


def _matched_baseline(model: str) -> str:
    """Return the historical-volatility baseline for a forecast column."""

    if model.endswith("_with_call"):
        return "baseline_price_with_call"
    if model.endswith("_no_call"):
        return "baseline_price_no_call"
    raise ValueError(f"cannot match a baseline to model {model!r}")


def select_models(
    summary: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Select on calibration dates, then evaluate untouched validation dates."""

    forecast_models = set(FORECAST_COLUMNS)
    calibration_all = summary.loc[
        summary["segment"].eq("calibration")
    ].copy()
    # Validation rows are lookup-only: 
    # no validation metric participates in coverage gating or candidate ranking.
    validation = summary.loc[summary["segment"].eq("validation")].copy()
    candidates = calibration_all.loc[
        calibration_all["model"].isin(forecast_models)
        & calibration_all["mape"].notna()
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates["baseline_model"] = candidates["model"].map(
        _matched_baseline
    )
    baseline_calibration = calibration_all.loc[
        calibration_all["model"].isin(
            {"baseline_price_with_call", "baseline_price_no_call"}
        ),
        [
            "bond_code",
            "engine",
            "model",
            "mape",
            "mean_gap_pct",
            "gap_pct_std",
            "gap_pct_change_mae",
        ],
    ].rename(
        columns={
            "model": "baseline_model",
            "mape": "calibration_baseline_mape",
            "mean_gap_pct": "calibration_baseline_mean_gap_pct",
            "gap_pct_std": "calibration_baseline_gap_pct_std",
            "gap_pct_change_mae": (
                "calibration_baseline_gap_pct_change_mae"
            ),
        }
    )
    candidates = candidates.merge(
        baseline_calibration,
        on=["bond_code", "engine", "baseline_model"],
        how="left",
        validate="many_to_one",
    )
    candidates["calibration_mape_improvement"] = (
        candidates["calibration_baseline_mape"] - candidates["mape"]
    )

    validation_lookup = validation.set_index(
        ["bond_code", "engine", "model"]
    )
    result_rows: list[dict[str, object]] = []
    engines = summary[["bond_code", "engine"]].drop_duplicates()
    for code, engine in engines.itertuples(index=False, name=None):
        bond_candidates = candidates.loc[
            candidates["bond_code"].eq(code)
            & candidates["engine"].eq(engine)
        ]
        eligible = bond_candidates.loc[
            bond_candidates["raw_coverage"].ge(
                config.minimum_forecast_coverage
            )
            & bond_candidates["comparison_coverage"].ge(
                config.minimum_forecast_coverage
            )
        ]
        if eligible.empty:
            style_values = bond_candidates["selection_style"].dropna()
            result_rows.append(
                {
                    "bond_code": code,
                    "engine": engine,
                    "selection_style": (
                        style_values.iloc[0]
                        if not style_values.empty
                        else np.nan
                    ),
                    "selection_status": "no_forecast_meets_coverage",
                    "selection_passed": False,
                    "minimum_forecast_coverage": (
                        config.minimum_forecast_coverage
                    ),
                    "maximum_calibration_comparison_coverage": (
                        bond_candidates["comparison_coverage"].max()
                    ),
                }
            )
            continue

        # Prefer forecasts that beat their engine-matched baseline. If none do,
        # retain the best covered candidate but mark selection as failed.
        improving = eligible.loc[
            eligible["calibration_mape_improvement"].gt(0)
        ]
        selection_passed = not improving.empty
        pool = improving if selection_passed else eligible
        best = pool.sort_values(["mape", "model"], kind="stable").iloc[0]
        selected_model = str(best["model"])
        baseline_model = str(best["baseline_model"])
        row: dict[str, object] = {
            "bond_code": code,
            "engine": engine,
            "selection_style": best["selection_style"],
            "selection_status": (
                "selected_on_calibration"
                if selection_passed
                else "best_candidate_does_not_beat_baseline"
            ),
            "selection_passed": selection_passed,
            "minimum_forecast_coverage": (
                config.minimum_forecast_coverage
            ),
            "selected_model": selected_model,
            "baseline_model": baseline_model,
            "calibration_mape": best["mape"],
            "calibration_start_date": best["segment_start_date"],
            "calibration_end_date": best["segment_end_date"],
            "calibration_baseline_mape": best[
                "calibration_baseline_mape"
            ],
            "calibration_baseline_mean_gap_pct": best[
                "calibration_baseline_mean_gap_pct"
            ],
            "calibration_baseline_gap_pct_std": best[
                "calibration_baseline_gap_pct_std"
            ],
            "calibration_baseline_gap_pct_change_mae": best[
                "calibration_baseline_gap_pct_change_mae"
            ],
            "calibration_mape_improvement": best[
                "calibration_mape_improvement"
            ],
            "calibration_relative_mape_improvement": (
                best["calibration_mape_improvement"]
                / best["calibration_baseline_mape"]
                if best["calibration_baseline_mape"] > 0
                else np.nan
            ),
            "calibration_raw_coverage": best["raw_coverage"],
            "calibration_comparison_coverage": best[
                "comparison_coverage"
            ],
            "calibration_mean_gap_pct": best["mean_gap_pct"],
            "calibration_gap_pct_std": best["gap_pct_std"],
            "calibration_gap_pct_change_mae": best[
                "gap_pct_change_mae"
            ],
        }

        selected_key = (code, engine, selected_model)
        baseline_key = (code, engine, baseline_model)
        selected_validation = (
            validation_lookup.loc[selected_key]
            if selected_key in validation_lookup.index
            else None
        )
        baseline_validation = (
            validation_lookup.loc[baseline_key]
            if baseline_key in validation_lookup.index
            else None
        )
        if selected_validation is not None:
            row.update(
                {
                    "validation_mape": selected_validation["mape"],
                    "validation_start_date": selected_validation[
                        "segment_start_date"
                    ],
                    "validation_end_date": selected_validation[
                        "segment_end_date"
                    ],
                    "validation_mean_gap_pct": selected_validation[
                        "mean_gap_pct"
                    ],
                    "validation_gap_pct_std": selected_validation[
                        "gap_pct_std"
                    ],
                    "validation_gap_pct_change_mae": selected_validation[
                        "gap_pct_change_mae"
                    ],
                    "validation_raw_coverage": selected_validation[
                        "raw_coverage"
                    ],
                    "validation_comparison_coverage": selected_validation[
                        "comparison_coverage"
                    ],
                }
            )
        if baseline_validation is not None:
            row.update(
                {
                    "validation_baseline_mape": baseline_validation["mape"],
                    "validation_baseline_mean_gap_pct": (
                        baseline_validation["mean_gap_pct"]
                    ),
                    "validation_baseline_gap_pct_std": baseline_validation[
                        "gap_pct_std"
                    ],
                    "validation_baseline_gap_pct_change_mae": (
                        baseline_validation["gap_pct_change_mae"]
                    ),
                }
            )

        validation_improvement = (
            row.get("validation_baseline_mape", np.nan)
            - row.get("validation_mape", np.nan)
        )
        row["validation_mape_improvement"] = validation_improvement
        validation_baseline_mape = row.get(
            "validation_baseline_mape",
            np.nan,
        )
        row["validation_relative_mape_improvement"] = (
            validation_improvement / validation_baseline_mape
            if np.isfinite(validation_baseline_mape)
            and validation_baseline_mape > 0
            else np.nan
        )
        row["validation_improves_baseline"] = (
            bool(validation_improvement > 0)
            if np.isfinite(validation_improvement)
            else pd.NA
        )
        validation_coverage = row.get(
            "validation_comparison_coverage",
            np.nan,
        )
        row["validation_meets_coverage"] = (
            bool(validation_coverage >= config.minimum_forecast_coverage)
            if np.isfinite(validation_coverage)
            else pd.NA
        )
        row["validation_passed"] = (
            bool(
                validation_improvement > 0
                and validation_coverage >= config.minimum_forecast_coverage
            )
            if np.isfinite(validation_improvement)
            and np.isfinite(validation_coverage)
            else pd.NA
        )
        row["validation_abs_bias_improvement"] = (
            abs(row.get("validation_baseline_mean_gap_pct", np.nan))
            - abs(row.get("validation_mean_gap_pct", np.nan))
        )
        row["validation_gap_std_improvement"] = (
            row.get("validation_baseline_gap_pct_std", np.nan)
            - row.get("validation_gap_pct_std", np.nan)
        )
        row["validation_gap_change_mae_improvement"] = (
            row.get("validation_baseline_gap_pct_change_mae", np.nan)
            - row.get("validation_gap_pct_change_mae", np.nan)
        )
        result_rows.append(row)
    return pd.DataFrame(result_rows)

