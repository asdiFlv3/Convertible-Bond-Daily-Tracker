"""Leakage-safe lagged-gap benchmarks for the convertible-bond CRR model.

The benchmark asks whether a cheap continuation of past pricing errors can
match the improvement delivered by lagged implied volatility.  It observes a
gap only on the same scheduled dates as the IV backtest, uses that observation
from the next usable trading day, and expires it after the same maximum age.

Run from the repository root::

    python scripts/gap_correction_backtest.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_COLUMNS_BY_ENGINE = {
    "with_call": "baseline_price_with_call",
    "no_call": "baseline_price_no_call",
}
LATEST_GAP_PRICE_BY_ENGINE = {
    engine: f"price_gap_latest_calibration_{engine}"
    for engine in BASELINE_COLUMNS_BY_ENGINE
}
EWMA_GAP_PRICE_BY_ENGINE = {
    engine: f"price_gap_ewma_calibrations_{engine}"
    for engine in BASELINE_COLUMNS_BY_ENGINE
}
HEAD_TO_HEAD_MODEL_ORDER = (
    "baseline",
    "selected_iv",
    "latest_gap",
    "ewma_gap",
)


@dataclass(frozen=True)
class GapCorrectionConfig:
    """Saved IV inputs and rules frozen before the gap benchmark is run."""

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
    iv_config_csv: Path = Path(
        "output/diagnostics/implied_volatility/"
        "implied_volatility_backtest_config.csv"
    )
    output_dir: Path = Path("output/diagnostics/gap_correction")
    calibration_stride: int = 5
    max_gap_age_trading_days: int = 10
    ewma_alpha: float = 0.5
    minimum_forecast_coverage: float = 0.80


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    source: str,
) -> None:
    """Raise a source-aware error if a required column is absent."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{source} missing columns: {missing}")


def _is_true(value: object) -> bool:
    """Interpret booleans that may have round-tripped through CSV."""

    return str(value).strip().lower() == "true"


def _finite(value: object) -> bool:
    """Return whether a scalar can be interpreted as a finite number."""

    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _validate_config(config: GapCorrectionConfig) -> None:
    """Reject ambiguous smoothing or chronology settings."""

    if config.calibration_stride <= 0:
        raise ValueError("calibration_stride must be positive")
    if config.max_gap_age_trading_days <= 0:
        raise ValueError("max_gap_age_trading_days must be positive")
    if not 0 < config.ewma_alpha <= 1:
        raise ValueError("ewma_alpha must be in (0, 1]")
    if not 0 <= config.minimum_forecast_coverage <= 1:
        raise ValueError(
            "minimum_forecast_coverage must be between zero and one"
        )


def validate_matched_iv_rules(
    iv_config: pd.DataFrame,
    config: GapCorrectionConfig,
) -> None:
    """Ensure the gap benchmark really matches the saved IV information rules."""

    _require_columns(
        iv_config,
        [
            "calibration_stride",
            "max_iv_age_trading_days",
            "minimum_forecast_coverage",
        ],
        "IV backtest config",
    )
    if len(iv_config) != 1:
        raise ValueError("IV backtest config must contain exactly one row")
    row = iv_config.iloc[0]
    checks = (
        (
            "calibration_stride",
            float(row["calibration_stride"]),
            float(config.calibration_stride),
        ),
        (
            "max_iv_age_trading_days",
            float(row["max_iv_age_trading_days"]),
            float(config.max_gap_age_trading_days),
        ),
        (
            "minimum_forecast_coverage",
            float(row["minimum_forecast_coverage"]),
            float(config.minimum_forecast_coverage),
        ),
    )
    mismatches = [
        f"{name}: IV={actual:g}, gap={expected:g}"
        for name, actual, expected in checks
        if not np.isclose(actual, expected)
    ]
    if mismatches:
        raise ValueError(
            "gap benchmark rules do not match saved IV rules: "
            + "; ".join(mismatches)
        )


def _ordered_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Return unique bond/date rows in deterministic chronological order."""

    required = [
        "date",
        "bond_code",
        "market_price",
        "bond_style",
        "is_calibration_date",
        *BASELINE_COLUMNS_BY_ENGINE.values(),
    ]
    _require_columns(daily, required, "IV daily backtest")
    result = daily.copy()
    result["date"] = pd.to_datetime(result["date"])
    duplicated = result.duplicated(["bond_code", "date"], keep=False)
    if duplicated.any():
        examples = (
            result.loc[duplicated, ["bond_code", "date"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "IV daily backtest contains duplicate bond/date rows: "
            f"{examples}"
        )
    return result.sort_values(["bond_code", "date"]).reset_index(drop=True)


def add_gap_correction_forecasts(
    daily: pd.DataFrame,
    config: GapCorrectionConfig,
) -> pd.DataFrame:
    """Add latest-gap and EWMA-gap prices without using the current-day gap."""

    _validate_config(config)
    result = _ordered_daily(daily)
    for engine in BASELINE_COLUMNS_BY_ENGINE:
        for column in (
            f"realized_gap_{engine}",
            f"observed_gap_calibration_{engine}",
            f"gap_latest_calibration_{engine}",
            f"gap_ewma_calibrations_{engine}",
            f"gap_calibration_age_trading_days_{engine}",
            LATEST_GAP_PRICE_BY_ENGINE[engine],
            EWMA_GAP_PRICE_BY_ENGINE[engine],
        ):
            result[column] = np.nan

    for _, group in result.groupby("bond_code", sort=False):
        states = {
            engine: {
                "latest": np.nan,
                "ewma": np.nan,
                "observed_position": None,
            }
            for engine in BASELINE_COLUMNS_BY_ENGINE
        }
        for position, index in enumerate(group.index):
            market = result.at[index, "market_price"]
            is_calibration = _is_true(
                result.at[index, "is_calibration_date"]
            )
            for engine, baseline_column in BASELINE_COLUMNS_BY_ENGINE.items():
                baseline = result.at[index, baseline_column]
                realized_gap = (
                    float(baseline) - float(market)
                    if _finite(baseline) and _finite(market)
                    else np.nan
                )
                result.at[index, f"realized_gap_{engine}"] = realized_gap

                state = states[engine]
                observed_position = state["observed_position"]
                age = (
                    position - int(observed_position)
                    if observed_position is not None
                    else None
                )
                usable_state = (
                    age is not None
                    and age <= config.max_gap_age_trading_days
                    and _finite(state["latest"])
                    and _finite(state["ewma"])
                )
                if usable_state:
                    latest = float(state["latest"])
                    ewma = float(state["ewma"])
                    result.at[
                        index, f"gap_latest_calibration_{engine}"
                    ] = latest
                    result.at[
                        index, f"gap_ewma_calibrations_{engine}"
                    ] = ewma
                    result.at[
                        index,
                        f"gap_calibration_age_trading_days_{engine}",
                    ] = float(age)
                    if _finite(baseline):
                        result.at[
                            index, LATEST_GAP_PRICE_BY_ENGINE[engine]
                        ] = float(baseline) - latest
                        result.at[
                            index, EWMA_GAP_PRICE_BY_ENGINE[engine]
                        ] = float(baseline) - ewma

                # Forecast first.  A gap observed on this row can only affect
                # the following usable trading row, matching the IV backtest.
                if is_calibration and _finite(realized_gap):
                    result.at[
                        index, f"observed_gap_calibration_{engine}"
                    ] = realized_gap
                    stale = (
                        observed_position is None
                        or position - int(observed_position)
                        > config.max_gap_age_trading_days
                    )
                    if stale or not _finite(state["ewma"]):
                        new_ewma = float(realized_gap)
                    else:
                        new_ewma = (
                            config.ewma_alpha * float(realized_gap)
                            + (1 - config.ewma_alpha)
                            * float(state["ewma"])
                        )
                    state["latest"] = float(realized_gap)
                    state["ewma"] = new_ewma
                    state["observed_position"] = position
    return result


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


def main() -> None:
    """Run the saved-data gap benchmark and persist reproducible outputs."""

    config = GapCorrectionConfig()
    _validate_config(config)
    daily = pd.read_csv(config.iv_daily_csv)
    iv_summary = pd.read_csv(config.iv_summary_csv)
    selected = pd.read_csv(config.iv_selected_csv)
    iv_config = pd.read_csv(config.iv_config_csv)
    validate_matched_iv_rules(iv_config, config)

    cutoff = validation_start_date(iv_summary)
    backtest = add_gap_correction_forecasts(daily, config)
    gap_summary = summarize_gap_corrections(backtest, cutoff)
    head_to_head = summarize_head_to_head(backtest, selected, cutoff)
    headline = summarize_headline(head_to_head)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(
        config.output_dir / "gap_correction_backtest_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gap_summary.to_csv(
        config.output_dir / "gap_correction_backtest_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    head_to_head.to_csv(
        config.output_dir / "iv_vs_gap_head_to_head.csv",
        index=False,
        encoding="utf-8-sig",
    )
    headline.to_csv(
        config.output_dir / "iv_vs_gap_headline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    config_values = asdict(config)
    for key, value in list(config_values.items()):
        if isinstance(value, Path):
            config_values[key] = str(value)
    config_values["gap_definition"] = "baseline_price_minus_market_price"
    config_values["observation_rule"] = (
        "same_scheduled_dates_as_iv_then_use_from_t_plus_1"
    )
    pd.DataFrame([config_values]).to_csv(
        config.output_dir / "gap_correction_backtest_config.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("IV-versus-gap headline (all IV selection-qualified bonds):")
    print(headline.to_string(index=False))
    print(f"\nSaved outputs to {config.output_dir}")


if __name__ == "__main__":
    main()
