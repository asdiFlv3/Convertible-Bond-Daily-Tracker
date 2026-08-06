"""Leakage-safe latest-gap and EWMA daily forecast engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GapCorrectionConfig


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
    """Add latest-gap and EWMA-gap prices without using the current-day gap.

    The signed gap is ``baseline - market``. A forecast therefore subtracts
    the stored gap from today's baseline to estimate today's market price.
    """

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
        # Each bond and engine owns an independent online state. Only a
        # scheduled calibration row is allowed to update that state.
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
                    # After a long gap in observations, restart EWMA rather
                    # than blending a fresh signal with an expired state.
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

