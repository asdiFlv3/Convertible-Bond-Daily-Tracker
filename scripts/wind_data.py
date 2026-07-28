"""WindPy data access for daily convertible-bond tracking.

Only this module knows about WindPy's connection object, WSS/WSD return shapes,
and code-generator field names.  Keeping those details here prevents the
pricing model and reporting code from becoming coupled to a particular data
vendor.

Market data are never replaced with manual values in this module.  When a Wind
field has not yet been identified, the code fails with a clear placeholder
message so that missing data cannot silently turn into a model assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
from WindPy import w


CODE_GENERATOR_PLACEHOLDER = "__WIND_CODE_GENERATOR_REQUIRED__"

# WindPy changes its return shape dynamically when ``usedf=True``: the runtime
# value is ``(error_code, DataFrame)`` even though the old WindPy source only
# advertises ``WindData | None``.  This local alias documents the real contract
# once, at the vendor boundary, so the rest of the project remains strictly
# typed without scattering ``type: ignore`` comments.
WindDataFrameResult: TypeAlias = tuple[int, pd.DataFrame | None]


@dataclass(frozen=True)
class WindFieldConfig:
    """Wind fields that must be copied from the terminal code generator.

    Generate the five WSS fields with one convertible bond as the security and
    a historical trade date.  Generate the two WSD fields over a historical
    date range, confirming that their values change on the correct effective
    dates.

    Do not replace a missing field with a manually entered value.  Paste the
    exact generated Wind field name here once its definition and unit have been
    checked.
    """

    # WSS: underlying stock Wind code, expected value such as ``300529.SZ``.
    underlying_code: str = CODE_GENERATOR_PLACEHOLDER
    # WSS: contractual maturity date.
    maturity_date: str = CODE_GENERATOR_PLACEHOLDER
    # WSS: first date on which conversion is contractually allowed.
    conversion_start_date: str = CODE_GENERATOR_PLACEHOLDER
    # WSS: first date on which the conditional put provision can apply.
    put_start_date: str = CODE_GENERATOR_PLACEHOLDER
    # WSS: amount paid at maturity.  Its generated definition must clarify
    # whether the last coupon and tax are included.
    maturity_redemption_price: str = CODE_GENERATOR_PLACEHOLDER
    # WSD: conversion price effective on each historical valuation date.  This
    # must not be a current snapshot repeated backwards through history.
    historical_conversion_price: str = CODE_GENERATOR_PLACEHOLDER
    # WSD: coupon rate effective in each historical accrual year.
    historical_coupon_rate: str = CODE_GENERATOR_PLACEHOLDER


@dataclass(frozen=True)
class TrackingConfig:
    """Run-level configuration unrelated to CRR model assumptions."""

    bond_code: str
    start: str
    end: str
    volatility_window: int = 60
    annualization_days: int = 252
    # ``F`` is Wind's documented forward-adjustment option for stock/fund
    # prices.  Adjusted stock prices avoid artificial return jumps caused by
    # dividends and splits when estimating historical volatility.
    price_adjustment: str = "F"
    output_dir: Path = Path("output")


def assert_wind_fields_configured(config: WindFieldConfig) -> None:
    """Fail before connecting to Wind if code-generator fields are missing."""

    missing = [
        item.name
        for item in fields(config)
        if getattr(config, item.name) == CODE_GENERATOR_PLACEHOLDER
    ]
    if missing:
        joined = "\n  - ".join(missing)
        raise RuntimeError(
            "以下字段必须用 Wind 代码生成器确认后填入 WindFieldConfig：\n"
            f"  - {joined}"
        )


def normalize_wind_code(code: str) -> str:
    """Normalize case and require an explicit Wind exchange suffix."""

    value = str(code).strip().upper()
    if "." not in value:
        raise ValueError(
            f"Wind代码必须包含市场后缀（例如 123117.SZ），收到：{code!r}"
        )
    return value


def start_wind() -> None:
    """Start WindPy and verify that the local terminal session is connected."""

    result = w.start(waitTime=120)
    if result.ErrorCode != 0:
        raise RuntimeError(f"WindPy启动失败，错误码：{result.ErrorCode}")
    if not w.isconnected():
        raise RuntimeError("WindPy启动后仍未连接，请确认Wind终端已登录")


def _wind_wss(
    code: str,
    field_names: list[str],
    trade_date: pd.Timestamp,
) -> pd.Series:
    """Return the first row of a one-security WSS request.

    ``usedf=True`` changes WindPy's return value from a WindData object to
    ``(error_code, DataFrame)``.  This wrapper centralizes that less-obvious
    calling convention and preserves the response in any raised error.
    """

    result = cast(
        WindDataFrameResult,
        w.wss(
            code,
            ",".join(field_names),
            f"tradeDate={trade_date:%Y%m%d}",
            usedf=True,
        ),
    )
    error_code, frame = result
    if error_code != 0:
        raise RuntimeError(
            f"WSS取数失败：code={code}, fields={field_names}, "
            f"error_code={error_code}, response={frame!r}"
        )
    if frame is None or frame.empty:
        raise ValueError(f"WSS返回空数据：code={code}, fields={field_names}")
    return frame.iloc[0]


def _wind_wsd(
    code: str,
    field_names: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    options: str,
) -> pd.DataFrame:
    """Return a normalized, ascending-date WSD DataFrame."""

    result = cast(
        WindDataFrameResult,
        w.wsd(
            code,
            ",".join(field_names),
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            options,
            usedf=True,
        ),
    )
    error_code, frame = result
    if error_code != 0:
        raise RuntimeError(
            f"WSD取数失败：code={code}, fields={field_names}, "
            f"error_code={error_code}, response={frame!r}"
        )
    if frame is None or frame.empty:
        raise ValueError(f"WSD返回空数据：code={code}, fields={field_names}")

    result = frame.copy()
    result.index = pd.to_datetime(result.index).normalize()
    result = result.sort_index()
    # Wind field casing may vary across functions/versions.  Normalize once so
    # downstream code does not contain repeated case-insensitive lookups.
    result.columns = [str(column).upper() for column in result.columns]
    return result


def _value_by_field(row: pd.Series, field_name: str) -> Any:
    """Read a WSS value without relying on output-field casing."""

    wanted = field_name.upper()
    lookup = {str(index).upper(): index for index in row.index}
    if wanted not in lookup:
        raise KeyError(
            f"Wind返回结果缺少字段 {field_name!r}；实际字段：{list(row.index)}"
        )
    return row[lookup[wanted]]


def _series_by_field(
    frame: pd.DataFrame,
    field_name: str,
) -> pd.Series:
    """Read one normalized WSD field and report useful schema diagnostics."""

    wanted = field_name.upper()
    if wanted not in frame.columns:
        raise KeyError(
            f"Wind返回结果缺少字段 {field_name!r}；实际字段：{list(frame.columns)}"
        )
    return frame[wanted]


def _numeric_series(
    frame: pd.DataFrame,
    field_name: str,
    output_name: str,
) -> pd.Series:
    """Coerce one Wind field to a named numeric series."""

    values = pd.to_numeric(
        _series_by_field(frame, field_name),
        errors="coerce",
    )
    values.name = output_name
    return values


def _parse_wind_date(value: object, field_name: str) -> pd.Timestamp:
    """Parse Wind dates without treating ``YYYYMMDD`` integers as nanoseconds.

    Wind may return the same conceptual date as ``20270623``, ``"20270623"``,
    ``"2027-06-23"`` or a pandas-compatible date object.  Passing the integer
    form directly to ``pd.to_datetime`` is dangerous: pandas interprets it as
    nanoseconds after 1970 rather than as a compact calendar date.
    """

    if pd.isna(value):
        raise ValueError(f"Wind字段 {field_name!r} 返回空日期")

    text = str(value).strip()
    # A numeric DataFrame column can turn YYYYMMDD into a float-shaped string.
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    try:
        if len(text) == 8 and text.isdigit():
            parsed = pd.to_datetime(
                text,
                format="%Y%m%d",
                errors="raise",
            )
        else:
            parsed = pd.to_datetime(text, errors="raise")
        return pd.Timestamp(parsed).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Wind字段 {field_name!r} 日期无法解析：{value!r}"
        ) from exc


def _load_snapshot_metadata(
    bond_code: str,
    end: pd.Timestamp,
    wind_fields: WindFieldConfig,
) -> dict[str, Any]:
    """Load security-master facts used by every valuation date."""

    snapshot_fields = [
        wind_fields.underlying_code,
        wind_fields.maturity_date,
        wind_fields.conversion_start_date,
        wind_fields.put_start_date,
        wind_fields.maturity_redemption_price,
    ]
    snapshot = _wind_wss(bond_code, snapshot_fields, end)
    stock_code = normalize_wind_code(
        _value_by_field(snapshot, wind_fields.underlying_code)
    )

    metadata: dict[str, Any] = {
        "bond_code": bond_code,
        "stock_code": stock_code,
        "maturity_date": _parse_wind_date(
            _value_by_field(snapshot, wind_fields.maturity_date),
            wind_fields.maturity_date,
        ),
        "conversion_start_date": _parse_wind_date(
            _value_by_field(snapshot, wind_fields.conversion_start_date),
            wind_fields.conversion_start_date,
        ),
        "put_start_date": _parse_wind_date(
            _value_by_field(snapshot, wind_fields.put_start_date),
            wind_fields.put_start_date,
        ),
        "maturity_redemption_price": float(
            pd.to_numeric(
                _value_by_field(
                    snapshot,
                    wind_fields.maturity_redemption_price,
                ),
                errors="raise",
            )
        ),
    }

    # ``NaT`` comparisons quietly produce False, which would hide bad security
    # master data later.  Reject missing dates at the boundary instead.
    date_keys = (
        "maturity_date",
        "conversion_start_date",
        "put_start_date",
    )
    missing_dates = [key for key in date_keys if pd.isna(metadata[key])]
    if missing_dates:
        raise ValueError(f"Wind静态资料包含无效日期：{missing_dates}")
    if metadata["maturity_redemption_price"] <= 0:
        raise ValueError("Wind返回的到期赎回金额必须为正数")
    return metadata


def load_wind_daily_inputs(
    tracking: TrackingConfig,
    wind_fields: WindFieldConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load all non-manual daily inputs from Wind.

    The returned table contains only dates shared by bond prices, stock prices,
    historical conversion prices and historical coupon rates.  Inner alignment
    prevents a value from one market date being paired with another.
    """

    assert_wind_fields_configured(wind_fields)
    bond_code = normalize_wind_code(tracking.bond_code)
    start = pd.Timestamp(tracking.start).normalize()
    end = pd.Timestamp(tracking.end).normalize()
    if end < start:
        raise ValueError("end不能早于start")
    if tracking.volatility_window not in (20, 60, 120, 252):
        raise ValueError("volatility_window目前支持20、60、120、252")

    metadata = _load_snapshot_metadata(
        bond_code,
        end,
        wind_fields,
    )
    stock_code = metadata["stock_code"]

    # A 252-observation window needs substantially more than 252 calendar days
    # because of weekends, holidays and suspensions.  The extra 420 calendar
    # days are loaded only before ``start`` and therefore cannot leak future
    # information into any volatility estimate.
    stock_start = start - pd.Timedelta(days=420)

    # Bond close is not adjusted.  Stock close is forward-adjusted solely for
    # return/volatility estimation, avoiding artificial corporate-action jumps.
    bond_market = _wind_wsd(
        bond_code,
        ["close"],
        start,
        end,
        "Period=D;Days=Trading;Fill=Blank",
    ).rename(columns={"CLOSE": "bond_close"})
    stock_market = _wind_wsd(
        stock_code,
        ["close"],
        stock_start,
        end,
        (
            "Period=D;Days=Trading;Fill=Blank;"
            f"PriceAdj={tracking.price_adjustment}"
        ),
    ).rename(columns={"CLOSE": "stock_close"})

    # ``Fill=Previous`` is appropriate for terms that remain legally effective
    # until a new value takes effect.  It is intentionally not used for prices
    # or volumes, where filling would manufacture observations on non-trading
    # or suspended dates.
    term_history = _wind_wsd(
        bond_code,
        [
            wind_fields.historical_conversion_price,
            wind_fields.historical_coupon_rate,
        ],
        stock_start,
        end,
        "Period=D;Days=Trading;Fill=Previous",
    )
    conversion_price = _numeric_series(
        term_history,
        wind_fields.historical_conversion_price,
        "K",
    )
    coupon_rate = _numeric_series(
        term_history,
        wind_fields.historical_coupon_rate,
        "coupon_used",
    )

    # Wind ``couponrate3`` returns percentage points: for example, 2.0 means a
    # 2% annual coupon.  CRR inputs use decimal rates, so convert 2.0 to 0.02.
    coupon_unit_multiplier = 0.01
    coupon_rate = coupon_rate * coupon_unit_multiplier

    stock_market["log_return"] = np.log(
        stock_market["stock_close"]
        / stock_market["stock_close"].shift(1)
    )
    for window in (20, 60, 120, 252):
        stock_market[f"vol_{window}d"] = (
            stock_market["log_return"]
            .rolling(window, min_periods=window)
            .std(ddof=1)
            * np.sqrt(tracking.annualization_days)
        )

    # An inner join is conservative: a pricing row is created only when all
    # four sources describe the same normalized trade date.
    daily = pd.concat(
        [bond_market, stock_market, conversion_price, coupon_rate],
        axis=1,
        join="inner",
    )
    daily = daily.loc[
        daily.index.to_series().between(start, end)
    ].copy()
    daily.index.name = "date"
    daily = daily.reset_index()
    if daily.empty:
        raise ValueError("Wind返回的转债、正股和条款序列没有共同交易日")

    return daily, metadata
