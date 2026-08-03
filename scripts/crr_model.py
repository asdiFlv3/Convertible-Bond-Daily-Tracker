"""CRR convertible-bond pricing model.

This module deliberately contains no WindPy, plotting, or file-system code.
Keeping the numerical model independent makes it possible to:

* test the pricing formula without opening the Wind terminal;
* reuse the model from a notebook or another data source;
* distinguish model assumptions from market-data problems.

The implementation reproduces the simplified model in
``notebook/daily_tracking_v1.ipynb``.  It is not yet a full prospectus-level
convertible-bond engine.  In particular, call and put provisions are represented
by immediate parity thresholds rather than path-dependent "M out of N days"
tests, and coupon cashflows are spread uniformly over tree steps rather than
paid on their contractual dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# Chinese exchange-listed public convertible bonds are quoted and modelled per
# RMB 100 of face value.  Treat this as the model's unit of account rather than
# a run-specific calibration parameter.
FACE_VALUE = 100.0


@dataclass(frozen=True)
class ManualModelTerms:
    """Model assumptions and the clauses retained as manual inputs.

    The values in this class fall into two groups:

    * ``risk_free_rate`` through ``tree_steps`` are modelling/calibration
      choices rather than security master data.
    * ``call_parity_trigger``, ``put_parity_trigger`` and ``put_price`` are
      simplified representations of prospectus clauses.  They must be checked
      against the selected bond's prospectus before production use.

    Security facts that Wind can supply—maturity date, coupon, conversion
    price, redemption amount and exercise-period dates—must not be added here.
    They belong in the Wind data layer.
    """

    risk_free_rate: float = 0.02
    dividend_yield: float = 0.0
    credit_spread: float = 0.01
    debt_equity_blend_low: float = 70.0
    debt_equity_blend_high: float = 130.0
    call_parity_trigger: float = 130.0
    put_parity_trigger: float = 70.0
    put_price: float = 103.0
    tree_steps: int = 200


def parity(
    stock_price: float | np.ndarray | pd.Series,
    face_value: float,
    conversion_price: float | np.ndarray | pd.Series,
) -> Any:
    """Return conversion parity per ``face_value`` of convertible bond.

    A bond with face value 100 and conversion price K converts into ``100 / K``
    shares.  Its conversion value is therefore ``stock_price * 100 / K``.
    Array-like inputs are accepted so the same function works for a full CRR
    layer and for a pandas daily tracking table.
    """

    return stock_price * face_value / conversion_price


def crr_convertible_basic(
    stock_price: float,
    conversion_price: float,
    sigma: float,
    risk_free_rate: float,
    maturity_years: float,
    dividend_yield: float,
    maturity_redemption_price: float,
    coupon_rate: float,
    credit_spread: float,
    blend_low: float,
    blend_high: float,
    conversion_wait_years: float,
    put_wait_years: float,
    steps: int,
    call_parity_trigger: float,
    put_parity_trigger: float,
    face_value: float,
    put_price: float,
    diagnostics: dict[str, float | bool] | None = None,
) -> float:
    """Price one convertible bond with the notebook's simplified CRR model.

    Parameters use annual decimal rates: 2% must be supplied as ``0.02``.
    ``conversion_wait_years`` and ``put_wait_years`` are measured from the
    valuation date, not from the issue date.

    Model outline
    -------------
    1. Build the recombining CRR stock-price tree.
    2. At maturity choose the greater of conversion parity and redemption.
    3. Roll values backwards using a parity-dependent blended discount rate.
    4. Once conversion is allowed, compare continuation with conversion value.
    5. Apply the simplified call and put threshold rules.

    Important approximations
    ------------------------
    * Coupon interest is accrued uniformly at every tree step.
    * Call/put clauses ignore consecutive-day and observation-window rules.
    * A call-threshold hit forces value to parity immediately.
    * Credit spread is a constant and only enters the blended discount rate.
    * Conversion-price reset/downward-revision optionality is not modelled
      inside the future tree; only the historically effective K at each
      valuation date is used.
    """

    positive_inputs = (
        stock_price,
        conversion_price,
        sigma,
        maturity_years,
        steps,
        face_value,
    )
    if min(positive_inputs) <= 0:
        raise ValueError("S、K、sigma、T、n、face_value必须为正数")
    if blend_high <= blend_low:
        raise ValueError("blend_high必须大于blend_low")
    if coupon_rate < 0:
        raise ValueError("coupon_rate不能为负数")

    steps = int(steps)
    dt = maturity_years / steps
    up = np.exp(sigma * np.sqrt(dt))
    down = 1.0 / up

    # Under the risk-neutral measure, the expected stock growth rate is r-b.
    # Checking p explicitly catches inconsistent parameters and overly coarse
    # trees instead of silently producing an invalid price.
    probability = (
        np.exp((risk_free_rate - dividend_yield) * dt) - down
    ) / (up - down)
    if not 0 <= probability <= 1:
        raise ValueError(
            f"CRR风险中性概率越界：p={probability:.6f}；请增加步数或检查参数"
        )

    # This evenly spread coupon is inherited from the notebook.  A production
    # engine should instead place cashflows on the actual coupon payment dates.
    interest_per_step = coupon_rate * face_value * dt

    node = np.arange(steps + 1)
    terminal_stock = stock_price * up**node * down ** (steps - node)
    terminal_parity = parity(
        terminal_stock,
        face_value,
        conversion_price,
    )
    value = np.maximum(terminal_parity, maturity_redemption_price)

    eligible_conversion_nodes = 0
    chosen_conversion_nodes = 0
    earliest_conversion_years = np.nan
    root_continuation = np.nan
    root_parity = np.nan
    root_conversion_eligible = False
    root_conversion_optimal = False

    for layer in range(steps - 1, -1, -1):
        node = np.arange(layer + 1)
        current_stock = stock_price * up**node * down ** (layer - node)
        current_parity = parity(
            current_stock,
            face_value,
            conversion_price,
        )
        current_time = layer * dt

        # Low-parity nodes behave more like credit debt and are discounted near
        # r + spread.  High-parity nodes behave more like equity and are
        # discounted near r.  The linear interpolation is a modelling choice,
        # not a contractual term.
        equity_weight = np.clip(
            (current_parity - blend_low) / (blend_high - blend_low),
            0,
            1,
        )
        discount_rate = (
            risk_free_rate * equity_weight
            + (1 - equity_weight) * (risk_free_rate + credit_spread)
        )
        continuation = (
            np.exp(-discount_rate * dt)
            * (probability * value[1:] + (1 - probability) * value[:-1])
            + interest_per_step
        )

        if layer == 0:
            root_continuation = float(continuation[0])
            root_parity = float(current_parity[0])

        if current_time >= conversion_wait_years:
            # The investor chooses between holding and converting.  The call
            # approximation then forces conversion when parity exceeds C1.
            conversion_optimal = current_parity >= continuation
            eligible_conversion_nodes += len(current_parity)
            chosen_conversion_nodes += int(conversion_optimal.sum())
            if conversion_optimal.any():
                earliest_conversion_years = current_time
            if layer == 0:
                root_conversion_eligible = True
                root_conversion_optimal = bool(conversion_optimal[0])
            value = np.maximum(continuation, current_parity)
            hit_call = current_parity > call_parity_trigger
            value[hit_call] = current_parity[hit_call]
        else:
            value = continuation

        if current_time >= put_wait_years:
            # The simplified put is available whenever parity falls below C2.
            # Real prospectuses normally contain path-dependent tests that are
            # not represented by this one-layer condition.
            hit_put = current_parity < put_parity_trigger
            value[hit_put] = np.maximum(
                continuation[hit_put],
                put_price,
            )

    if diagnostics is not None:
        diagnostics.update(
            {
                "root_continuation": root_continuation,
                "root_parity": root_parity,
                "root_continuation_minus_parity": (
                    root_continuation - root_parity
                ),
                "root_conversion_eligible": root_conversion_eligible,
                "root_conversion_optimal": root_conversion_optimal,
                "conversion_node_share": (
                    chosen_conversion_nodes / eligible_conversion_nodes
                    if eligible_conversion_nodes
                    else np.nan
                ),
                "earliest_conversion_years": earliest_conversion_years,
            }
        )
    return float(value[0])
