"""Numerically robust implied-volatility inversion utilities.

The solver is deliberately independent of the convertible-bond model.  A
caller supplies a scalar pricing function and a market target.  Grid points
that the underlying numerical model cannot price are retained as diagnostics
and excluded from bracketing rather than causing the whole date to fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ImpliedVolatilityResult:
    """One solver outcome, including the root bracket and failure evidence."""

    implied_sigma: float
    success: bool
    reason: str
    iterations: int
    target_price: float
    model_price: float
    residual: float
    lower_sigma: float
    upper_sigma: float
    lower_price: float
    upper_price: float
    grid_min_price: float
    grid_max_price: float
    valid_grid_points: int
    monotonic: bool
    bracket_count: int


def _result(
    *,
    target_price: float,
    success: bool,
    reason: str,
    implied_sigma: float = np.nan,
    iterations: int = 0,
    model_price: float = np.nan,
    lower_sigma: float = np.nan,
    upper_sigma: float = np.nan,
    lower_price: float = np.nan,
    upper_price: float = np.nan,
    grid_min_price: float = np.nan,
    grid_max_price: float = np.nan,
    valid_grid_points: int = 0,
    monotonic: bool = False,
    bracket_count: int = 0,
) -> ImpliedVolatilityResult:
    """Build a consistently populated solver result and calculate residual."""

    residual = (
        model_price - target_price
        if np.isfinite(model_price)
        else np.nan
    )
    return ImpliedVolatilityResult(
        implied_sigma=implied_sigma,
        success=success,
        reason=reason,
        iterations=iterations,
        target_price=target_price,
        model_price=model_price,
        residual=residual,
        lower_sigma=lower_sigma,
        upper_sigma=upper_sigma,
        lower_price=lower_price,
        upper_price=upper_price,
        grid_min_price=grid_min_price,
        grid_max_price=grid_max_price,
        valid_grid_points=valid_grid_points,
        monotonic=monotonic,
        bracket_count=bracket_count,
    )


def solve_implied_volatility(
    price_function: Callable[[float], float],
    target_price: float,
    *,
    sigma_grid: tuple[float, ...] = (
        0.01,
        0.10,
        0.20,
        0.40,
        0.60,
        1.00,
        1.50,
        2.00,
        3.00,
    ),
    max_iterations: int = 24,
    price_tolerance: float = 1e-6,
    sigma_tolerance: float = 1e-8,
) -> ImpliedVolatilityResult:
    """Invert a model price only when the grid admits one reliable root."""

    if not np.isfinite(target_price) or target_price <= 0:
        raise ValueError("target_price must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(price_tolerance) or price_tolerance <= 0:
        raise ValueError("price_tolerance must be finite and positive")
    if not np.isfinite(sigma_tolerance) or sigma_tolerance <= 0:
        raise ValueError("sigma_tolerance must be finite and positive")
    grid = np.asarray(sigma_grid, dtype=float)
    if (
        len(grid) < 2
        or not np.isfinite(grid).all()
        or np.any(grid <= 0)
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("sigma_grid must be finite, positive and increasing")

    prices = np.full(len(grid), np.nan, dtype=float)
    for index, sigma in enumerate(grid):
        try:
            candidate = float(price_function(float(sigma)))
            if np.isfinite(candidate):
                prices[index] = candidate
        except Exception:
            # Invalid low-volatility CRR points are common when risk-neutral p
            # leaves [0, 1].  Remaining valid points can still define a root.
            continue

    # Bracketing uses only finite samples, but diagnostics retain how much of
    # the requested grid was actually priceable by the underlying model.
    valid = np.isfinite(prices)
    valid_count = int(valid.sum())
    if valid_count < 2:
        return _result(
            target_price=target_price,
            success=False,
            reason="insufficient_valid_grid_prices",
            valid_grid_points=valid_count,
        )
    valid_grid = grid[valid]
    valid_prices = prices[valid]
    grid_min_price = float(valid_prices.min())
    grid_max_price = float(valid_prices.max())
    # A unique IV is meaningful only on a non-decreasing price curve. Small
    # downward moves inside the price tolerance are treated as numerical noise.
    monotonic = bool(np.all(np.diff(valid_prices) >= -price_tolerance))
    errors = valid_prices - target_price
    exact = np.flatnonzero(np.abs(errors) <= price_tolerance)
    brackets = [
        index
        for index in range(len(valid_grid) - 1)
        if errors[index] * errors[index + 1] < 0
    ]

    common = {
        "target_price": target_price,
        "grid_min_price": grid_min_price,
        "grid_max_price": grid_max_price,
        "valid_grid_points": valid_count,
        "monotonic": monotonic,
        "bracket_count": len(brackets) + len(exact),
    }
    if not monotonic:
        return _result(
            **common,
            success=False,
            reason="non_monotonic_price_curve",
        )
    if len(exact) > 1:
        return _result(
            **common,
            success=False,
            reason="multiple_grid_roots_or_flat_price",
        )
    if len(exact) == 1:
        index = int(exact[0])
        return _result(
            **common,
            success=True,
            reason="grid_exact",
            implied_sigma=float(valid_grid[index]),
            model_price=float(valid_prices[index]),
            lower_sigma=float(valid_grid[index]),
            upper_sigma=float(valid_grid[index]),
            lower_price=float(valid_prices[index]),
            upper_price=float(valid_prices[index]),
        )
    if len(brackets) == 0:
        return _result(
            **common,
            success=False,
            reason="target_outside_model_price_range",
        )
    if len(brackets) > 1:
        return _result(
            **common,
            success=False,
            reason="multiple_root_brackets",
        )

    bracket = brackets[0]
    low_sigma = float(valid_grid[bracket])
    high_sigma = float(valid_grid[bracket + 1])
    low_price = float(valid_prices[bracket])
    high_price = float(valid_prices[bracket + 1])
    middle_sigma = (low_sigma + high_sigma) / 2
    middle_price = np.nan
    for iteration in range(1, max_iterations + 1):
        middle_sigma = (low_sigma + high_sigma) / 2
        try:
            middle_price = float(price_function(middle_sigma))
        except Exception as exc:
            return _result(
                **common,
                success=False,
                reason=f"bisection_pricing_error: {type(exc).__name__}: {exc}",
                iterations=iteration,
                model_price=middle_price,
                lower_sigma=low_sigma,
                upper_sigma=high_sigma,
                lower_price=low_price,
                upper_price=high_price,
            )
        if not np.isfinite(middle_price):
            return _result(
                **common,
                success=False,
                reason="non_finite_bisection_price",
                iterations=iteration,
                model_price=middle_price,
                lower_sigma=low_sigma,
                upper_sigma=high_sigma,
                lower_price=low_price,
                upper_price=high_price,
            )
        if abs(middle_price - target_price) <= price_tolerance:
            return _result(
                **common,
                success=True,
                reason="converged",
                iterations=iteration,
                implied_sigma=middle_sigma,
                model_price=middle_price,
                lower_sigma=low_sigma,
                upper_sigma=high_sigma,
                lower_price=low_price,
                upper_price=high_price,
            )
        if high_sigma - low_sigma <= sigma_tolerance:
            return _result(
                **common,
                success=False,
                reason="sigma_interval_collapsed_without_price_match",
                iterations=iteration,
                model_price=middle_price,
                lower_sigma=low_sigma,
                upper_sigma=high_sigma,
                lower_price=low_price,
                upper_price=high_price,
            )
        if middle_price < target_price:
            low_sigma = middle_sigma
            low_price = middle_price
        else:
            high_sigma = middle_sigma
            high_price = middle_price

    return _result(
        **common,
        success=False,
        reason="max_iterations_without_convergence",
        iterations=max_iterations,
        model_price=middle_price,
        lower_sigma=low_sigma,
        upper_sigma=high_sigma,
        lower_price=low_price,
        upper_price=high_price,
    )
