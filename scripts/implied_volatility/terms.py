"""Reconstruct saved CRR terms for offline IV repricing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crr_model import ManualModelTerms


def load_terms_snapshot(summary: pd.DataFrame) -> dict[str, ManualModelTerms]:
    """Reconstruct the exact per-bond assumptions saved by the batch run."""

    required = [
        "bond_code",
        "risk_free_rate",
        "dividend_yield",
        "credit_spread",
        "call_parity_trigger",
        "put_parity_trigger",
        "put_price",
        "tree_steps",
    ]
    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise KeyError(f"comparison summary missing terms columns: {missing}")
    if summary["bond_code"].isna().any():
        raise ValueError("comparison summary contains a missing bond code")
    normalized_codes = summary["bond_code"].astype(str)
    duplicated = normalized_codes.duplicated(keep=False)
    if duplicated.any():
        codes = normalized_codes.loc[duplicated].unique().tolist()
        raise ValueError(f"comparison summary contains duplicate bonds: {codes}")

    def finite_value(row: object, column: str) -> float:
        """Read one required finite scalar from a saved terms row."""

        value = float(getattr(row, column))
        if not np.isfinite(value):
            code = str(getattr(row, "bond_code"))
            raise ValueError(f"{code} has invalid {column}: {value}")
        return value

    has_blend_snapshot = {
        "debt_equity_blend_low",
        "debt_equity_blend_high",
    }.issubset(summary.columns)
    result: dict[str, ManualModelTerms] = {}
    for row in summary.itertuples(index=False):
        code = str(row.bond_code)
        blend_low = (
            finite_value(row, "debt_equity_blend_low")
            if has_blend_snapshot
            else 70.0
        )
        blend_high = (
            finite_value(row, "debt_equity_blend_high")
            if has_blend_snapshot
            else 130.0
        )
        tree_steps_value = finite_value(row, "tree_steps")
        tree_steps = int(tree_steps_value)
        if tree_steps <= 0 or tree_steps != tree_steps_value:
            raise ValueError(f"{code} has invalid tree_steps: {tree_steps_value}")
        result[code] = ManualModelTerms(
            risk_free_rate=finite_value(row, "risk_free_rate"),
            dividend_yield=finite_value(row, "dividend_yield"),
            credit_spread=finite_value(row, "credit_spread"),
            debt_equity_blend_low=blend_low,
            debt_equity_blend_high=blend_high,
            call_parity_trigger=finite_value(row, "call_parity_trigger"),
            put_parity_trigger=finite_value(row, "put_parity_trigger"),
            put_price=finite_value(row, "put_price"),
            tree_steps=tree_steps,
        )
        if blend_low <= 0 or blend_low >= blend_high:
            raise ValueError(
                f"{code} blend thresholds must satisfy 0 < low < high"
            )
        if result[code].put_price <= 0:
            raise ValueError(f"{code} put_price must be positive")
        if result[code].call_parity_trigger <= 0:
            raise ValueError(f"{code} call_parity_trigger must be positive")
        if result[code].put_parity_trigger <= 0:
            raise ValueError(f"{code} put_parity_trigger must be positive")
    return result

