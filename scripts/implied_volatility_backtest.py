"""Compatibility entry point for the modular IV backtest implementation.

Implementation lives under :mod:`implied_volatility`; imports are re-exported
here so existing notebooks and scripts can migrate without a flag day.
"""

from implied_volatility import ImpliedVolatilityResult, solve_implied_volatility
from implied_volatility.backtest_config import BacktestConfig
from implied_volatility.backtest_engine import (
    BASELINE_COLUMNS_BY_ENGINE,
    FORECAST_COLUMNS,
    FORECAST_COLUMNS_BY_ENGINE,
    _forecast_sigmas,
    _price,
    _required_daily,
    _safe_forecast_price,
    _solve_row,
    _solver_fields,
    run_backtest,
)
from implied_volatility.backtest_reporting import (
    calibration_diagnostics,
    summarize_backtest,
    summarize_by_style,
)
from implied_volatility.model_selection import _matched_baseline, select_models
from implied_volatility.backtest_stage import main, run_iv_stage
from implied_volatility.terms import load_terms_snapshot


if __name__ == "__main__":
    main()
