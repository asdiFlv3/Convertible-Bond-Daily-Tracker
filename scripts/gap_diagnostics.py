"""Compatibility entry point for the modular gap diagnostic suite."""

from gap.diagnostics import (
    DiagnosticConfig,
    _implied_no_call_volatility,
    _no_call_price,
    _valid_daily,
    main,
    run_diagnostics,
    summarize_reset_proximity,
)


if __name__ == "__main__":
    main()
