"""Compatibility entry point for modular IV and gap/EWMA plotting."""

from implied_volatility.plot_common import (
    BASELINE_COLOR,
    BATCH_SAMPLE_MARKER,
    CALL_COLOR,
    GAP_EWMA_COLOR,
    GAP_LATEST_COLOR,
    GRID_COLOR,
    IV_COLOR,
    MARKET_COLOR,
    NO_CALL_COLOR,
    STYLE_LABELS,
    STYLE_ORDER,
    PlotConfig,
    _format_date_axis,
    _is_true,
    _load_inputs,
    _qualification_note,
    _require_columns,
    _save_figure,
    _selection_lookup,
    _selection_style_codes,
    _small_multiple_figure,
    _style_axis,
    _subplot_shape,
    _validation_lookup,
)
from implied_volatility.plot_gap_comparison import (
    save_iv_vs_gap_mape_chart,
    save_iv_vs_gap_tradeoff_chart,
)
from implied_volatility.plot_prices import save_iv_group_price_chart
from implied_volatility.plot_stage import generate_plots, main
from implied_volatility.plot_summaries import (
    save_solver_success_chart,
    save_strategy_tradeoff_chart,
    save_validation_mape_chart,
)


if __name__ == "__main__":
    main()
