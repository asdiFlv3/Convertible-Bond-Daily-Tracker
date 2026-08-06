"""Compatibility entry point for the modular gap/EWMA backtest."""

from gap.config import GapCorrectionConfig
from gap.correction import (
    BASELINE_COLUMNS_BY_ENGINE,
    EWMA_GAP_PRICE_BY_ENGINE,
    HEAD_TO_HEAD_MODEL_ORDER,
    LATEST_GAP_PRICE_BY_ENGINE,
    _finite,
    _is_true,
    _ordered_daily,
    _require_columns,
    _validate_config,
    add_gap_correction_forecasts,
    validate_matched_iv_rules,
)
from gap.reporting import (
    _metric_values,
    _segments,
    summarize_gap_corrections,
    summarize_head_to_head,
    summarize_headline,
    validation_start_date,
)
from gap.stage import main, run_gap_stage


if __name__ == "__main__":
    main()
