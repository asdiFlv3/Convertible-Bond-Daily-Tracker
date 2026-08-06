

# Convertible Bond Daily Tracker

Research tools for comparing Chinese exchange-listed convertible-bond prices
with a simplified Cox-Ross-Rubinstein (CRR) model. The project retrieves data
from Wind, builds point-in-time daily inputs, runs single- or multi-bond
valuations, and provides offline diagnostics and implied-volatility backtests.

> This is a research model, not a production trading or risk-management
> system. Its simplified treatment of bond clauses can materially affect
> theoretical prices.

## Requirements

- Windows, Python 3.12+, and [`uv`](https://docs.astral.sh/uv/)
- Wind Financial Terminal with an active, logged-in account
- WindPy exposed to this project's Python environment

WindPy ships with Wind and is intentionally absent from `pyproject.toml`.
Install the remaining dependencies with:

```text
uv sync
```

## Quick start

1. Open `scripts/dayTrack_crr.py`.
2. Fill `WindFieldConfig` using fields verified in the Wind code generator.
3. Set the bond, dates, and volatility window in `TrackingConfig`.
4. Check `ManualModelTerms`, especially call/put terms, against the prospectus.
5. Run:

```text
python scripts/dayTrack_crr.py
```

Rates use annual decimals: `0.02` means 2%. Historical conversion-price and
coupon fields must contain values effective on each valuation date, not the
latest value repeated backward through history.

## Entry points

| Script | Purpose | Uses Wind |
| --- | --- | --- |
| `scripts/dayTrack_crr.py` | Track and price one bond | Yes |
| `scripts/batch_compare_crr.py` | Compare multiple bonds | Yes |
| `scripts/gap_diagnostics.py` | Analyze saved pricing gaps and sensitivities | No |
| `scripts/implied_volatility_backtest.py` | Backtest lagged implied-volatility forecasts | No |
| `scripts/gap_correction_backtest.py` | Compare lagged gap/EWMA corrections with IV | No |
| `scripts/implied_volatility_plots.py` | Regenerate IV backtest figures | No |

The two Wind-backed entry points contain example configuration blocks rather
than a command-line argument parser. The batch bond-code list is the source of
truth for the offline pipeline: a batch run writes a code-derived run manifest,
and IV, gap/EWMA, and plotting scripts resolve their inputs from
`output/current_run.json`.

## Project structure

```text
scripts/
|-- crr_model.py                    # Pure numerical pricing engine
|-- wind_data.py                    # Wind access and daily input alignment
|-- tracking.py                     # Daily valuation and reporting
|-- batch_compare_crr.py            # Multi-bond orchestration
|-- pipeline_paths.py               # Shared run paths and manifests
|-- implied_volatility/
|   |-- solver.py                   # Generic IV root solver
|   |-- backtest_engine.py          # Leakage-safe daily forecasts
|   |-- backtest_reporting.py       # Summaries and solver diagnostics
|   |-- model_selection.py          # Calibration-time model selection
|   |-- backtest_stage.py           # CSV-oriented stage runner
|   |-- plot_common.py              # Plot inputs and shared helpers
|   |-- plot_prices.py              # Per-bond price charts
|   |-- plot_summaries.py           # Cross-bond comparison charts
|   |-- plot_gap_comparison.py      # IV-versus-gap comparison charts
|   `-- plot_stage.py               # Figure-suite runner
|-- gap/
|   |-- correction.py               # Latest-gap and EWMA forecasts
|   |-- reporting.py                # IV-versus-gap summaries
|   |-- diagnostics.py              # Offline sensitivity diagnostics
|   `-- stage.py                    # CSV-oriented stage runner
|-- implied_volatility_backtest.py  # Thin compatibility entry point
|-- gap_correction_backtest.py      # Thin compatibility entry point
|-- gap_diagnostics.py              # Thin compatibility entry point
`-- implied_volatility_plots.py     # Thin compatibility entry point
tests/                              # Standard-library unittest suite
output/                             # Generated CSV and PNG artifacts
```

The Wind/CRR core remains flat because those modules form one small, stable
execution chain. Offline IV and gap analytics are packages so calculation,
reporting, persistence, and plotting can be reviewed independently. The thin
root entry points preserve the existing commands and import names.

Unadjusted stock closes are used for conversion parity and pricing. A separate
forward-adjusted series is used only for historical-return and volatility
estimation, keeping stock and conversion-price scales consistent.

## Outputs

A single-bond run writes an Excel-friendly CSV and a PNG chart under `output/`.
The table retains aligned market/model inputs, with-call and no-call prices,
estimated call impact, and row-level `pricing_error` diagnostics. Batch and
offline outputs are isolated by the normalized bond-code set:

```text
output/runs/<run_id>/
|-- batch/
|-- implied_volatility/
|-- gap_correction/
`-- figures/
```

`output/current_run.json` points to the latest batch with at least one
successful bond. One bond failure does not discard successful batch results;
an all-failed run does not replace the previous current run.

The IV backtest prevents same-day leakage: a calibration first becomes
available on the next usable trading day. Model selection uses the earlier
calendar segment, while the final segment remains out-of-sample validation.

The gap-correction benchmark reuses the saved IV calendar and information
rules. It observes the CRR-minus-market gap on the same scheduled dates,
starts using it at t+1, and expires it after ten usable trading days.  Its
latest-gap and EWMA-gap prices require no new CRR solve.  Run the offline
sequence as:

```text
python scripts/batch_compare_crr.py
python scripts/implied_volatility_backtest.py
python scripts/gap_correction_backtest.py
python scripts/implied_volatility_plots.py
```

After changing `bond_specs` in `batch_compare_crr.py`, rerun this sequence; no
IV, gap/EWMA, or plotting path needs to be edited manually.

Gap correction is a market-anchored quote-tracking benchmark, not an
independent fair-value estimate: past convertible-bond market prices enter the
correction through past observed gaps.

## Model scope

For each valuation date, the model builds a recombining stock tree, sets
terminal value to the greater of conversion parity and redemption, and rolls
the value backward with a parity-dependent blend of equity and credit discount
rates. Conversion parity per RMB 100 face value is:

```text
parity = stock price * 100 / conversion price
```

Important simplifications:

- call and put rules use immediate parity thresholds instead of full
  observation-window clauses;
- coupons are spread over tree steps rather than contractual payment dates;
- credit spread, risk-free rate, and dividend yield are constant per run;
- future conversion-price reset optionality is not represented in the tree;
- volatility is a scalar input rather than a surface.

## Tests

The suite does not require a live Wind request:

```text
python -m unittest discover -s tests -v
```
