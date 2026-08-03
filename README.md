# Convertible Bond Daily Tracker

A research-oriented Python project for tracking Chinese exchange-listed
convertible bonds and comparing their market prices with theoretical values
from a simplified Cox–Ross–Rubinstein (CRR) binomial-tree model.

The application retrieves bond terms and historical market data from Wind,
builds a daily valuation table, prices each usable trading day, and exports a
CSV report and diagnostic chart.

> [!IMPORTANT]
> This project is for research and model development. The current CRR engine
> simplifies several prospectus clauses and is not a production trading or
> risk-management system.

## Features

- Retrieves bond, underlying-stock, and contract data through WindPy.
- Uses historically effective conversion prices and coupon rates.
- Estimates rolling historical volatility without using future observations.
- Reprices each day with a CRR binomial tree.
- Separates market-data access, numerical pricing, and reporting logic.
- Records row-level pricing errors instead of discarding the entire run.
- Compares the base valuation with a no-call approximation.
- Exports an Excel-friendly UTF-8 CSV and a PNG tracking chart.

## Requirements

- Windows with the Wind Financial Terminal installed
- An active Wind account and a logged-in Wind terminal session
- Python 3.12 or later
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management
- WindPy installed for the Python environment used by this project

WindPy is distributed with the Wind terminal and is therefore not declared in
`pyproject.toml` or downloaded from PyPI. Follow the Wind installation
instructions to install or expose WindPy to the project environment.


## Configuration

The current command-line entry point is
[`scripts/dayTrack_crr.py`](scripts/dayTrack_crr.py). Before running it, review
the three configuration blocks in `main()`.

### 1. Wind fields

`WindFieldConfig` contains the WSS and WSD field names used to retrieve:

- the underlying stock code;
- maturity and exercise-period dates;
- maturity redemption value;
- historical conversion prices; and
- historical coupon rates.

Verify these names and their units with the Wind code generator. In particular,
the historical fields must reflect the value effective on each valuation date,
not repeat the latest value across the full history.

### 2. Tracking settings

Edit `TrackingConfig` to select the bond, date range, and volatility window:

```python
tracking = TrackingConfig(
    bond_code="123117.SZ",
    start="2025-07-21",
    end="2026-07-21",
    volatility_window=60,
)
```

The underlying stock code is obtained from Wind rather than entered manually,
which helps prevent bond/stock mismatches.

### 3. Model assumptions

Edit `ManualModelTerms` to set the interest-rate, credit-spread, tree, and
simplified clause assumptions:

```python
manual_terms = ManualModelTerms(
    risk_free_rate=0.02,
    dividend_yield=0.0,
    credit_spread=0.01,
    debt_equity_blend_low=70.0,
    debt_equity_blend_high=130.0,
    call_parity_trigger=130.0,
    put_parity_trigger=70.0,
    put_price=103.0,
    tree_steps=200,
)
```

Rates are annual decimal values, so `0.02` represents 2%. Check the call, put,
and put-price terms against the selected bond's prospectus before interpreting
the results.

## Usage

By default, results are written to `output/`:

```text
output/
├── daily_tracking_<bond_code>.csv
└── daily_tracking_<bond_code>.png
```

The CSV contains the aligned market inputs, model inputs, theoretical price,
no-call theoretical price, estimated call impact, and a `pricing_error` column
for dates that could not be valued.

To investigate a persistent model/market gap without querying Wind again, run:

```text
python scripts/gap_diagnostics.py
```

The offline diagnostics use the saved nine-bond batch table and write
volatility/credit-spread sensitivities, no-call implied volatility, conversion
exercise diagnostics, and generic downward-reset trigger proximity to
`output/diagnostics/gap/`.  The sensitivity grids are sampled across the full
date range to keep this research-only run separate from normal batch cost.

Batch runs additionally write percentage-based cross-bond rankings,
`comparison_summary_by_style.csv`, and `failures.csv`.  The style summary uses
the latest conversion parity and the configured blend thresholds to classify
each bond as `debt`, `balanced`, or `equity`.  Cross-bond fit rankings use MAPE,
percentage gap, and percentage-gap stability rather than absolute RMB errors.
Wind failures retain the request type, actual requested Wind code, fields,
vendor error code, returned columns and a short response preview.


## Model overview

For each valuation date, the model:

1. Builds a recombining CRR stock-price tree.
2. Sets the terminal bond value to the greater of conversion parity and
   maturity redemption value.
3. Rolls the value backward with a parity-dependent blend of equity and credit
   discount rates.
4. Allows conversion after the contractual conversion start date.
5. Applies simplified call and put parity thresholds.

Conversion parity is calculated per RMB 100 face value:

```text
parity = stock price × 100 / conversion price
```

## Current limitations

- Call and put provisions use immediate parity thresholds rather than
  prospectus-level consecutive-day or observation-window tests.
- Coupons are spread uniformly across tree steps instead of being paid on
  contractual cash-flow dates.
- Credit spread is constant.
- Future conversion-price reset optionality is not represented inside the tree.
- Historical volatility is used instead of an implied-volatility surface.
- The risk-free rate and dividend yield are constant during a run.

These approximations can materially affect theoretical values. Treat the output
as a diagnostic research estimate rather than a definitive fair value.
