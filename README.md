# Exchequer

**The UK OBR macroeconomic model, rewritten in Python.**

A full Python translation of the Office for Budget Responsibility's ~370-equation simultaneous macroeconomic model — the engine behind every UK Budget forecast. Originally written in EViews, now open and hackable.

## What is this?

The OBR model is the forecasting tool used by the UK's independent fiscal watchdog to produce economic and fiscal projections at each Budget. It covers:

- **GDP and expenditure** — consumption, investment, trade, inventories
- **Labour market** — employment, wages, unemployment, productivity
- **Prices** — CPI, RPI, import/export deflators, input-output cost structure
- **Public finances** — tax revenues, spending, borrowing (PSNB), debt (PSND)
- **Balance of payments** — current account, capital flows, exchange rates
- **Financial balance sheets** — household wealth, corporate liabilities, international investment position
- **North Sea oil** — production, trade, and fiscal revenues

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with synthetic placeholder data
python run_model.py
```

## Usage

```python
from obr_model.model import OBRModel

# Create and run the model
model = OBRModel()
model.generate_synthetic_data()       # or model.load_data(your_dataframe)
model.solve('2025Q1', '2030Q4')

# Inspect results
model.summary('2026Q4')
gdp = model.get_variable('GDPM', start='2025Q1')
print(gdp)

# Export
model.export_results('results.xlsx')  # or .csv
```

## Architecture

```
obr_model/
    config.py                # Solver parameters and constants
    eviews_functions.py      # EViews-to-Python translation helpers
    data_generator.py        # Synthetic placeholder data (~600 variables)
    solver.py                # Gauss-Seidel iterative solver
    model.py                 # Main OBRModel orchestrator
    equations/
        group01_consumption.py    # Household consumption and durables
        group02_inventories.py    # Stockbuilding
        group03_investment.py     # Business, housing, and public investment
        group04_labour.py         # Employment, unemployment, participation
        group05_exports.py        # Goods and services exports
        group06_imports.py        # Goods and services imports
        group07_prices.py         # Wages, costs, CPI, RPI, deflators
        group08_northsea.py       # North Sea oil production and trade
        group09_public_exp.py     # Government spending
        group10_receipts.py       # Tax revenues
        group11_bop.py            # Balance of payments and investment income
        group12_public_totals.py  # Fiscal aggregates (PSNB, PSND, debt)
        group14_financial.py      # Interest rates, money, equity prices
        group15_income.py         # Household income, saving, sectoral balances
        group16_gdp.py            # GDP identities and market sector GVA
        group18_balance_sheet.py  # Household, corporate, and RoW balance sheets
```

## How the Solver Works

The model is a system of ~370 simultaneous equations. For each forecast quarter:

1. **Initialise** endogenous variables from the previous quarter
2. **Pre-solve** the SCOST/CCOST/UTCOST input-output cost block (3×3 linear system)
3. **Iterate** through all equations (Gauss-Seidel with damping), updating each variable in place
4. **Converge** when max relative change < 1e-8 (typically 50-60 iterations)
5. **Apply** additive adjustments for forecast alignment

## Using Real Data

The synthetic data lets you test the structure. To use real data:

1. Build a pandas DataFrame with a quarterly `PeriodIndex` (1970Q1 onwards)
2. Populate all ~150 exogenous variables (interest rates, oil prices, exchange rates, demographics, policy settings)
3. Populate historical values for all endogenous variables
4. Set additive adjustment variables (`_A` suffix) for near-term forecast alignment
5. Call `model.load_data(df)` instead of `model.generate_synthetic_data()`

Key data sources: ONS (national accounts), Bank of England (rates), OBR databank (fiscal), HMRC (tax receipts).

## Key Variables

| Variable | Description |
|----------|-------------|
| `GDPM` | Real GDP |
| `GDPMPS` | Nominal GDP |
| `CONS` | Real household consumption |
| `IF` | Real gross fixed capital formation |
| `CPI` | Consumer Price Index |
| `LFSUR` | ILO unemployment rate (%) |
| `PSNBCY` | Public sector net borrowing |
| `PSND` | Public sector net debt |
| `CB` | Current account balance |

## Acknowledgements

Based on the OBR's published model equations. The original EViews model is published by the OBR for transparency purposes.
