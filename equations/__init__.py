"""
Equation registry for the OBR macroeconomic model.

Collects all equations from all group modules in evaluation order.
"""

from . import group01_consumption
from . import group02_inventories
from . import group03_investment
from . import group04_labour
from . import group05_exports
from . import group06_imports
from . import group07_prices
from . import group08_northsea
from . import group09_public_exp
from . import group10_receipts
from . import group11_bop
from . import group12_public_totals
from . import group14_financial
from . import group15_income
from . import group16_gdp
from . import group18_balance_sheet


# Evaluation order follows the OBR group numbering.
# The Gauss-Seidel solver iterates over all equations multiple times,
# so exact ordering matters less than in a recursive model, but putting
# "upstream" groups first helps convergence speed.
_GROUP_MODULES = [
    group04_labour,          # Group 4: Labour market (feeds into wages)
    group07_prices,          # Group 7: Prices and wages (core simultaneous block)
    group15_income,          # Group 15: Income account
    group01_consumption,     # Group 1: Consumption
    group02_inventories,     # Group 2: Inventories
    group03_investment,      # Group 3: Investment
    group05_exports,         # Group 5: Exports
    group06_imports,         # Group 6: Imports
    group08_northsea,        # Group 8: North Sea Oil
    group16_gdp,             # Group 16: GDP + Group 17: Market sector GVA
    group09_public_exp,      # Group 9: Public expenditure
    group10_receipts,        # Group 10: Public sector receipts
    group11_bop,             # Group 11: Balance of payments
    group12_public_totals,   # Group 12: Public sector totals
    group14_financial,       # Group 14: Domestic financial sector
    group18_balance_sheet,   # Group 18: Financial balance sheets
]


def get_all_equations():
    """Return all equations from all groups in evaluation order."""
    all_equations = []
    for module in _GROUP_MODULES:
        all_equations.extend(module.get_equations())
    return all_equations
