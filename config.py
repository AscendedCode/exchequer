"""Configuration constants for the OBR model."""

import pandas as pd


# Solver parameters
SOLVER_MAX_ITER = 200
SOLVER_TOLERANCE = 1e-8
SOLVER_DAMPING = 0.7

# Time configuration
HISTORY_START = pd.Period('1970Q1', freq='Q')
HISTORY_END = pd.Period('2024Q4', freq='Q')
FORECAST_START = pd.Period('2025Q1', freq='Q')
FORECAST_END = pd.Period('2030Q4', freq='Q')

# Constants from commented-out equations in the original model
W1 = 0.084      # CPI rent weight
W4 = 0.024      # Mortgage interest weight in RPI
W5 = 0.172      # OOH weight in CPIH
I4 = 222.8      # Index base values
I7 = 317.7
I9 = 319.5
I10 = 115.1
I11 = 114.7
I12 = 111.2

# Additive adjustment variable names (from @ADD(V) directives)
ADDITIVE_ADJUSTMENTS = {
    'PRMIP': 'PRMIP_A',
    'PSNBCY': 'PSNBCY_A',
    'SBHH': 'SBHH_A',
    'TYWHH': 'TYWHH_A',
    'EESC': 'EESC_A',
    'MGDPNSA': 'MGDPNSA_A',
}
