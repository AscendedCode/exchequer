"""
Group 2: Inventories
Lines 19-33 of obr_model.txt

Equations:
    DINV   - Change in inventories, volume (identity)
    INV    - Level of inventories, volume (identity)
    BV     - Book value of inventories (identity)
    SA     - Stock appreciation (identity)
    DINVPS - Change in inventories, current prices (identity)
    DINVHH - Household share of inventory investment (identity)
    DINVCG - Central government share of inventory investment (identity)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# --------------------------------------------------------------------------- #
# DINV
# --------------------------------------------------------------------------- #
def compute_DINV(data, t):
    """DINV = (GDPM + M - SDE) - CGG - CONS - VAL - IF - X"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return ((v('GDPM') + v('M') - v('SDE'))
            - v('CGG') - v('CONS') - v('VAL') - v('IF') - v('X'))


# --------------------------------------------------------------------------- #
# INV
# --------------------------------------------------------------------------- #
def compute_INV(data, t):
    """INV = INV(-1) + DINV"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('INV', 1) + v('DINV')


# --------------------------------------------------------------------------- #
# BV
# --------------------------------------------------------------------------- #
def compute_BV(data, t):
    """BV = BV(-1) + DINVPS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('BV', 1) + v('DINVPS')


# --------------------------------------------------------------------------- #
# SA
# --------------------------------------------------------------------------- #
def compute_SA(data, t):
    """SA = BV(-1) * (PINV / PINV(-1) - 1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('BV', 1) * (v('PINV') / v('PINV', 1) - 1)


# --------------------------------------------------------------------------- #
# DINVPS
# --------------------------------------------------------------------------- #
def compute_DINVPS(data, t):
    """DINVPS = DINV * PDINV / 100"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DINV') * v('PDINV') / 100


# --------------------------------------------------------------------------- #
# DINVHH
# --------------------------------------------------------------------------- #
def compute_DINVHH(data, t):
    """DINVHH = 0.07 * DINVPS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.07 * v('DINVPS')


# --------------------------------------------------------------------------- #
# DINVCG
# --------------------------------------------------------------------------- #
def compute_DINVCG(data, t):
    """
    DINVCG = PSNI - CGIPS - LAIPS - IPCPS - IBPC
             - (NPACG + NPALA)
             - (KCGPSO - KPSCG)
             - (KLA - KGLAPC - KGLA)
             - (KPCPS - KPSPC)
             - ASSETSA + DEP + ASSETSA
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('PSNI')
            - v('CGIPS') - v('LAIPS') - v('IPCPS') - v('IBPC')
            - (v('NPACG') + v('NPALA'))
            - (v('KCGPSO') - v('KPSCG'))
            - (v('KLA') - v('KGLAPC') - v('KGLA'))
            - (v('KPCPS') - v('KPSPC'))
            - v('ASSETSA') + v('DEP') + v('ASSETSA'))


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('DINV',   compute_DINV,   'identity'),
        ('INV',    compute_INV,    'identity'),
        ('BV',     compute_BV,     'identity'),
        ('SA',     compute_SA,     'identity'),
        ('DINVPS', compute_DINVPS, 'identity'),
        ('DINVHH', compute_DINVHH, 'identity'),
        ('DINVCG', compute_DINVCG, 'identity'),
    ]
