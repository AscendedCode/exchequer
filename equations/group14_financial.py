"""
Group 14: Domestic Financial Sector
Lines 576-585 of obr_model.txt

Equations:
    RIC   - Interest rate on corporate lending (d)
    EQPR  - Equity prices (dlog)
    M0    - Notes and coin (dlog)
    M4IC  - M4 industrial & commercial companies (ratio)
    M4    - Broad money (identity)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# --------------------------------------------------------------------------- #
# RIC
# --------------------------------------------------------------------------- #
def compute_RIC(data, t):
    """
    d(RIC) = 0.755375 * d(R) - 0.286805 * (RIC(-1) - 0.822845 * R(-1) - 2.583124)

    EViews original (line 576):
    d(RIC) = 0.755375 * d(R) - 0.286805 * (RIC(-1) - 0.822845 * R(-1) - 2.583124)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_R = v('R') - v('R', 1)
    ecm = v('RIC', 1) - 0.822845 * v('R', 1) - 2.583124

    rhs = 0.755375 * d_R - 0.286805 * ecm

    return v('RIC', 1) + rhs


# --------------------------------------------------------------------------- #
# EQPR
# --------------------------------------------------------------------------- #
def compute_EQPR(data, t):
    """
    dlog(EQPR) = dlog(GDPMPS)

    EViews original (line 578):
    dlog(EQPR) = dlog(GDPMPS)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    dlog_GDPMPS = np.log(v('GDPMPS') / v('GDPMPS', 1))

    return v('EQPR', 1) * np.exp(dlog_GDPMPS)


# --------------------------------------------------------------------------- #
# M0
# --------------------------------------------------------------------------- #
def compute_M0(data, t):
    """
    dlog(M0) = dlog(GDPMPS)

    EViews original (line 580):
    dlog(M0) = dlog(GDPMPS)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    dlog_GDPMPS = np.log(v('GDPMPS') / v('GDPMPS', 1))

    return v('M0', 1) * np.exp(dlog_GDPMPS)


# --------------------------------------------------------------------------- #
# M4IC
# --------------------------------------------------------------------------- #
def compute_M4IC(data, t):
    """
    M4IC / M4IC(-1) = GDPMPS / GDPMPS(-1)

    EViews original (line 582):
    M4IC / M4IC(-1) = GDPMPS / GDPMPS(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('M4IC', 1) * (v('GDPMPS') / v('GDPMPS', 1))


# --------------------------------------------------------------------------- #
# M4
# --------------------------------------------------------------------------- #
def compute_M4(data, t):
    """
    M4 = DEPHH + M4IC + M4OFC

    EViews original (line 584):
    M4 = DEPHH + M4IC + M4OFC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DEPHH') + v('M4IC') + v('M4OFC')


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('RIC',  compute_RIC,  'd'),
        ('EQPR', compute_EQPR, 'dlog'),
        ('M0',   compute_M0,   'dlog'),
        ('M4IC', compute_M4IC, 'ratio'),
        ('M4',   compute_M4,   'identity'),
    ]
