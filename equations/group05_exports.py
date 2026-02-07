"""
Group 5: Exports of Goods & Services
OBR macroeconomic model equations (lines 184-190).
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# ---------------------------------------------------------------------------
# XNOG = X - XS - XOIL
# ---------------------------------------------------------------------------
def compute_XNOG(data, t):
    """XNOG = X - XS - XOIL"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return v('X') - v('XS') - v('XOIL')


# ---------------------------------------------------------------------------
# dlog(RPRICE) = dlog(PXNOG) + dlog(RXD) - 0.9351684 * dlog(WPG)
# ---------------------------------------------------------------------------
def compute_RPRICE(data, t):
    """dlog(RPRICE) = dlog(PXNOG) + dlog(RXD) - 0.9351684 * dlog(WPG)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    dlog_PXNOG = safe_log(v('PXNOG')) - safe_log(v('PXNOG', 1))
    dlog_RXD = safe_log(v('RXD')) - safe_log(v('RXD', 1))
    dlog_WPG = safe_log(v('WPG')) - safe_log(v('WPG', 1))

    rhs = dlog_PXNOG + dlog_RXD - 0.9351684 * dlog_WPG

    return v('RPRICE', 1) * np.exp(rhs)


# ---------------------------------------------------------------------------
# XPS = (PXNOG / 100) * XNOG + (PXS / 100) * XS + (PXOIL / 100) * XOIL
# ---------------------------------------------------------------------------
def compute_XPS(data, t):
    """XPS = (PXNOG / 100) * XNOG + (PXS / 100) * XS + (PXOIL / 100) * XOIL"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return ((v('PXNOG') / 100.0) * v('XNOG')
            + (v('PXS') / 100.0) * v('XS')
            + (v('PXOIL') / 100.0) * v('XOIL'))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('XNOG',   compute_XNOG,   'identity'),
        ('RPRICE', compute_RPRICE, 'dlog'),
        ('XPS',    compute_XPS,    'identity'),
    ]
