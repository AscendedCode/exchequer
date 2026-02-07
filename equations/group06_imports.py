"""
Group 6: Imports of Goods & Services
OBR macroeconomic model equations (lines 195-225).
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# ---------------------------------------------------------------------------
# MC = 0.257 * CONS
# ---------------------------------------------------------------------------
def compute_MC(data, t):
    """MC = 0.257 * CONS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return 0.257 * v('CONS')


# ---------------------------------------------------------------------------
# MCGG = 0.094 * CGG
# ---------------------------------------------------------------------------
def compute_MCGG(data, t):
    """MCGG = 0.094 * CGG"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return 0.094 * v('CGG')


# ---------------------------------------------------------------------------
# MIF = 0.234 * IF
# ---------------------------------------------------------------------------
def compute_MIF(data, t):
    """MIF = 0.234 * IF"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return 0.234 * v('IF')


# ---------------------------------------------------------------------------
# MDINV = 0.106 * (DINV - ALAD)
# ---------------------------------------------------------------------------
def compute_MDINV(data, t):
    """MDINV = 0.106 * (DINV - ALAD)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return 0.106 * (v('DINV') - v('ALAD'))


# ---------------------------------------------------------------------------
# MXS = 0.142 * XS
# ---------------------------------------------------------------------------
def compute_MXS(data, t):
    """MXS = 0.142 * XS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return 0.142 * v('XS')


# ---------------------------------------------------------------------------
# MXG = 0.376 * (XOIL + XNOG)
# ---------------------------------------------------------------------------
def compute_MXG(data, t):
    """MXG = 0.376 * (XOIL + XNOG)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return 0.376 * (v('XOIL') + v('XNOG'))


# ---------------------------------------------------------------------------
# MTFE = MC + MCGG + MIF + MDINV + MXS + MXG
# ---------------------------------------------------------------------------
def compute_MTFE(data, t):
    """MTFE = MC + MCGG + MIF + MDINV + MXS + MXG"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return v('MC') + v('MCGG') + v('MIF') + v('MDINV') + v('MXS') + v('MXG')


# ---------------------------------------------------------------------------
# MINTY = 100 * M / MTFE
# ---------------------------------------------------------------------------
def compute_MINTY(data, t):
    """MINTY = 100 * (M) / MTFE"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return 100.0 * v('M') / v('MTFE')


# ---------------------------------------------------------------------------
# MGTFE = 0.176*CONS + 0.064*CGG + 0.175*IF + 0.094*DINV
#        + 0.410*XNOG + 0.049*XS
# ---------------------------------------------------------------------------
def compute_MGTFE(data, t):
    """MGTFE = 0.176*CONS + 0.064*CGG + 0.175*IF + 0.094*DINV + 0.410*XNOG + 0.049*XS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return (0.176 * v('CONS')
            + 0.064 * v('CGG')
            + 0.175 * v('IF')
            + 0.094 * v('DINV')
            + 0.410 * v('XNOG')
            + 0.049 * v('XS'))


# ---------------------------------------------------------------------------
# PMGREL = PMNOG / (0.156*PCE + 0.097*GGFCD + 0.203*PIF + 0.096*PINV
#                   + 0.352*PXNOG + 0.063*PXS)
# ---------------------------------------------------------------------------
def compute_PMGREL(data, t):
    """PMGREL = PMNOG / (0.156*PCE + 0.097*GGFCD + 0.203*PIF + 0.096*PINV
    + 0.352*PXNOG + 0.063*PXS)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    denominator = (0.156 * v('PCE')
                   + 0.097 * v('GGFCD')
                   + 0.203 * v('PIF')
                   + 0.096 * v('PINV')
                   + 0.352 * v('PXNOG')
                   + 0.063 * v('PXS'))
    return v('PMNOG') / denominator


# ---------------------------------------------------------------------------
# MNOG = M - MS - MOIL
# ---------------------------------------------------------------------------
def compute_MNOG(data, t):
    """MNOG = M - MS - MOIL"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return v('M') - v('MS') - v('MOIL')


# ---------------------------------------------------------------------------
# MSTFE = 0.081*CONS + 0.030*CGG + 0.059*IF + 0.012*DINV
#        + 0.029*XNOG + 0.093*XS
# ---------------------------------------------------------------------------
def compute_MSTFE(data, t):
    """MSTFE = 0.081*CONS + 0.030*CGG + 0.059*IF + 0.012*DINV + 0.029*XNOG + 0.093*XS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return (0.081 * v('CONS')
            + 0.030 * v('CGG')
            + 0.059 * v('IF')
            + 0.012 * v('DINV')
            + 0.029 * v('XNOG')
            + 0.093 * v('XS'))


# ---------------------------------------------------------------------------
# PMSREL = PMS / (0.060*PCE + 0.040*GGFCD + 0.067*PIF + 0.040*PINV
#                + 0.024*PXNOG + 0.098*PXS)
# ---------------------------------------------------------------------------
def compute_PMSREL(data, t):
    """PMSREL = PMS / (0.060*PCE + 0.040*GGFCD + 0.067*PIF + 0.040*PINV
    + 0.024*PXNOG + 0.098*PXS)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    denominator = (0.060 * v('PCE')
                   + 0.040 * v('GGFCD')
                   + 0.067 * v('PIF')
                   + 0.040 * v('PINV')
                   + 0.024 * v('PXNOG')
                   + 0.098 * v('PXS'))
    return v('PMS') / denominator


# ---------------------------------------------------------------------------
# dlog(MS) = 0.819114*dlog(MSTFE) + 0.389511*dlog(MSTFE(-1))
#   - 0.525436*dlog(MSTFE(-2)) + 0.288639*dlog(MSTFE(-3))
#   - 0.477411*dlog(PMSREL) - 0.292804*dlog(PMSREL(-1))
#   - 0.271392*dlog(MS(-1))
#   - 0.171294*(log(MS(-1)) - 1.079017*log(MSTFE(-1))
#       - 0.662445*log(SPECX(-1))
#       + 0.112661*((@recode(@date >= @dateval("2007:01"), 1, 0)) * SPECX)
#       + 0.874335*log(PMSREL(-1))
#       - 0.126418*(@recode(@date >= @dateval("2007:01"), 1, 0)
#                   - @recode(@date >= @dateval("2013:01"), 1, 0)))
#   - 0.031665
# ---------------------------------------------------------------------------
def compute_MS(data, t):
    """dlog(MS) = 0.819114*dlog(MSTFE) + 0.389511*dlog(MSTFE(-1))
    - 0.525436*dlog(MSTFE(-2)) + 0.288639*dlog(MSTFE(-3))
    - 0.477411*dlog(PMSREL) - 0.292804*dlog(PMSREL(-1))
    - 0.271392*dlog(MS(-1))
    - 0.171294*(log(MS(-1)) - 1.079017*log(MSTFE(-1))
        - 0.662445*log(SPECX(-1))
        + 0.112661*((@recode(@date >= @dateval("2007:01"), 1, 0)) * SPECX)
        + 0.874335*log(PMSREL(-1))
        - 0.126418*(@recode(@date >= @dateval("2007:01"), 1, 0)
                    - @recode(@date >= @dateval("2013:01"), 1, 0)))
    - 0.031665"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    # Short-run dynamics: dlog terms
    dlog_MSTFE_0 = safe_log(v('MSTFE')) - safe_log(v('MSTFE', 1))
    dlog_MSTFE_1 = safe_log(v('MSTFE', 1)) - safe_log(v('MSTFE', 2))
    dlog_MSTFE_2 = safe_log(v('MSTFE', 2)) - safe_log(v('MSTFE', 3))
    dlog_MSTFE_3 = safe_log(v('MSTFE', 3)) - safe_log(v('MSTFE', 4))

    dlog_PMSREL_0 = safe_log(v('PMSREL')) - safe_log(v('PMSREL', 1))
    dlog_PMSREL_1 = safe_log(v('PMSREL', 1)) - safe_log(v('PMSREL', 2))

    dlog_MS_1 = safe_log(v('MS', 1)) - safe_log(v('MS', 2))

    # Step dummies
    d2007 = recode_geq(t, "2007:01")
    d2013 = recode_geq(t, "2013:01")

    # Error-correction mechanism (ECM) term
    ecm = (safe_log(v('MS', 1))
           - 1.079017 * safe_log(v('MSTFE', 1))
           - 0.662445 * safe_log(v('SPECX', 1))
           + 0.112661 * (d2007 * v('SPECX'))
           + 0.874335 * safe_log(v('PMSREL', 1))
           - 0.126418 * (d2007 - d2013))

    rhs = (0.819114 * dlog_MSTFE_0
           + 0.389511 * dlog_MSTFE_1
           - 0.525436 * dlog_MSTFE_2
           + 0.288639 * dlog_MSTFE_3
           - 0.477411 * dlog_PMSREL_0
           - 0.292804 * dlog_PMSREL_1
           - 0.271392 * dlog_MS_1
           - 0.171294 * ecm
           - 0.031665)

    return v('MS', 1) * np.exp(rhs)


# ---------------------------------------------------------------------------
# MPS = MNOG * (PMNOG / 100) + MS * (PMS / 100) + MOIL * (PMOIL / 100)
# ---------------------------------------------------------------------------
def compute_MPS(data, t):
    """MPS = MNOG * (PMNOG / 100) + MS * (PMS / 100) + MOIL * (PMOIL / 100)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return (v('MNOG') * (v('PMNOG') / 100.0)
            + v('MS') * (v('PMS') / 100.0)
            + v('MOIL') * (v('PMOIL') / 100.0))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('MC',     compute_MC,     'identity'),
        ('MCGG',   compute_MCGG,   'identity'),
        ('MIF',    compute_MIF,    'identity'),
        ('MDINV',  compute_MDINV,  'identity'),
        ('MXS',    compute_MXS,    'identity'),
        ('MXG',    compute_MXG,    'identity'),
        ('MTFE',   compute_MTFE,   'identity'),
        ('MINTY',  compute_MINTY,  'identity'),
        ('MGTFE',  compute_MGTFE,  'identity'),
        ('PMGREL', compute_PMGREL, 'identity'),
        ('MNOG',   compute_MNOG,   'identity'),
        ('MSTFE',  compute_MSTFE,  'identity'),
        ('PMSREL', compute_PMSREL, 'identity'),
        ('MS',     compute_MS,     'dlog'),
        ('MPS',    compute_MPS,    'identity'),
    ]
