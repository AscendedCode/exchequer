"""
Group 9: Public Expenditure
Lines 369-404 of obr_model.txt

Equations:
    CGWS   - Central government wages and salaries (identity)
    LAWS   - Local authority wages and salaries (identity)
    OSGG   - Other sub-sector of general government (identity)
    CGP    - Central government procurement (identity)
    GGFCD  - General government final consumption deflator (identity)
    CGG    - Central government goods (dlog)
    CGTSUB - Central government total subsidies (identity)
    LASUBPR - Local authority subsidies (4Q average scaled by PGDP)
    LATSUB - Local authority total subsidies (identity)
    CGASC  - CG administrative social contributions (ratio)
    CGISC  - CG imputed social contributions (ratio)
    EESCCG - Employers' social contributions CG (ratio)
    LASC   - LA social contributions (ratio)
    EESCLA - Employers' social contributions LA (ratio)
    CGNCGA - CG non-cash grants (identity)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# --------------------------------------------------------------------------- #
# CGWS
# --------------------------------------------------------------------------- #
def compute_CGWS(data, t):
    """CGWS  = CGWADJ  * ERCG  * ECG  * (52  / 4000)  * (1  + (1.249  * EMPSC  / WFP))"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('CGWADJ') * v('ERCG') * v('ECG') * (52 / 4000)
            * (1 + (1.249 * v('EMPSC') / v('WFP'))))


# --------------------------------------------------------------------------- #
# LAWS
# --------------------------------------------------------------------------- #
def compute_LAWS(data, t):
    """LAWS  = LAWADJ  * ERLA  * ELA  * (52  / 4000)  * (1  + (1.418  * EMPSC  / WFP))"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('LAWADJ') * v('ERLA') * v('ELA') * (52 / 4000)
            * (1 + (1.418 * v('EMPSC') / v('WFP'))))


# --------------------------------------------------------------------------- #
# OSGG
# --------------------------------------------------------------------------- #
def compute_OSGG(data, t):
    """OSGG  = RCGIM  + RLAIM  + 100"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('RCGIM') + v('RLAIM') + 100


# --------------------------------------------------------------------------- #
# CGP
# --------------------------------------------------------------------------- #
def compute_CGP(data, t):
    """CGP  = CGGPSPSF  - (CGWS  + LAWS)  - LAPR  - (RCGIM  + RLAIM)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('CGGPSPSF') - (v('CGWS') + v('LAWS'))
            - v('LAPR') - (v('RCGIM') + v('RLAIM')))


# --------------------------------------------------------------------------- #
# GGFCD
# --------------------------------------------------------------------------- #
def compute_GGFCD(data, t):
    """GGFCD  = 100  * CGGPS  / CGG"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 100 * v('CGGPS') / v('CGG')


# --------------------------------------------------------------------------- #
# CGG
# --------------------------------------------------------------------------- #
def compute_CGG(data, t):
    """
    dlog(CGG)  = 0.0007011  + 0.3739498  * dlog(CGGPS)
                 + 0.1802323  * dlog(CGGPS(-1))
                 - 0.4198339  * dlog(CGG(-1))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    dlog_CGGPS = np.log(v('CGGPS') / v('CGGPS', 1))
    dlog_CGGPS_1 = np.log(v('CGGPS', 1) / v('CGGPS', 2))
    dlog_CGG_1 = np.log(v('CGG', 1) / v('CGG', 2))

    rhs = (0.0007011
           + 0.3739498 * dlog_CGGPS
           + 0.1802323 * dlog_CGGPS_1
           - 0.4198339 * dlog_CGG_1)

    return v('CGG', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# CGTSUB
# --------------------------------------------------------------------------- #
def compute_CGTSUB(data, t):
    """CGTSUB  = CGSUBP  + CGSUBPR"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGSUBP') + v('CGSUBPR')


# --------------------------------------------------------------------------- #
# LASUBPR
# --------------------------------------------------------------------------- #
def compute_LASUBPR(data, t):
    """
    LASUBPR  = (LASUBPR(-4)  + LASUBPR(-3)  + LASUBPR(-2)  + LASUBPR(-1))
               * 0.25  * (PGDP  * 4)
               / (PGDP(-4)  + PGDP(-3)  + PGDP(-2)  + PGDP(-1))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    lasubpr_avg = (v('LASUBPR', 4) + v('LASUBPR', 3)
                   + v('LASUBPR', 2) + v('LASUBPR', 1))

    pgdp_avg = (v('PGDP', 4) + v('PGDP', 3)
                + v('PGDP', 2) + v('PGDP', 1))

    return lasubpr_avg * 0.25 * (v('PGDP') * 4) / pgdp_avg


# --------------------------------------------------------------------------- #
# LATSUB
# --------------------------------------------------------------------------- #
def compute_LATSUB(data, t):
    """LATSUB  = LASUBP  + LASUBPR"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('LASUBP') + v('LASUBPR')


# --------------------------------------------------------------------------- #
# CGASC
# --------------------------------------------------------------------------- #
def compute_CGASC(data, t):
    """CGASC  / CGASC(-1)  = CGWS  / CGWS(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGASC', 1) * (v('CGWS') / v('CGWS', 1))


# --------------------------------------------------------------------------- #
# CGISC
# --------------------------------------------------------------------------- #
def compute_CGISC(data, t):
    """CGISC  / CGISC(-1)  = CGWS  / CGWS(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGISC', 1) * (v('CGWS') / v('CGWS', 1))


# --------------------------------------------------------------------------- #
# EESCCG
# --------------------------------------------------------------------------- #
def compute_EESCCG(data, t):
    """EESCCG  / EESCCG(-1)  = CGWS  / CGWS(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EESCCG', 1) * (v('CGWS') / v('CGWS', 1))


# --------------------------------------------------------------------------- #
# LASC
# --------------------------------------------------------------------------- #
def compute_LASC(data, t):
    """LASC  / LASC(-1)  = LAWS  / LAWS(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('LASC', 1) * (v('LAWS') / v('LAWS', 1))


# --------------------------------------------------------------------------- #
# EESCLA
# --------------------------------------------------------------------------- #
def compute_EESCLA(data, t):
    """EESCLA  / EESCLA(-1)  = LAWS  / LAWS(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EESCLA', 1) * (v('LAWS') / v('LAWS', 1))


# --------------------------------------------------------------------------- #
# CGNCGA
# --------------------------------------------------------------------------- #
def compute_CGNCGA(data, t):
    """CGNCGA  = TROD"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('TROD')


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('CGWS',    compute_CGWS,    'identity'),
        ('LAWS',    compute_LAWS,    'identity'),
        ('OSGG',    compute_OSGG,    'identity'),
        ('CGP',     compute_CGP,     'identity'),
        ('GGFCD',   compute_GGFCD,   'identity'),
        ('CGG',     compute_CGG,     'dlog'),
        ('CGTSUB',  compute_CGTSUB,  'identity'),
        ('LASUBPR', compute_LASUBPR, 'identity'),
        ('LATSUB',  compute_LATSUB,  'identity'),
        ('CGASC',   compute_CGASC,   'ratio'),
        ('CGISC',   compute_CGISC,   'ratio'),
        ('EESCCG',  compute_EESCCG,  'ratio'),
        ('LASC',    compute_LASC,    'ratio'),
        ('EESCLA',  compute_EESCLA,  'ratio'),
        ('CGNCGA',  compute_CGNCGA,  'identity'),
    ]
