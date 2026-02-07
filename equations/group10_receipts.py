"""
Group 10: Public Sector Receipts
Lines 408-432 of obr_model.txt

Equations:
    CT       - Council tax (identity)
    CETAX    - Customs and excise taxes (identity)
    VED      - Vehicle excise duty (identity)
    OCT      - Other current transfers (identity)
    CGC      - Central government consumption of fixed capital (d/ratio)
    PSINTR   - Public sector interest receipts (identity)
    CGRENT   - Central government rent (identity)
    TAXCRED  - Tax credits (identity)
    INCTAXG  - Income tax gross (identity)
    PUBSTIW  - Public sector taxes on income and wealth (identity)
    PUBSTPD  - Public sector taxes on production (identity)
    PSCR     - Public sector current receipts (identity)
    NATAXES  - National accounts taxes (identity)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# --------------------------------------------------------------------------- #
# CT
# --------------------------------------------------------------------------- #
def compute_CT(data, t):
    """CT  = NSCTP  + NNSCTP"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('NSCTP') + v('NNSCTP')


# --------------------------------------------------------------------------- #
# CETAX
# --------------------------------------------------------------------------- #
def compute_CETAX(data, t):
    """CETAX  = VREC  + TXFUEL  + TXTOB  + TXALC  + CUST  + CCL  + AL  + TXCUS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('VREC') + v('TXFUEL') + v('TXTOB') + v('TXALC')
            + v('CUST') + v('CCL') + v('AL') + v('TXCUS'))


# --------------------------------------------------------------------------- #
# VED
# --------------------------------------------------------------------------- #
def compute_VED(data, t):
    """VED  = VEDHH  + VEDCO"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('VEDHH') + v('VEDCO')


# --------------------------------------------------------------------------- #
# OCT
# --------------------------------------------------------------------------- #
def compute_OCT(data, t):
    """OCT  = VEDHH  + BBC  + PASSPORT  + OHT"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('VEDHH') + v('BBC') + v('PASSPORT') + v('OHT')


# --------------------------------------------------------------------------- #
# CGC
# --------------------------------------------------------------------------- #
def compute_CGC(data, t):
    """
    d(CGC)  / CGC(-1)  = 0.21  * d(ROCB)  / ROCB(-1)

    Rearranged: CGC = CGC(-1) + CGC(-1) * 0.21 * (ROCB - ROCB(-1)) / ROCB(-1)
              = CGC(-1) * (1 + 0.21 * (ROCB - ROCB(-1)) / ROCB(-1))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_ROCB = v('ROCB') - v('ROCB', 1)

    return v('CGC', 1) * (1 + 0.21 * d_ROCB / v('ROCB', 1))


# --------------------------------------------------------------------------- #
# PSINTR
# --------------------------------------------------------------------------- #
def compute_PSINTR(data, t):
    """PSINTR  = CGNDIV  + LANDIV  + PCNDIV"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGNDIV') + v('LANDIV') + v('PCNDIV')


# --------------------------------------------------------------------------- #
# CGRENT
# --------------------------------------------------------------------------- #
def compute_CGRENT(data, t):
    """CGRENT  = RNCG  + HHTCG"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('RNCG') + v('HHTCG')


# --------------------------------------------------------------------------- #
# TAXCRED
# --------------------------------------------------------------------------- #
def compute_TAXCRED(data, t):
    """TAXCRED  = MILAPM  + CTC"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('MILAPM') + v('CTC')


# --------------------------------------------------------------------------- #
# INCTAXG
# --------------------------------------------------------------------------- #
def compute_INCTAXG(data, t):
    """INCTAXG  = TYEM  + TSEOP  + TCINV  - INCTAC  + CTC  - NPISHTC"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('TYEM') + v('TSEOP') + v('TCINV')
            - v('INCTAC') + v('CTC') - v('NPISHTC'))


# --------------------------------------------------------------------------- #
# PUBSTIW
# --------------------------------------------------------------------------- #
def compute_PUBSTIW(data, t):
    """
    PUBSTIW  = TYEM  + TSEOP  + PRT  + TCINV  + CT  + CGT  + FCACA
               + BETPRF  + BETLEVY  + OFGEM  - NPISHTC  - TYPCO  + PROV  - LAEPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('TYEM') + v('TSEOP') + v('PRT') + v('TCINV')
            + v('CT') + v('CGT') + v('FCACA')
            + v('BETPRF') + v('BETLEVY') + v('OFGEM')
            - v('NPISHTC') - v('TYPCO') + v('PROV') - v('LAEPS'))


# --------------------------------------------------------------------------- #
# PUBSTPD
# --------------------------------------------------------------------------- #
def compute_PUBSTPD(data, t):
    """
    PUBSTPD  = (CETAX  - BETPRF)  + EXDUTAC  + XLAVAT  + LAVAT  - EUOT
               + TSD  + ROCS  + TXMIS  + RFP
               + (NNDRA  + VEDCO  + LAPT  + OPT  + EUETS)
               + CIL  + ENVLEVY  + BANKROLL  + RULC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return ((v('CETAX') - v('BETPRF'))
            + v('EXDUTAC') + v('XLAVAT') + v('LAVAT') - v('EUOT')
            + v('TSD') + v('ROCS') + v('TXMIS') + v('RFP')
            + (v('NNDRA') + v('VEDCO') + v('LAPT') + v('OPT') + v('EUETS'))
            + v('CIL') + v('ENVLEVY') + v('BANKROLL') + v('RULC'))


# --------------------------------------------------------------------------- #
# PSCR
# --------------------------------------------------------------------------- #
def compute_PSCR(data, t):
    """
    PSCR  = PUBSTIW  + PUBSTPD  + OCT  + CC  + INHT  + EENIC  + EMPNIC
            + (RCGIM  + RLAIM  + OSPC)  + PSINTR  + (RNCG  + HHTCG)
            + LARENT  + PCRENT  + BLEVY  + LAEPS  + SWISSCAP
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('PUBSTIW') + v('PUBSTPD') + v('OCT')
            + v('CC') + v('INHT') + v('EENIC') + v('EMPNIC')
            + (v('RCGIM') + v('RLAIM') + v('OSPC'))
            + v('PSINTR') + (v('RNCG') + v('HHTCG'))
            + v('LARENT') + v('PCRENT') + v('BLEVY')
            + v('LAEPS') + v('SWISSCAP'))


# --------------------------------------------------------------------------- #
# NATAXES
# --------------------------------------------------------------------------- #
def compute_NATAXES(data, t):
    """
    NATAXES  = PUBSTIW  + PUBSTPD  + OCT  + BLEVY  + INHT  + LAEPS
               + SWISSCAP  + EENIC  + EMPNIC  + CC  + EUOT
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('PUBSTIW') + v('PUBSTPD') + v('OCT')
            + v('BLEVY') + v('INHT') + v('LAEPS') + v('SWISSCAP')
            + v('EENIC') + v('EMPNIC') + v('CC') + v('EUOT'))


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('CT',      compute_CT,      'identity'),
        ('CETAX',   compute_CETAX,   'identity'),
        ('VED',     compute_VED,     'identity'),
        ('OCT',     compute_OCT,     'identity'),
        ('CGC',     compute_CGC,     'd_ratio'),
        ('PSINTR',  compute_PSINTR,  'identity'),
        ('CGRENT',  compute_CGRENT,  'identity'),
        ('TAXCRED', compute_TAXCRED, 'identity'),
        ('INCTAXG', compute_INCTAXG, 'identity'),
        ('PUBSTIW', compute_PUBSTIW, 'identity'),
        ('PUBSTPD', compute_PUBSTPD, 'identity'),
        ('PSCR',    compute_PSCR,    'identity'),
        ('NATAXES', compute_NATAXES, 'identity'),
    ]
