"""
Group 8: North Sea Oil
Lines 353-364 of obr_model.txt

Equations:
    TDOIL  - Total demand for oil (dlog, ECM)
    MOIL   - Oil imports (identity)
    PXOIL  - Price of oil exports (dlog)
    PMOIL  - Price of oil imports (dlog)
    NSGTP  - North Sea gross trading profits (ratio)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# --------------------------------------------------------------------------- #
# TDOIL
# --------------------------------------------------------------------------- #
def compute_TDOIL(data, t):
    """
    dlog(TDOIL)  =  - 0.2444325  * dlog(TDOIL(-1))
                    + 1.896486  * dlog(NNSGVA(-1))
                    - 0.1077816  * (log(PBRENT  / (RXD  * (GDPMPS(-1)  - BPAPS(-1)
                        - (NSGVA(-1)  * PBRENT(-1)  / (OILBASE  * RXD(-1))))
                        / NNSGVA(-1) ) )
                        - log(PBRENT(-1)  / (RXD(-1)  * (GDPMPS(-2)  - BPAPS(-2)
                        - (NSGVA(-2)  * PBRENT(-2)  / (OILBASE  * RXD(-2))))
                        / NNSGVA(-2) ) ) )
                    + 0.0780697  * (@recode(@date >= @dateval("1984:01")  , 1  , 0)
                        * @recode(@date <= @dateval("1985:01")  , 1  , 0))
                    - 0.0143727
                    - 0.2216107  * (@recode(@date  = @dateval("1986:01")  , 1  , 0)
                        - @recode(@date  = @dateval("1986:02")  , 1  , 0))
                    - 0.2457494  * (@recode(@date  = @dateval("2001:03")  , 1  , 0)
                        - @recode(@date  = @dateval("2001:04")  , 1  , 0))
                    + 0.1907036  * (@recode(@date  = @dateval("2010:03")  , 1  , 0)
                        - @recode(@date  = @dateval("2010:04")  , 1  , 0))
                    - 0.4334139  * @recode(@date  = @dateval("2013:01")  , 1  , 0)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    # dlog(TDOIL(-1))
    dlog_TDOIL_1 = np.log(v('TDOIL', 1) / v('TDOIL', 2))

    # dlog(NNSGVA(-1))
    dlog_NNSGVA_1 = np.log(v('NNSGVA', 1) / v('NNSGVA', 2))

    # ECM difference term:
    # log(PBRENT / (RXD * (GDPMPS(-1) - BPAPS(-1) - (NSGVA(-1)*PBRENT(-1)/(OILBASE*RXD(-1)))) / NNSGVA(-1)))
    # minus the same expression lagged once
    ecm_inner_t = (v('GDPMPS', 1) - v('BPAPS', 1)
                   - (v('NSGVA', 1) * v('PBRENT', 1) / (v('OILBASE') * v('RXD', 1))))
    ecm_t = safe_log(v('PBRENT') / (v('RXD') * ecm_inner_t / v('NNSGVA', 1)))

    ecm_inner_t1 = (v('GDPMPS', 2) - v('BPAPS', 2)
                    - (v('NSGVA', 2) * v('PBRENT', 2) / (v('OILBASE') * v('RXD', 2))))
    ecm_t1 = safe_log(v('PBRENT', 1) / (v('RXD', 1) * ecm_inner_t1 / v('NNSGVA', 2)))

    d_ecm = ecm_t - ecm_t1

    # Dummy: 1984Q1 to 1985Q1 inclusive
    dum_8485 = recode_geq(t, "1984:01") * recode_leq(t, "1985:01")

    # Dummy pairs
    dum_86 = recode_eq(t, "1986:01") - recode_eq(t, "1986:02")
    dum_01 = recode_eq(t, "2001:03") - recode_eq(t, "2001:04")
    dum_10 = recode_eq(t, "2010:03") - recode_eq(t, "2010:04")
    dum_13 = recode_eq(t, "2013:01")

    rhs = (- 0.2444325 * dlog_TDOIL_1
           + 1.896486 * dlog_NNSGVA_1
           - 0.1077816 * d_ecm
           + 0.0780697 * dum_8485
           - 0.0143727
           - 0.2216107 * dum_86
           - 0.2457494 * dum_01
           + 0.1907036 * dum_10
           - 0.4334139 * dum_13)

    return v('TDOIL', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# MOIL
# --------------------------------------------------------------------------- #
def compute_MOIL(data, t):
    """MOIL  = TDOIL  + XOIL  - NSGVA"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('TDOIL') + v('XOIL') - v('NSGVA')


# --------------------------------------------------------------------------- #
# PXOIL
# --------------------------------------------------------------------------- #
def compute_PXOIL(data, t):
    """dlog(PXOIL)  = dlog(PBRENT)  - dlog(RXD)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    dlog_PBRENT = np.log(v('PBRENT') / v('PBRENT', 1))
    dlog_RXD = np.log(v('RXD') / v('RXD', 1))

    rhs = dlog_PBRENT - dlog_RXD

    return v('PXOIL', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# PMOIL
# --------------------------------------------------------------------------- #
def compute_PMOIL(data, t):
    """dlog(PMOIL)  = dlog(PXOIL)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    dlog_PXOIL = np.log(v('PXOIL') / v('PXOIL', 1))

    return v('PMOIL', 1) * np.exp(dlog_PXOIL)


# --------------------------------------------------------------------------- #
# NSGTP
# --------------------------------------------------------------------------- #
def compute_NSGTP(data, t):
    """NSGTP  / NSGTP(-1)  = (NSGVA  / NSGVA(-1) )  * (PBRENT  / PBRENT(-1))  / (RXD  / RXD(-1))"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    ratio = ((v('NSGVA') / v('NSGVA', 1))
             * (v('PBRENT') / v('PBRENT', 1))
             / (v('RXD') / v('RXD', 1)))

    return v('NSGTP', 1) * ratio


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('TDOIL', compute_TDOIL, 'dlog'),
        ('MOIL',  compute_MOIL,  'identity'),
        ('PXOIL', compute_PXOIL, 'dlog'),
        ('PMOIL', compute_PMOIL, 'dlog'),
        ('NSGTP', compute_NSGTP, 'ratio'),
    ]
