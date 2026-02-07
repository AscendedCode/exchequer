"""
Group 1: Consumption
Lines 4-13 of obr_model.txt

Equations:
    CONS   - Household consumption (dlog)
    CONSPS - Consumption in current prices (identity)
    CDUR   - Durable consumption (dlog)
    CDURPS - Durable consumption in current prices (identity)
    PD     - Price of durables / house prices proxy (dlog)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# --------------------------------------------------------------------------- #
# CONS
# --------------------------------------------------------------------------- #
def compute_CONS(data, t):
    """
    dlog(CONS) = 0.2645906 + 0.1029795 * dlog(RHHDI)
                 - 0.0083736 * d(LFSUR)
                 + 0.1269445 * dlog((GPW * 1000) / (PCE / 100))
                 - 0.0004036 * d(R(-1) - (-1 + PCE / PCE(-4)) * 100)
                 - 0.1250582 * ( log(CONS(-1))
                     - 0.4392933 * log(RHHDI(-1))
                     - 0.1059181 * log((GPW(-1) * 1000) / (PCE(-1) / 100))
                     - 0.2215558 * log( NFWPE(-1) / (PCE(-1) / 100) ) )
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    # dlog(RHHDI)
    dlog_RHHDI = np.log(v('RHHDI') / v('RHHDI', 1))

    # d(LFSUR)
    d_LFSUR = v('LFSUR') - v('LFSUR', 1)

    # dlog((GPW * 1000) / (PCE / 100))
    real_gpw_t = (v('GPW') * 1000) / (v('PCE') / 100)
    real_gpw_t1 = (v('GPW', 1) * 1000) / (v('PCE', 1) / 100)
    dlog_real_gpw = np.log(real_gpw_t / real_gpw_t1)

    # d(R(-1) - (-1 + PCE / PCE(-4)) * 100)
    # Real interest rate at t-1: R(-1) - (-1 + PCE(-1)/PCE(-5)) * 100
    # Real interest rate at t-2: R(-2) - (-1 + PCE(-2)/PCE(-6)) * 100
    # d() means the change, so value at t minus value at t-1
    # But note R(-1) means R lagged once from perspective of the equation.
    # In EViews: d(R(-1) - (-1 + PCE/PCE(-4))*100) means:
    #   [R(-1) - (-1 + PCE/PCE(-4))*100] - [R(-2) - (-1 + PCE(-1)/PCE(-5))*100]
    real_r_t = v('R', 1) - (-1 + v('PCE') / v('PCE', 4)) * 100
    real_r_t1 = v('R', 2) - (-1 + v('PCE', 1) / v('PCE', 5)) * 100
    d_real_r = real_r_t - real_r_t1

    # Error correction term
    ecm = (safe_log(v('CONS', 1))
           - 0.4392933 * safe_log(v('RHHDI', 1))
           - 0.1059181 * safe_log((v('GPW', 1) * 1000) / (v('PCE', 1) / 100))
           - 0.2215558 * safe_log(v('NFWPE', 1) / (v('PCE', 1) / 100)))

    rhs = (0.2645906
           + 0.1029795 * dlog_RHHDI
           - 0.0083736 * d_LFSUR
           + 0.1269445 * dlog_real_gpw
           - 0.0004036 * d_real_r
           - 0.1250582 * ecm)

    return v('CONS', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# CONSPS
# --------------------------------------------------------------------------- #
def compute_CONSPS(data, t):
    """CONSPS = CONS * PCE / 100"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CONS') * v('PCE') / 100


# --------------------------------------------------------------------------- #
# CDUR
# --------------------------------------------------------------------------- #
def compute_CDUR(data, t):
    """
    dlog(CDUR) = dlog(CONS)
                 - 0.6408491 * (dlog(PCDUR) - dlog(PCE))
                 + 0.0378296 * dlog(PD)
                 + 0.4517152 * dlog(RHHDI)
                 + 0.3438288 * dlog(RHHDI(-1))
                 - 0.0421498 * log(CDUR(-1) / CONS(-1))
                 - 0.0145656 * log(PCDUR(-1) * ((((1+R(-1)/100)^0.25)-1)
                     + ((1.25^0.25)-1) - d(PCDUR(-1))/PCDUR(-1)) / 100)
                 + 0.0313983 * log(NFWPE(-1) / (PCE(-1) / 100))
                 - 0.6203775
                 + 0.0636941 * (@recode(@date=@dateval("2009:04"),1,0)
                     - @recode(@date=@dateval("2010:01"),1,0))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    dlog_CONS = np.log(v('CONS') / v('CONS', 1))
    dlog_PCDUR = np.log(v('PCDUR') / v('PCDUR', 1))
    dlog_PCE = np.log(v('PCE') / v('PCE', 1))
    dlog_PD = np.log(v('PD') / v('PD', 1))
    dlog_RHHDI = np.log(v('RHHDI') / v('RHHDI', 1))
    dlog_RHHDI_1 = np.log(v('RHHDI', 1) / v('RHHDI', 2))

    # User cost term inside log
    nom_rate = ((1 + v('R', 1) / 100) ** 0.25) - 1
    depr_rate = (1.25 ** 0.25) - 1
    d_PCDUR_1 = v('PCDUR', 1) - v('PCDUR', 2)  # d(PCDUR(-1))
    cap_gain = d_PCDUR_1 / v('PCDUR', 1)
    user_cost = v('PCDUR', 1) * (nom_rate + depr_rate - cap_gain) / 100

    # Wealth term
    log_wealth = safe_log(v('NFWPE', 1) / (v('PCE', 1) / 100))

    # Dummies
    dummy = (recode_eq(t, "2009:04") - recode_eq(t, "2010:01"))

    rhs = (dlog_CONS
           - 0.6408491 * (dlog_PCDUR - dlog_PCE)
           + 0.0378296 * dlog_PD
           + 0.4517152 * dlog_RHHDI
           + 0.3438288 * dlog_RHHDI_1
           - 0.0421498 * safe_log(v('CDUR', 1) / v('CONS', 1))
           - 0.0145656 * safe_log(user_cost)
           + 0.0313983 * log_wealth
           - 0.6203775
           + 0.0636941 * dummy)

    return v('CDUR', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# CDURPS
# --------------------------------------------------------------------------- #
def compute_CDURPS(data, t):
    """CDURPS = (PCDUR / 100) * CDUR"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('PCDUR') / 100) * v('CDUR')


# --------------------------------------------------------------------------- #
# PD
# --------------------------------------------------------------------------- #
def compute_PD(data, t):
    """
    dlog(PD) = dlog(GPW / APH)
               - 0.1278181 * log(PD(-1) / (GPW(-1) / APH(-1)))
               + 1.54494 * (dlog(APH) - dlog(PCE))
               + 0.2058841 * (@recode(@date=@dateval("1992:03"),1,0)
                   - @recode(@date=@dateval("1992:04"),1,0))
               + 0.340128 * @recode(@date=@dateval("2004:01"),1,0)
               + 0.1437075 * (@recode(@date=@dateval("2009:04"),1,0)
                   - @recode(@date=@dateval("2010:01"),1,0))
               + 0.2732277 * (@recode(@date=@dateval("2016:01"),1,0)
                   - @recode(@date=@dateval("2016:02"),1,0))
               + 0.2217687
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    # dlog(GPW / APH)
    ratio_t = v('GPW') / v('APH')
    ratio_t1 = v('GPW', 1) / v('APH', 1)
    dlog_ratio = np.log(ratio_t / ratio_t1)

    # Error correction
    ecm = safe_log(v('PD', 1) / (v('GPW', 1) / v('APH', 1)))

    # dlog(APH) - dlog(PCE)
    dlog_APH = np.log(v('APH') / v('APH', 1))
    dlog_PCE = np.log(v('PCE') / v('PCE', 1))

    # Dummies
    d1 = recode_eq(t, "1992:03") - recode_eq(t, "1992:04")
    d2 = recode_eq(t, "2004:01")
    d3 = recode_eq(t, "2009:04") - recode_eq(t, "2010:01")
    d4 = recode_eq(t, "2016:01") - recode_eq(t, "2016:02")

    rhs = (dlog_ratio
           - 0.1278181 * ecm
           + 1.54494 * (dlog_APH - dlog_PCE)
           + 0.2058841 * d1
           + 0.340128 * d2
           + 0.1437075 * d3
           + 0.2732277 * d4
           + 0.2217687)

    return v('PD', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('CONS',   compute_CONS,   'dlog'),
        ('CONSPS', compute_CONSPS, 'identity'),
        ('CDUR',   compute_CDUR,   'dlog'),
        ('CDURPS', compute_CDURPS, 'identity'),
        ('PD',     compute_PD,     'dlog'),
    ]
