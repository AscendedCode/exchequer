"""
Group 18: Financial Account and Financial Balance Sheet
Lines 765-918 of obr_model.txt

Equations:
    --- Household Financial Accounts (lines 768-803) ---
    NAFHHNSA - Household net financial account (NSA, identity)
    SDLHH    - Statistical discrepancy households (identity)
    NLHH     - Household net lending (identity)
    GMF      - Gross mortgage flows ratio (identity)
    DEPHHx   - Household deposits excl. adj (d)
    DEPHH    - Household deposits (d)
    NAEQHHx  - Net acquisition of equities by HH excl. adj (identity)
    NAEQHH   - Net acquisition of equities by HH (identity)
    EQHH     - Household equity holdings (identity)
    NAPEN    - Net acquisition of pensions (identity)
    NAINSx   - Net acquisition of insurance excl. adj (identity)
    NAINS    - Net acquisition of insurance (identity)
    PIHH     - Pensions & insurance assets HH (identity)
    DBR      - Discount bond rate (identity)
    OAHHx    - Other assets HH excl. adj (dlog)
    OAHH     - Other assets HH (d)
    GFWPE    - Gross financial wealth (identity)

    --- Household Financial Liabilities (lines 808-825) ---
    NAOLPEx  - Net acquisition of other lending to persons excl. adj (identity)
    NAOLPE   - Net acquisition of other lending to persons (identity)
    DEBTU    - Unsecured debt growth rate (identity)
    OLPEx    - Other lending to persons excl. adj (identity)
    OLPE     - Other lending to persons (identity)
    AAHH     - Acquisition of financial assets HH (identity)
    ALHH     - Acquisition of financial liabilities HH (identity)
    HHRES    - Household residual (identity)
    OAHHADJ  - Other assets HH adjustment (identity)

    --- Aggregates (lines 829-831) ---
    NFWPE    - Net financial wealth (identity)
    GPW      - Gross personal wealth (identity)

    --- Rest of World (lines 836-889) ---
    NAFROWNSA - ROW net financial account (NSA, identity)
    SDLROW    - Statistical discrepancy ROW (identity)
    NLROW     - ROW net lending (identity)
    DAROW     - Deposits assets ROW (d)
    EQAROW    - Equity assets ROW (identity)
    NAEQAROW  - Net acquisition of equity assets ROW (identity)
    BAROW     - Bond assets ROW (identity)
    NABAROW   - Net acquisition of bond assets ROW (identity)
    OTAROW    - Other assets ROW (identity)
    NAOTAROW  - Net acquisition of other assets ROW (identity)
    AROW      - Total assets ROW (identity)
    AAROW     - Acquisition of assets ROW (identity)
    DLROW     - Deposit liabilities ROW (identity)
    NADLROW   - Net acquisition of deposit liabilities ROW (identity)
    EQLROW    - Equity liabilities ROW (identity)
    NAEQLROW  - Net acquisition of equity liabilities ROW (identity)
    BLROW     - Bond liabilities ROW (identity)
    NABLROW   - Net acquisition of bond liabilities ROW (identity)
    OTLROW    - Other liabilities ROW (identity)
    NAOTLROW  - Net acquisition of other liabilities ROW (identity)
    LROW      - Total liabilities ROW (identity)
    ALROW     - Acquisition of liabilities ROW (identity)
    NIIP      - Net international investment position (d)

    --- PNFC Balance Sheet (lines 894-918) ---
    BLIC      - Bank lending to IC companies (identity)
    STLIC     - Short-term liabilities IC (identity)
    FXLIC     - Foreign currency liabilities IC (identity)
    EQLIC     - Equity liabilities IC (identity)
    OLIC      - Other liabilities IC (identity)
    LIC       - Total liabilities IC (identity)
    NABLIC    - Net acquisition of bank lending IC (identity)
    NAFXLIC   - Net acquisition of FX liabilities IC (identity)
    NAEQLIC   - Net acquisition of equity liabilities IC (identity)
    NALIC     - Net acquisition of liabilities IC (identity)
    AIC       - Assets IC (identity)
    NAAIC     - Net acquisition of assets IC (identity)
    NWIC      - Net worth IC (identity)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# =========================================================================== #
#  HOUSEHOLD FINANCIAL ACCOUNTS  (lines 768-803)
# =========================================================================== #


# --------------------------------------------------------------------------- #
# NAFHHNSA
# --------------------------------------------------------------------------- #
def compute_NAFHHNSA(data, t):
    """
    NAFHHNSA = NAFHH + NAFHH(-1) + NAFHH(-2) + NAFHH(-3)
               - NAFHHNSA(-1) - NAFHHNSA(-2) - NAFHHNSA(-3)

    EViews original (line 768):
    NAFHHNSA = NAFHH + NAFHH(-1) + NAFHH(-2) + NAFHH(-3)
               - NAFHHNSA(-1) - NAFHHNSA(-2) - NAFHHNSA(-3)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('NAFHH') + v('NAFHH', 1) + v('NAFHH', 2) + v('NAFHH', 3)
            - v('NAFHHNSA', 1) - v('NAFHHNSA', 2) - v('NAFHHNSA', 3))


# --------------------------------------------------------------------------- #
# SDLHH
# --------------------------------------------------------------------------- #
def compute_SDLHH(data, t):
    """
    SDLHH = 0

    EViews original (line 770):
    SDLHH = 0
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0


# --------------------------------------------------------------------------- #
# NLHH
# --------------------------------------------------------------------------- #
def compute_NLHH(data, t):
    """
    NLHH = NAFHHNSA - SDLHH

    EViews original (line 772):
    NLHH = NAFHHNSA - SDLHH
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('NAFHHNSA') - v('SDLHH')


# --------------------------------------------------------------------------- #
# GMF
# --------------------------------------------------------------------------- #
def compute_GMF(data, t):
    """
    GMF = (PD * APH * 0.858) / DEPHH(-1)

    EViews original (line 777):
    GMF = (PD * APH * 0.858) / DEPHH(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('PD') * v('APH') * 0.858) / v('DEPHH', 1)


# --------------------------------------------------------------------------- #
# DEPHHx
# --------------------------------------------------------------------------- #
def compute_DEPHHx(data, t):
    """
    d(DEPHHx) = 3.9056 * d(CONSPS)
                + exp(5.1811 * (RDEP - R)) - exp(5.1811 * (RDEP(-1) - R(-1)))
                + exp(0.8206 * LFSUR) - exp(0.8206 * LFSUR(-1))
                + exp(106.3011 * GMF)
                - 0.0369 * (DEPHH(-1) - 5.5399 * CONSPS(-1)
                            - exp(0.8479 * RDEP(-1))
                            - exp(1.0821 * LFSUR(-1)) + 233379.6)

    EViews original (line 779):
    d(DEPHHx) = 3.9056 * d(CONSPS) + exp(5.1811 * (RDEP - R))
                - exp(5.1811 * (RDEP(-1) - R(-1)))
                + exp(0.8206 * LFSUR) - exp(0.8206 * LFSUR(-1))
                + exp(106.3011 * GMF)
                - 0.0369 * (DEPHH(-1) - 5.5399 * CONSPS(-1)
                            - exp(0.8479 * RDEP(-1))
                            - exp(1.0821 * LFSUR(-1)) + 233379.6)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_CONSPS = v('CONSPS') - v('CONSPS', 1)

    ecm = (v('DEPHH', 1)
           - 5.5399 * v('CONSPS', 1)
           - np.exp(0.8479 * v('RDEP', 1))
           - np.exp(1.0821 * v('LFSUR', 1))
           + 233379.6)

    rhs = (3.9056 * d_CONSPS
           + np.exp(5.1811 * (v('RDEP') - v('R')))
           - np.exp(5.1811 * (v('RDEP', 1) - v('R', 1)))
           + np.exp(0.8206 * v('LFSUR'))
           - np.exp(0.8206 * v('LFSUR', 1))
           + np.exp(106.3011 * v('GMF'))
           - 0.0369 * ecm)

    return v('DEPHHx', 1) + rhs


# --------------------------------------------------------------------------- #
# DEPHH
# --------------------------------------------------------------------------- #
def compute_DEPHH(data, t):
    """
    d(DEPHH) = (DEPHHx - DEPHHx(-1)) + DEPHHADJ

    EViews original (line 781):
    d(DEPHH) = (DEPHHx - DEPHHx(-1)) + DEPHHADJ
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    rhs = (v('DEPHHx') - v('DEPHHx', 1)) + v('DEPHHADJ')

    return v('DEPHH', 1) + rhs


# --------------------------------------------------------------------------- #
# NAEQHHx
# --------------------------------------------------------------------------- #
def compute_NAEQHHx(data, t):
    """
    NAEQHHx = 0.4560 * NLHH - 12867

    EViews original (line 783):
    NAEQHHx = 0.4560 * NLHH - 12867
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.4560 * v('NLHH') - 12867


# --------------------------------------------------------------------------- #
# NAEQHH
# --------------------------------------------------------------------------- #
def compute_NAEQHH(data, t):
    """
    NAEQHH = NAEQHHx + NAEQHHADJ

    EViews original (line 785):
    NAEQHH = NAEQHHx + NAEQHHADJ
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('NAEQHHx') + v('NAEQHHADJ')


# --------------------------------------------------------------------------- #
# EQHH
# --------------------------------------------------------------------------- #
def compute_EQHH(data, t):
    """
    EQHH = (1 + 0.844 * (EQPR / EQPR(-1) - 1)
              + 0.156 * ((WEQPR / WEQPR(-1)) / (RX / RX(-1)) - 1))
           * EQHH(-1) + NAEQHH

    EViews original (line 787):
    EQHH = (1 + 0.844 * (EQPR / EQPR(-1) - 1)
              + 0.156 * ((WEQPR / WEQPR(-1)) / (RX / RX(-1)) - 1))
           * EQHH(-1) + NAEQHH
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    revaluation = (1
                   + 0.844 * (v('EQPR') / v('EQPR', 1) - 1)
                   + 0.156 * ((v('WEQPR') / v('WEQPR', 1)) / (v('RX') / v('RX', 1)) - 1))

    return revaluation * v('EQHH', 1) + v('NAEQHH')


# --------------------------------------------------------------------------- #
# NAPEN
# --------------------------------------------------------------------------- #
def compute_NAPEN(data, t):
    """
    NAPEN = NEAHH

    EViews original (line 789):
    NAPEN = NEAHH
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('NEAHH')


# --------------------------------------------------------------------------- #
# NAINSx
# --------------------------------------------------------------------------- #
def compute_NAINSx(data, t):
    """
    NAINSx = 13293.71 + 0.627 * NAINSx(-1) - 236267.3 * SIPT(-3)

    EViews original (line 791):
    NAINSx = 13293.71 + 0.627 * (NAINSx(-1)) - 236267.3 * (SIPT(-3))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 13293.71 + 0.627 * v('NAINSx', 1) - 236267.3 * v('SIPT', 3)


# --------------------------------------------------------------------------- #
# NAINS
# --------------------------------------------------------------------------- #
def compute_NAINS(data, t):
    """
    NAINS = NAINSx + NAINSADJ

    EViews original (line 793):
    NAINS = NAINSx + NAINSADJ
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('NAINSx') + v('NAINSADJ')


# --------------------------------------------------------------------------- #
# PIHH
# --------------------------------------------------------------------------- #
def compute_PIHH(data, t):
    """
    PIHH = (1 + 0.200 * (EQPR / EQPR(-1) - 1)
              + 0.098 * (RX(-1) / RX - 1)
              + 0.170 * ((WEQPR / WEQPR(-1)) / (RX / RX(-1)) - 1)
              + 0.574 * (DBR / DBR(-1) - 1))
           * PIHH(-1) + NAPEN + NAINS

    EViews original (line 795):
    PIHH = (1 + 0.200 * ((EQPR / EQPR(-1)) - 1)
              + 0.098 * (RX(-1) / RX - 1)
              + 0.170 * ((WEQPR / WEQPR(-1)) / (RX / RX(-1)) - 1)
              + 0.574 * (DBR / DBR(-1) - 1))
           * PIHH(-1) + NAPEN + NAINS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    revaluation = (1
                   + 0.200 * (v('EQPR') / v('EQPR', 1) - 1)
                   + 0.098 * (v('RX', 1) / v('RX') - 1)
                   + 0.170 * ((v('WEQPR') / v('WEQPR', 1)) / (v('RX') / v('RX', 1)) - 1)
                   + 0.574 * (v('DBR') / v('DBR', 1) - 1))

    return revaluation * v('PIHH', 1) + v('NAPEN') + v('NAINS')


# --------------------------------------------------------------------------- #
# DBR
# --------------------------------------------------------------------------- #
def compute_DBR(data, t):
    """
    DBR = 1 / ((1 + RL / 100) ^ 15)

    EViews original (line 797):
    DBR = 1 / ((1 + (RL / 100))^(15))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 1.0 / ((1.0 + v('RL') / 100.0) ** 15)


# --------------------------------------------------------------------------- #
# OAHHx
# --------------------------------------------------------------------------- #
def compute_OAHHx(data, t):
    """
    dlog(OAHHx) = 1.6091 - 0.1607 * log(OAHHx(-1))
                  + 0.0169 * log(GDPMPS(-1))
                  - 0.57443 * (log(GDPMPS) - log(GDPMPS(-1)))
                  + 0.001796 * @TREND(1986Q4)

    EViews original (line 799):
    dlog(OAHHx) = 1.6091 - 0.1607 * log(OAHHx(-1))
                  + 0.0169 * log(GDPMPS(-1))
                  - 0.57443 * (log(GDPMPS) - log(GDPMPS(-1)))
                  + 0.001796 * @TREND(1986Q4)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    t_trend = trend(t, '1986Q4')

    rhs = (1.6091
           - 0.1607 * safe_log(v('OAHHx', 1))
           + 0.0169 * safe_log(v('GDPMPS', 1))
           - 0.57443 * (safe_log(v('GDPMPS')) - safe_log(v('GDPMPS', 1)))
           + 0.001796 * t_trend)

    return v('OAHHx', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# OAHH
# --------------------------------------------------------------------------- #
def compute_OAHH(data, t):
    """
    d(OAHH) = (OAHHx - OAHHx(-1)) + OAHHADJ

    EViews original (line 801):
    d(OAHH) = (OAHHx - OAHHx(-1)) + OAHHADJ
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    rhs = (v('OAHHx') - v('OAHHx', 1)) + v('OAHHADJ')

    return v('OAHH', 1) + rhs


# --------------------------------------------------------------------------- #
# GFWPE
# --------------------------------------------------------------------------- #
def compute_GFWPE(data, t):
    """
    GFWPE = DEPHH + EQHH + PIHH + OAHH

    EViews original (line 803):
    GFWPE = DEPHH + EQHH + PIHH + OAHH
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DEPHH') + v('EQHH') + v('PIHH') + v('OAHH')


# =========================================================================== #
#  HOUSEHOLD FINANCIAL LIABILITIES  (lines 808-825)
# =========================================================================== #


# --------------------------------------------------------------------------- #
# NAOLPEx
# --------------------------------------------------------------------------- #
def compute_NAOLPEx(data, t):
    """
    NAOLPEx = OLPEx(-1) * DEBTU

    EViews original (line 808):
    NAOLPEx = OLPEx(-1) * DEBTU
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('OLPEx', 1) * v('DEBTU')


# --------------------------------------------------------------------------- #
# NAOLPE
# --------------------------------------------------------------------------- #
def compute_NAOLPE(data, t):
    """
    NAOLPE = NAOLPEx + d(STUDENT) + NAOLPEADJ

    EViews original (line 810):
    NAOLPE = NAOLPEx + d(STUDENT) + NAOLPEADJ
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_STUDENT = v('STUDENT') - v('STUDENT', 1)

    return v('NAOLPEx') + d_STUDENT + v('NAOLPEADJ')


# --------------------------------------------------------------------------- #
# DEBTU
# --------------------------------------------------------------------------- #
def compute_DEBTU(data, t):
    """
    DEBTU = 0.0812616 + 0.4338504 * DEBTU(-1)
            - 0.0248383 * log(OLPEx(-1))
            + 0.013581 * log(CONSPS(-1))
            - 0.0014364 * LFSUR(-1)
            + 0.0143662 * log(PD(-1))

    EViews original (line 812):
    DEBTU = 0.0812616 + 0.4338504 * DEBTU(-1) - 0.0248383 * log(OLPEx(-1))
            + 0.013581 * log(CONSPS(-1)) - 0.0014364 * LFSUR(-1)
            + 0.0143662 * log(PD(-1))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (0.0812616
            + 0.4338504 * v('DEBTU', 1)
            - 0.0248383 * safe_log(v('OLPEx', 1))
            + 0.013581 * safe_log(v('CONSPS', 1))
            - 0.0014364 * v('LFSUR', 1)
            + 0.0143662 * safe_log(v('PD', 1)))


# --------------------------------------------------------------------------- #
# OLPEx
# --------------------------------------------------------------------------- #
def compute_OLPEx(data, t):
    """
    OLPEx = OLPEx(-1) - 0.00219 * OLPEx(-1) + NAOLPEx + NAOLPEADJ

    EViews original (line 814):
    OLPEx = OLPEx(-1) - 0.00219 * OLPEx(-1) + NAOLPEx + NAOLPEADJ
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('OLPEx', 1) - 0.00219 * v('OLPEx', 1) + v('NAOLPEx') + v('NAOLPEADJ')


# --------------------------------------------------------------------------- #
# OLPE
# --------------------------------------------------------------------------- #
def compute_OLPE(data, t):
    """
    OLPE = OLPEx + STUDENT

    EViews original (line 816):
    OLPE = OLPEx + STUDENT
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('OLPEx') + v('STUDENT')


# --------------------------------------------------------------------------- #
# AAHH
# --------------------------------------------------------------------------- #
def compute_AAHH(data, t):
    """
    AAHH = d(OAHH) + d(DEPHH) + NAEQHH + NAPEN + NAINS

    EViews original (line 818):
    AAHH = d(OAHH) + d(DEPHH) + NAEQHH + NAPEN + NAINS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_OAHH = v('OAHH') - v('OAHH', 1)
    d_DEPHH = v('DEPHH') - v('DEPHH', 1)

    return d_OAHH + d_DEPHH + v('NAEQHH') + v('NAPEN') + v('NAINS')


# --------------------------------------------------------------------------- #
# ALHH
# --------------------------------------------------------------------------- #
def compute_ALHH(data, t):
    """
    ALHH = NAOLPE + d(LHP)

    EViews original (line 820):
    ALHH = NAOLPE + d(LHP)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_LHP = v('LHP') - v('LHP', 1)

    return v('NAOLPE') + d_LHP


# --------------------------------------------------------------------------- #
# HHRES
# --------------------------------------------------------------------------- #
def compute_HHRES(data, t):
    """
    HHRES = NLHH - ((d(DEPHHx) + NAEQHHx + NAPEN + NAINSx + d(OAHHx))
                     - (NAOLPEx + d(STUDENT) + d(LHP)))

    EViews original (line 822):
    HHRES = NLHH - ((d(DEPHHx) + NAEQHHx + NAPEN + NAINSx + d(OAHHx))
                     - (NAOLPEx + d(STUDENT) + d(LHP)))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_DEPHHx = v('DEPHHx') - v('DEPHHx', 1)
    d_OAHHx = v('OAHHx') - v('OAHHx', 1)
    d_STUDENT = v('STUDENT') - v('STUDENT', 1)
    d_LHP = v('LHP') - v('LHP', 1)

    assets_flow = d_DEPHHx + v('NAEQHHx') + v('NAPEN') + v('NAINSx') + d_OAHHx
    liabilities_flow = v('NAOLPEx') + d_STUDENT + d_LHP

    return v('NLHH') - (assets_flow - liabilities_flow)


# --------------------------------------------------------------------------- #
# OAHHADJ
# --------------------------------------------------------------------------- #
def compute_OAHHADJ(data, t):
    """
    OAHHADJ = HHRES - DEPHHADJ - NAEQHHADJ - NAINSADJ + NAOLPEADJ

    EViews original (line 824):
    OAHHADJ = HHRES - DEPHHADJ - NAEQHHADJ - NAINSADJ + NAOLPEADJ
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('HHRES') - v('DEPHHADJ') - v('NAEQHHADJ') - v('NAINSADJ') + v('NAOLPEADJ')


# =========================================================================== #
#  AGGREGATES  (lines 829-831)
# =========================================================================== #


# --------------------------------------------------------------------------- #
# NFWPE
# --------------------------------------------------------------------------- #
def compute_NFWPE(data, t):
    """
    NFWPE = GFWPE - LHP - OLPE

    EViews original (line 829):
    NFWPE = GFWPE - LHP - OLPE
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GFWPE') - v('LHP') - v('OLPE')


# --------------------------------------------------------------------------- #
# GPW
# --------------------------------------------------------------------------- #
def compute_GPW(data, t):
    """
    GPW = 0.9933 * GPW(-1) * APH / APH(-1) + 0.001 * IHHPS

    EViews original (line 831):
    GPW = 0.9933 * GPW(-1) * APH / APH(-1) + .001 * IHHPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.9933 * v('GPW', 1) * v('APH') / v('APH', 1) + 0.001 * v('IHHPS')


# =========================================================================== #
#  REST OF WORLD  (lines 836-889)
# =========================================================================== #


# --------------------------------------------------------------------------- #
# NAFROWNSA
# --------------------------------------------------------------------------- #
def compute_NAFROWNSA(data, t):
    """
    NAFROWNSA = NAFROW + NAFROW(-1) + NAFROW(-2) + NAFROW(-3)
                - NAFROWNSA(-1) - NAFROWNSA(-2) - NAFROWNSA(-3)

    EViews original (line 836):
    NAFROWNSA = NAFROW + NAFROW(-1) + NAFROW(-2) + NAFROW(-3)
                - NAFROWNSA(-1) - NAFROWNSA(-2) - NAFROWNSA(-3)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('NAFROW') + v('NAFROW', 1) + v('NAFROW', 2) + v('NAFROW', 3)
            - v('NAFROWNSA', 1) - v('NAFROWNSA', 2) - v('NAFROWNSA', 3))


# --------------------------------------------------------------------------- #
# SDLROW
# --------------------------------------------------------------------------- #
def compute_SDLROW(data, t):
    """
    SDLROW = 0

    EViews original (line 838):
    SDLROW = 0
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0


# --------------------------------------------------------------------------- #
# NLROW
# --------------------------------------------------------------------------- #
def compute_NLROW(data, t):
    """
    NLROW = NAFROWNSA - SDLROW

    EViews original (line 840):
    NLROW = NAFROWNSA - SDLROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('NAFROWNSA') - v('SDLROW')


# --------------------------------------------------------------------------- #
# DAROW
# --------------------------------------------------------------------------- #
def compute_DAROW(data, t):
    """
    d(DAROW) = (0.3813 * (XPS + MPS) / TFEPS
                + 0.7067 * ICCPS / TFEPS - 0.1872) * TFEPS

    EViews original (line 845):
    d(DAROW) = (0.3813 * (XPS + MPS) / TFEPS + 0.7067 * ICCPS / TFEPS
                - 0.1872) * TFEPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    rhs = (0.3813 * (v('XPS') + v('MPS')) / v('TFEPS')
           + 0.7067 * v('ICCPS') / v('TFEPS')
           - 0.1872) * v('TFEPS')

    return v('DAROW', 1) + rhs


# --------------------------------------------------------------------------- #
# EQAROW
# --------------------------------------------------------------------------- #
def compute_EQAROW(data, t):
    """
    EQAROW = EQAROW(-1) * (EQPR / EQPR(-1)) + NAEQAROW

    EViews original (line 847):
    EQAROW = EQAROW(-1) * (EQPR / EQPR(-1)) + NAEQAROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EQAROW', 1) * (v('EQPR') / v('EQPR', 1)) + v('NAEQAROW')


# --------------------------------------------------------------------------- #
# NAEQAROW
# --------------------------------------------------------------------------- #
def compute_NAEQAROW(data, t):
    """
    NAEQAROW = (0.25 * (EQAROW(-1) + EQAROW(-2) + EQAROW(-3) + EQAROW(-4))
                / (0.25 * (EQAROW(-1) + EQAROW(-2) + EQAROW(-3) + EQAROW(-4))
                   + 0.25 * (BAROW(-1) + BAROW(-2) + BAROW(-3) + BAROW(-4))))
               * (AAROW - d(DAROW) - NAOTAROW)

    EViews original (line 849):
    NAEQAROW = (0.25 * (EQAROW(-1) + EQAROW(-2) + EQAROW(-3) + EQAROW(-4))
                / (0.25 * (EQAROW(-1) + EQAROW(-2) + EQAROW(-3) + EQAROW(-4))
                   + 0.25 * (BAROW(-1) + BAROW(-2) + BAROW(-3) + BAROW(-4))))
               * (AAROW - d(DAROW) - NAOTAROW)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    avg_eq = 0.25 * (v('EQAROW', 1) + v('EQAROW', 2) + v('EQAROW', 3) + v('EQAROW', 4))
    avg_ba = 0.25 * (v('BAROW', 1) + v('BAROW', 2) + v('BAROW', 3) + v('BAROW', 4))

    eq_share = avg_eq / (avg_eq + avg_ba)

    d_DAROW = v('DAROW') - v('DAROW', 1)

    return eq_share * (v('AAROW') - d_DAROW - v('NAOTAROW'))


# --------------------------------------------------------------------------- #
# BAROW
# --------------------------------------------------------------------------- #
def compute_BAROW(data, t):
    """
    BAROW = BAROW(-1) * (0.40 / (RX / RX(-1)) + (1 - 0.40)) + NABAROW

    EViews original (line 851):
    BAROW = BAROW(-1) * (0.40 / (RX / RX(-1)) + (1 - 0.40)) + NABAROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    rx_ratio = v('RX') / v('RX', 1)
    revaluation = 0.40 / rx_ratio + 0.60

    return v('BAROW', 1) * revaluation + v('NABAROW')


# --------------------------------------------------------------------------- #
# NABAROW
# --------------------------------------------------------------------------- #
def compute_NABAROW(data, t):
    """
    NABAROW = (0.25 * (BAROW(-1) + BAROW(-2) + BAROW(-3) + BAROW(-4))
               / (0.25 * (EQAROW(-1) + EQAROW(-2) + EQAROW(-3) + EQAROW(-4))
                  + 0.25 * (BAROW(-1) + BAROW(-2) + BAROW(-3) + BAROW(-4))))
              * (AAROW - d(DAROW) - NAOTAROW)

    EViews original (line 853):
    NABAROW = (0.25 * (BAROW(-1) + BAROW(-2) + BAROW(-3) + BAROW(-4))
               / (0.25 * (EQAROW(-1) + EQAROW(-2) + EQAROW(-3) + EQAROW(-4))
                  + 0.25 * (BAROW(-1) + BAROW(-2) + BAROW(-3) + BAROW(-4))))
              * (AAROW - d(DAROW) - NAOTAROW)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    avg_eq = 0.25 * (v('EQAROW', 1) + v('EQAROW', 2) + v('EQAROW', 3) + v('EQAROW', 4))
    avg_ba = 0.25 * (v('BAROW', 1) + v('BAROW', 2) + v('BAROW', 3) + v('BAROW', 4))

    ba_share = avg_ba / (avg_eq + avg_ba)

    d_DAROW = v('DAROW') - v('DAROW', 1)

    return ba_share * (v('AAROW') - d_DAROW - v('NAOTAROW'))


# --------------------------------------------------------------------------- #
# OTAROW
# --------------------------------------------------------------------------- #
def compute_OTAROW(data, t):
    """
    OTAROW = OTAROW(-1) * (0.84 / (RX / RX(-1)) + (1 - 0.84)) + NAOTAROW

    EViews original (line 855):
    OTAROW = OTAROW(-1) * (0.84 / (RX / RX(-1)) + (1 - 0.84)) + NAOTAROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    rx_ratio = v('RX') / v('RX', 1)
    revaluation = 0.84 / rx_ratio + 0.16

    return v('OTAROW', 1) * revaluation + v('NAOTAROW')


# --------------------------------------------------------------------------- #
# NAOTAROW
# --------------------------------------------------------------------------- #
def compute_NAOTAROW(data, t):
    """
    NAOTAROW = NAOTLROW

    EViews original (line 857):
    NAOTAROW = NAOTLROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('NAOTLROW')


# --------------------------------------------------------------------------- #
# AROW
# --------------------------------------------------------------------------- #
def compute_AROW(data, t):
    """
    AROW = DAROW + EQAROW + BAROW + OTAROW

    EViews original (line 859):
    AROW = DAROW + EQAROW + BAROW + OTAROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DAROW') + v('EQAROW') + v('BAROW') + v('OTAROW')


# --------------------------------------------------------------------------- #
# AAROW
# --------------------------------------------------------------------------- #
def compute_AAROW(data, t):
    """
    AAROW = ALROW + NLROW

    EViews original (line 861):
    AAROW = ALROW + NLROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('ALROW') + v('NLROW')


# --------------------------------------------------------------------------- #
# DLROW
# --------------------------------------------------------------------------- #
def compute_DLROW(data, t):
    """
    DLROW = DLROW(-1) / (RX / RX(-1)) + NADLROW

    EViews original (line 866):
    DLROW = DLROW(-1) / (RX / RX(-1)) + NADLROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    rx_ratio = v('RX') / v('RX', 1)

    return v('DLROW', 1) / rx_ratio + v('NADLROW')


# --------------------------------------------------------------------------- #
# NADLROW
# --------------------------------------------------------------------------- #
def compute_NADLROW(data, t):
    """
    NADLROW = DLROW(-1) * (-0.0375 - 0.2124 * DLROW(-1) / LROW(-1)
              - 0.2004 * (FYCPR(-1) + FISIMPS(-1)) / EQLIC
              + 0.1026 * WEQPR / WEQPR(-1))

    EViews original (line 868):
    NADLROW = DLROW(-1) * (-0.0375 - 0.2124 * DLROW(-1) / LROW(-1)
              - 0.2004 * (FYCPR(-1) + FISIMPS(-1)) / EQLIC
              + 0.1026 * WEQPR / WEQPR(-1))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    factor = (-0.0375
              - 0.2124 * v('DLROW', 1) / v('LROW', 1)
              - 0.2004 * (v('FYCPR', 1) + v('FISIMPS', 1)) / v('EQLIC')
              + 0.1026 * v('WEQPR') / v('WEQPR', 1))

    return v('DLROW', 1) * factor


# --------------------------------------------------------------------------- #
# EQLROW
# --------------------------------------------------------------------------- #
def compute_EQLROW(data, t):
    """
    EQLROW = EQLROW(-1) * (WEQPR / WEQPR(-1)) / (RX / RX(-1)) + NAEQLROW

    EViews original (line 870):
    EQLROW = EQLROW(-1) * (WEQPR / WEQPR(-1)) / (RX / RX(-1)) + NAEQLROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    weqpr_ratio = v('WEQPR') / v('WEQPR', 1)
    rx_ratio = v('RX') / v('RX', 1)

    return v('EQLROW', 1) * weqpr_ratio / rx_ratio + v('NAEQLROW')


# --------------------------------------------------------------------------- #
# NAEQLROW
# --------------------------------------------------------------------------- #
def compute_NAEQLROW(data, t):
    """
    NAEQLROW = 0.196 * (NAINS + NAPEN) + 0.132 * NAEQHH + 0.003 * GDPMPS

    EViews original (line 872):
    NAEQLROW = 0.196 * (NAINS + NAPEN) + 0.132 * NAEQHH + 0.003 * GDPMPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.196 * (v('NAINS') + v('NAPEN')) + 0.132 * v('NAEQHH') + 0.003 * v('GDPMPS')


# --------------------------------------------------------------------------- #
# BLROW
# --------------------------------------------------------------------------- #
def compute_BLROW(data, t):
    """
    BLROW = BLROW(-1) / (RX / RX(-1)) + NABLROW

    EViews original (line 874):
    BLROW = BLROW(-1) / (RX / RX(-1)) + NABLROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    rx_ratio = v('RX') / v('RX', 1)

    return v('BLROW', 1) / rx_ratio + v('NABLROW')


# --------------------------------------------------------------------------- #
# NABLROW
# --------------------------------------------------------------------------- #
def compute_NABLROW(data, t):
    """
    NABLROW = 0.17 * (NAINS + NAPEN) + 0.0325 * GDPMPS

    EViews original (line 876):
    NABLROW = 0.17 * (NAINS + NAPEN) + 0.0325 * GDPMPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.17 * (v('NAINS') + v('NAPEN')) + 0.0325 * v('GDPMPS')


# --------------------------------------------------------------------------- #
# OTLROW
# --------------------------------------------------------------------------- #
def compute_OTLROW(data, t):
    """
    OTLROW = OTLROW(-1) * (0.90 / (RX / RX(-1)) + (1 - 0.90)) + NAOTLROW

    EViews original (line 878):
    OTLROW = OTLROW(-1) * (0.90 / (RX / RX(-1)) + (1 - 0.90)) + NAOTLROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    rx_ratio = v('RX') / v('RX', 1)
    revaluation = 0.90 / rx_ratio + 0.10

    return v('OTLROW', 1) * revaluation + v('NAOTLROW')


# --------------------------------------------------------------------------- #
# NAOTLROW
# --------------------------------------------------------------------------- #
def compute_NAOTLROW(data, t):
    """
    NAOTLROW = OTLROW(-1) * ((GDPMPS / GDPMPS(-1)) - 1)

    EViews original (line 880):
    NAOTLROW = OTLROW(-1) * ((GDPMPS / GDPMPS(-1)) - 1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('OTLROW', 1) * (v('GDPMPS') / v('GDPMPS', 1) - 1)


# --------------------------------------------------------------------------- #
# LROW
# --------------------------------------------------------------------------- #
def compute_LROW(data, t):
    """
    LROW = DLROW + EQLROW + BLROW + OTLROW

    EViews original (line 882):
    LROW = DLROW + EQLROW + BLROW + OTLROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DLROW') + v('EQLROW') + v('BLROW') + v('OTLROW')


# --------------------------------------------------------------------------- #
# ALROW
# --------------------------------------------------------------------------- #
def compute_ALROW(data, t):
    """
    ALROW = NADLROW + NAEQLROW + NABLROW + NAOTLROW - DRES

    EViews original (line 884):
    ALROW = NADLROW + NAEQLROW + NABLROW + NAOTLROW - DRES
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('NADLROW') + v('NAEQLROW') + v('NABLROW') + v('NAOTLROW') - v('DRES')


# --------------------------------------------------------------------------- #
# NIIP
# --------------------------------------------------------------------------- #
def compute_NIIP(data, t):
    """
    d(NIIP) = d(LROW) + d(SRES) - d(AROW)

    EViews original (line 889):
    d(NIIP) = d(LROW) + d(SRES) - d(AROW)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_LROW = v('LROW') - v('LROW', 1)
    d_SRES = v('SRES') - v('SRES', 1)
    d_AROW = v('AROW') - v('AROW', 1)

    rhs = d_LROW + d_SRES - d_AROW

    return v('NIIP', 1) + rhs


# =========================================================================== #
#  PNFC BALANCE SHEET MODEL  (lines 894-918)
# =========================================================================== #


# --------------------------------------------------------------------------- #
# BLIC
# --------------------------------------------------------------------------- #
def compute_BLIC(data, t):
    """
    BLIC = BLIC(-1) + NABLIC

    EViews original (line 894):
    BLIC = BLIC(-1) + NABLIC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('BLIC', 1) + v('NABLIC')


# --------------------------------------------------------------------------- #
# STLIC
# --------------------------------------------------------------------------- #
def compute_STLIC(data, t):
    """
    STLIC = STLIC(-1) + 0.09 * NALIC

    EViews original (line 896):
    STLIC = STLIC(-1) + 0.09 * NALIC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('STLIC', 1) + 0.09 * v('NALIC')


# --------------------------------------------------------------------------- #
# FXLIC
# --------------------------------------------------------------------------- #
def compute_FXLIC(data, t):
    """
    FXLIC = FXLIC(-1) * (RX(-1) / RX) + NAFXLIC

    EViews original (line 898):
    FXLIC = FXLIC(-1) * (RX(-1) / RX) + NAFXLIC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('FXLIC', 1) * (v('RX', 1) / v('RX')) + v('NAFXLIC')


# --------------------------------------------------------------------------- #
# EQLIC
# --------------------------------------------------------------------------- #
def compute_EQLIC(data, t):
    """
    EQLIC = EQLIC(-1) * (EQPR / EQPR(-1)) + NAEQLIC

    EViews original (line 900):
    EQLIC = EQLIC(-1) * (EQPR / EQPR(-1)) + NAEQLIC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EQLIC', 1) * (v('EQPR') / v('EQPR', 1)) + v('NAEQLIC')


# --------------------------------------------------------------------------- #
# OLIC
# --------------------------------------------------------------------------- #
def compute_OLIC(data, t):
    """
    OLIC = OLIC(-1) + 0.04 * NALIC

    EViews original (line 902):
    OLIC = OLIC(-1) + 0.04 * NALIC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('OLIC', 1) + 0.04 * v('NALIC')


# --------------------------------------------------------------------------- #
# LIC
# --------------------------------------------------------------------------- #
def compute_LIC(data, t):
    """
    LIC = BLIC + STLIC + FXLIC + EQLIC + OLIC

    EViews original (line 904):
    LIC = BLIC + STLIC + FXLIC + EQLIC + OLIC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('BLIC') + v('STLIC') + v('FXLIC') + v('EQLIC') + v('OLIC')


# --------------------------------------------------------------------------- #
# NABLIC
# --------------------------------------------------------------------------- #
def compute_NABLIC(data, t):
    """
    NABLIC = 0.14 * NALIC

    EViews original (line 906):
    NABLIC = 0.14 * NALIC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.14 * v('NALIC')


# --------------------------------------------------------------------------- #
# NAFXLIC
# --------------------------------------------------------------------------- #
def compute_NAFXLIC(data, t):
    """
    NAFXLIC = 0.07 * NALIC

    EViews original (line 908):
    NAFXLIC = 0.07 * NALIC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.07 * v('NALIC')


# --------------------------------------------------------------------------- #
# NAEQLIC
# --------------------------------------------------------------------------- #
def compute_NAEQLIC(data, t):
    """
    NAEQLIC = (1.6035 + 0.9385 * EQLIC(-1) / (FYCPR(-1) + FISIMPS(-1)))
              * (FYCPR + FISIMPS) - EQLIC(-1) * GDPMPS / GDPMPS(-1)

    EViews original (line 910):
    NAEQLIC = (1.6035 + 0.9385 * EQLIC(-1) / (FYCPR(-1) + FISIMPS(-1)))
              * (FYCPR + FISIMPS) - EQLIC(-1) * GDPMPS / GDPMPS(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    ratio = v('EQLIC', 1) / (v('FYCPR', 1) + v('FISIMPS', 1))

    return ((1.6035 + 0.9385 * ratio) * (v('FYCPR') + v('FISIMPS'))
            - v('EQLIC', 1) * v('GDPMPS') / v('GDPMPS', 1))


# --------------------------------------------------------------------------- #
# NALIC
# --------------------------------------------------------------------------- #
def compute_NALIC(data, t):
    """
    NALIC = -27362 + 1.513178 * IBUS * (PIF / 100)

    EViews original (line 912):
    NALIC = -27362 + 1.513178 * IBUS * (PIF / 100)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return -27362 + 1.513178 * v('IBUS') * (v('PIF') / 100.0)


# --------------------------------------------------------------------------- #
# AIC
# --------------------------------------------------------------------------- #
def compute_AIC(data, t):
    """
    AIC = AIC(-1) + (NAAIC - d(M4IC))

    EViews original (line 914):
    AIC = AIC(-1) + (NAAIC - d(M4IC))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_M4IC = v('M4IC') - v('M4IC', 1)

    return v('AIC', 1) + (v('NAAIC') - d_M4IC)


# --------------------------------------------------------------------------- #
# NAAIC
# --------------------------------------------------------------------------- #
def compute_NAAIC(data, t):
    """
    NAAIC = AIC(-1) * (GDPMPS / GDPMPS(-1) - 1)

    EViews original (line 916):
    NAAIC = AIC(-1) * (GDPMPS / GDPMPS(-1) - 1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('AIC', 1) * (v('GDPMPS') / v('GDPMPS', 1) - 1)


# --------------------------------------------------------------------------- #
# NWIC
# --------------------------------------------------------------------------- #
def compute_NWIC(data, t):
    """
    NWIC = AIC - LIC

    EViews original (line 918):
    NWIC = AIC - LIC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('AIC') - v('LIC')


# =========================================================================== #
# get_equations
# =========================================================================== #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        # --- Household Financial Accounts (lines 768-803) ---
        ('NAFHHNSA',  compute_NAFHHNSA,  'identity'),
        ('SDLHH',     compute_SDLHH,     'identity'),
        ('NLHH',      compute_NLHH,      'identity'),
        ('GMF',       compute_GMF,       'identity'),
        ('DEPHHx',    compute_DEPHHx,    'd'),
        ('DEPHH',     compute_DEPHH,     'd'),
        ('NAEQHHx',   compute_NAEQHHx,   'identity'),
        ('NAEQHH',    compute_NAEQHH,    'identity'),
        ('EQHH',      compute_EQHH,      'identity'),
        ('NAPEN',     compute_NAPEN,     'identity'),
        ('NAINSx',    compute_NAINSx,    'identity'),
        ('NAINS',     compute_NAINS,     'identity'),
        ('PIHH',      compute_PIHH,      'identity'),
        ('DBR',       compute_DBR,       'identity'),
        ('OAHHx',     compute_OAHHx,     'dlog'),
        ('OAHH',      compute_OAHH,      'd'),
        ('GFWPE',     compute_GFWPE,     'identity'),

        # --- Household Financial Liabilities (lines 808-825) ---
        ('NAOLPEx',   compute_NAOLPEx,   'identity'),
        ('NAOLPE',    compute_NAOLPE,    'identity'),
        ('DEBTU',     compute_DEBTU,     'identity'),
        ('OLPEx',     compute_OLPEx,     'identity'),
        ('OLPE',      compute_OLPE,      'identity'),
        ('AAHH',      compute_AAHH,      'identity'),
        ('ALHH',      compute_ALHH,      'identity'),
        ('HHRES',     compute_HHRES,     'identity'),
        ('OAHHADJ',   compute_OAHHADJ,   'identity'),

        # --- Aggregates (lines 829-831) ---
        ('NFWPE',     compute_NFWPE,     'identity'),
        ('GPW',       compute_GPW,       'identity'),

        # --- Rest of World (lines 836-889) ---
        ('NAFROWNSA', compute_NAFROWNSA, 'identity'),
        ('SDLROW',    compute_SDLROW,    'identity'),
        ('NLROW',     compute_NLROW,     'identity'),
        ('DAROW',     compute_DAROW,     'd'),
        ('EQAROW',    compute_EQAROW,    'identity'),
        ('NAEQAROW',  compute_NAEQAROW,  'identity'),
        ('BAROW',     compute_BAROW,     'identity'),
        ('NABAROW',   compute_NABAROW,   'identity'),
        ('OTAROW',    compute_OTAROW,    'identity'),
        ('NAOTAROW',  compute_NAOTAROW,  'identity'),
        ('AROW',      compute_AROW,      'identity'),
        ('AAROW',     compute_AAROW,     'identity'),
        ('DLROW',     compute_DLROW,     'identity'),
        ('NADLROW',   compute_NADLROW,   'identity'),
        ('EQLROW',    compute_EQLROW,    'identity'),
        ('NAEQLROW',  compute_NAEQLROW,  'identity'),
        ('BLROW',     compute_BLROW,     'identity'),
        ('NABLROW',   compute_NABLROW,   'identity'),
        ('OTLROW',    compute_OTLROW,    'identity'),
        ('NAOTLROW',  compute_NAOTLROW,  'identity'),
        ('LROW',      compute_LROW,      'identity'),
        ('ALROW',     compute_ALROW,     'identity'),
        ('NIIP',      compute_NIIP,      'd'),

        # --- PNFC Balance Sheet (lines 894-918) ---
        ('BLIC',      compute_BLIC,      'identity'),
        ('STLIC',     compute_STLIC,     'identity'),
        ('FXLIC',     compute_FXLIC,     'identity'),
        ('EQLIC',     compute_EQLIC,     'identity'),
        ('OLIC',      compute_OLIC,      'identity'),
        ('LIC',       compute_LIC,       'identity'),
        ('NABLIC',    compute_NABLIC,    'identity'),
        ('NAFXLIC',   compute_NAFXLIC,   'identity'),
        ('NAEQLIC',   compute_NAEQLIC,   'identity'),
        ('NALIC',     compute_NALIC,     'identity'),
        ('AIC',       compute_AIC,       'identity'),
        ('NAAIC',     compute_NAAIC,     'identity'),
        ('NWIC',      compute_NWIC,      'identity'),
    ]
