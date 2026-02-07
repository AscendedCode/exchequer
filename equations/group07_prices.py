"""
Group 7: Prices and Wages
Lines 228-348 of obr_model.txt

Equations:
    OILBASE    - Oil price base (2009 average of PBRENT/RXD)
    PSAVEI     - Average weekly earnings index (dlog ECM)
    EARN       - Earnings per employee (identity)
    RPW        - Real product wage (identity)
    RCW        - Real consumption wage (identity)
    ULCPS      - Unit labour costs, private sector (identity)
    MSGVAPSEMP - Market sector GVA at factor cost (identity)
    FYEMPMS    - Compensation of employees, market sector (identity)
    ULCMS      - Unit labour costs, market sector (identity)
    ULCPSBASE  - ULC private sector base (2009 average)
    ULCMSBASE  - ULC market sector base (2009 average)
    PMNOGBASE  - Import price non-oil goods base (2009 average)
    PMSBASE    - Import price services base (2009 average)
    TXRATEBASE - Tax rate base (2009 average of BPAPS/GVA)
    PPIYBASE   - PPI output base (2009 average)
    CPIXBASE   - CPI ex rent base (2009 average)
    MCOST      - Manufacturing cost index (identity, simultaneous)
    SCOST      - Services cost index (identity, simultaneous)
    CCOST      - Construction cost index (identity, simultaneous)
    UTCOST     - Utilities & transport cost index (identity, simultaneous)
    RPCOST     - Retail price cost index (identity)
    ICOST      - Investment cost index (identity)
    XGCOST     - Export goods cost index (identity)
    XSCOST     - Export services cost index (identity)
    MKGW       - Markup, goods wholesale (identity)
    MKR        - Markup, retail (dlog)
    CPIX       - CPI ex rent (identity)
    PRENT      - Private rents (identity)
    CPIH       - CPI including OOH (identity)
    CPIRENT    - CPI rent component (identity)
    PRMIP      - RPI mortgage interest payments (identity, additive adj)
    PR         - RPI index (identity)
    RPI        - RPI inflation (identity)
    PXNOG      - Export price non-oil goods (dlog ECM)
    PXS        - Export price services (ratio)
    PMNOG      - Import price non-oil goods (dlog ECM)
    PMS        - Import price services (ratio)
    PINV       - Price of inventories (identity)
    PCE        - Consumer expenditure deflator (ratio, 4-quarter)
    PIF        - Fixed investment deflator (identity)
    PCDUR      - Durable goods deflator (ratio)
    RHF        - Effective household interest rate (identity)
    HD         - Housing depreciation (ratio)
    PMSGVA     - Market sector GVA deflator (identity)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log
from .. import config


# --------------------------------------------------------------------------- #
# OILBASE
# --------------------------------------------------------------------------- #
def compute_OILBASE(data, t):
    """
    OILBASE = ((@elem(PBRENT, "2009Q1") / @elem(RXD, "2009Q1"))
             + (@elem(PBRENT, "2009Q2") / @elem(RXD, "2009Q2"))
             + (@elem(PBRENT, "2009Q3") / @elem(RXD, "2009Q3"))
             + (@elem(PBRENT, "2009Q4") / @elem(RXD, "2009Q4"))) / 4
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return ((elem(data, 'PBRENT', '2009Q1') / elem(data, 'RXD', '2009Q1'))
            + (elem(data, 'PBRENT', '2009Q2') / elem(data, 'RXD', '2009Q2'))
            + (elem(data, 'PBRENT', '2009Q3') / elem(data, 'RXD', '2009Q3'))
            + (elem(data, 'PBRENT', '2009Q4') / elem(data, 'RXD', '2009Q4'))) / 4


# --------------------------------------------------------------------------- #
# PSAVEI
# --------------------------------------------------------------------------- #
def compute_PSAVEI(data, t):
    """
    dlog(PSAVEI) = -0.0282
        + 0.575 * dlog(PMSGVA)
        + 0.250 * dlog(PMSGVA(-1))
        + 0.105 * dlog(PMSGVA(-2))
        + (1 - 0.575 - 0.250 - 0.105) * dlog(PMSGVA(-3))
        - 0.0096 * (LFSUR - LFSUR(-1))
        + 0.264 * (dlog(MSGVA) - dlog(EMS))
        + 0.282 * (dlog(CPI) - dlog(PMSGVA))
        - 0.04328 * (log(PSAVEI(-1))
            - log(MSGVA(-1) / EMS(-1))
            - log(PMSGVA(-1))
            + log(1 + (EMPSC(-1) / WFP(-1)))
            + 0.0137 * LFSUR(-1))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    # dlog(PMSGVA) at various lags
    dlog_PMSGVA_0 = np.log(v('PMSGVA') / v('PMSGVA', 1))
    dlog_PMSGVA_1 = np.log(v('PMSGVA', 1) / v('PMSGVA', 2))
    dlog_PMSGVA_2 = np.log(v('PMSGVA', 2) / v('PMSGVA', 3))
    dlog_PMSGVA_3 = np.log(v('PMSGVA', 3) / v('PMSGVA', 4))

    # d(LFSUR)
    d_LFSUR = v('LFSUR') - v('LFSUR', 1)

    # dlog(MSGVA) - dlog(EMS)
    dlog_MSGVA = np.log(v('MSGVA') / v('MSGVA', 1))
    dlog_EMS = np.log(v('EMS') / v('EMS', 1))

    # dlog(CPI) - dlog(PMSGVA)
    dlog_CPI = np.log(v('CPI') / v('CPI', 1))

    # Error correction term
    ecm = (safe_log(v('PSAVEI', 1))
           - safe_log(v('MSGVA', 1) / v('EMS', 1))
           - safe_log(v('PMSGVA', 1))
           + safe_log(1 + (v('EMPSC', 1) / v('WFP', 1)))
           + 0.0137 * v('LFSUR', 1))

    rhs = (-0.0282
           + 0.575 * dlog_PMSGVA_0
           + 0.250 * dlog_PMSGVA_1
           + 0.105 * dlog_PMSGVA_2
           + (1 - 0.575 - 0.250 - 0.105) * dlog_PMSGVA_3
           - 0.0096 * d_LFSUR
           + 0.264 * (dlog_MSGVA - dlog_EMS)
           + 0.282 * (dlog_CPI - dlog_PMSGVA_0)
           - 0.04328 * ecm)

    return v('PSAVEI', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# EARN
# --------------------------------------------------------------------------- #
def compute_EARN(data, t):
    """EARN = WFP / (ETLFS - ESLFS)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('WFP') / (v('ETLFS') - v('ESLFS'))


# --------------------------------------------------------------------------- #
# RPW
# --------------------------------------------------------------------------- #
def compute_RPW(data, t):
    """RPW = (FYEMP / PGVA) / (ETLFS - ESLFS)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('FYEMP') / v('PGVA')) / (v('ETLFS') - v('ESLFS'))


# --------------------------------------------------------------------------- #
# RCW
# --------------------------------------------------------------------------- #
def compute_RCW(data, t):
    """RCW = (FYEMP / PCE) / (ETLFS - ESLFS)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('FYEMP') / v('PCE')) / (v('ETLFS') - v('ESLFS'))


# --------------------------------------------------------------------------- #
# ULCPS
# --------------------------------------------------------------------------- #
def compute_ULCPS(data, t):
    """ULCPS = 0.17910 * (PSAVEI * (52 / 4) * (1 + (EMPSC + NIS) / WFP) * EMS / GVA)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.17910 * (v('PSAVEI') * (52 / 4)
                      * (1 + (v('EMPSC') + v('NIS')) / v('WFP'))
                      * v('EMS') / v('GVA'))


# --------------------------------------------------------------------------- #
# MSGVAPSEMP
# --------------------------------------------------------------------------- #
def compute_MSGVAPSEMP(data, t):
    """MSGVAPSEMP = MSGVAPS - MI"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('MSGVAPS') - v('MI')


# --------------------------------------------------------------------------- #
# FYEMPMS
# --------------------------------------------------------------------------- #
def compute_FYEMPMS(data, t):
    """FYEMPMS = FYEMP - CGWS - LAWS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('FYEMP') - v('CGWS') - v('LAWS')


# --------------------------------------------------------------------------- #
# ULCMS
# --------------------------------------------------------------------------- #
def compute_ULCMS(data, t):
    """ULCMS = 100 * 1.6715 * FYEMPMS * (1 + (MI / MSGVAPSEMP)) / MSGVA"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 100 * 1.6715 * v('FYEMPMS') * (1 + (v('MI') / v('MSGVAPSEMP'))) / v('MSGVA')


# --------------------------------------------------------------------------- #
# ULCPSBASE
# --------------------------------------------------------------------------- #
def compute_ULCPSBASE(data, t):
    """
    ULCPSBASE = (@elem(ULCPS, "2009Q1") + @elem(ULCPS, "2009Q2")
               + @elem(ULCPS, "2009Q3") + @elem(ULCPS, "2009Q4")) / 4
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (elem(data, 'ULCPS', '2009Q1') + elem(data, 'ULCPS', '2009Q2')
            + elem(data, 'ULCPS', '2009Q3') + elem(data, 'ULCPS', '2009Q4')) / 4


# --------------------------------------------------------------------------- #
# ULCMSBASE
# --------------------------------------------------------------------------- #
def compute_ULCMSBASE(data, t):
    """
    ULCMSBASE = (@elem(ULCMS, "2009Q1") + @elem(ULCMS, "2009Q2")
               + @elem(ULCMS, "2009Q3") + @elem(ULCMS, "2009Q4")) / 4
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (elem(data, 'ULCMS', '2009Q1') + elem(data, 'ULCMS', '2009Q2')
            + elem(data, 'ULCMS', '2009Q3') + elem(data, 'ULCMS', '2009Q4')) / 4


# --------------------------------------------------------------------------- #
# PMNOGBASE
# --------------------------------------------------------------------------- #
def compute_PMNOGBASE(data, t):
    """
    PMNOGBASE = (@elem(PMNOG, "2009Q1") + @elem(PMNOG, "2009Q2")
               + @elem(PMNOG, "2009Q3") + @elem(PMNOG, "2009Q4")) / 4
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (elem(data, 'PMNOG', '2009Q1') + elem(data, 'PMNOG', '2009Q2')
            + elem(data, 'PMNOG', '2009Q3') + elem(data, 'PMNOG', '2009Q4')) / 4


# --------------------------------------------------------------------------- #
# PMSBASE
# --------------------------------------------------------------------------- #
def compute_PMSBASE(data, t):
    """
    PMSBASE = (@elem(PMS, "2009Q1") + @elem(PMS, "2009Q2")
             + @elem(PMS, "2009Q3") + @elem(PMS, "2009Q4")) / 4
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (elem(data, 'PMS', '2009Q1') + elem(data, 'PMS', '2009Q2')
            + elem(data, 'PMS', '2009Q3') + elem(data, 'PMS', '2009Q4')) / 4


# --------------------------------------------------------------------------- #
# TXRATEBASE
# --------------------------------------------------------------------------- #
def compute_TXRATEBASE(data, t):
    """
    TXRATEBASE = ((@elem(BPAPS, "2009Q1") / @elem(GVA, "2009Q1"))
               + (@elem(BPAPS, "2009Q2") / @elem(GVA, "2009Q2"))
               + (@elem(BPAPS, "2009Q3") / @elem(GVA, "2009Q3"))
               + (@elem(BPAPS, "2009Q4") / @elem(GVA, "2009Q4"))) / 4
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return ((elem(data, 'BPAPS', '2009Q1') / elem(data, 'GVA', '2009Q1'))
            + (elem(data, 'BPAPS', '2009Q2') / elem(data, 'GVA', '2009Q2'))
            + (elem(data, 'BPAPS', '2009Q3') / elem(data, 'GVA', '2009Q3'))
            + (elem(data, 'BPAPS', '2009Q4') / elem(data, 'GVA', '2009Q4'))) / 4


# --------------------------------------------------------------------------- #
# PPIYBASE
# --------------------------------------------------------------------------- #
def compute_PPIYBASE(data, t):
    """
    PPIYBASE = (@elem(PPIY, "2009Q1") + @elem(PPIY, "2009Q2")
              + @elem(PPIY, "2009Q3") + @elem(PPIY, "2009Q4")) / 4
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (elem(data, 'PPIY', '2009Q1') + elem(data, 'PPIY', '2009Q2')
            + elem(data, 'PPIY', '2009Q3') + elem(data, 'PPIY', '2009Q4')) / 4


# --------------------------------------------------------------------------- #
# CPIXBASE
# --------------------------------------------------------------------------- #
def compute_CPIXBASE(data, t):
    """
    CPIXBASE = (@elem(CPIX, "2009Q1") + @elem(CPIX, "2009Q2")
              + @elem(CPIX, "2009Q3") + @elem(CPIX, "2009Q4")) / 4
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (elem(data, 'CPIX', '2009Q1') + elem(data, 'CPIX', '2009Q2')
            + elem(data, 'CPIX', '2009Q3') + elem(data, 'CPIX', '2009Q4')) / 4


# --------------------------------------------------------------------------- #
# MCOST
# --------------------------------------------------------------------------- #
def compute_MCOST(data, t):
    """
    MCOST = 36.83 * (ULCMS / ULCMSBASE)
          + 24.64 * (PMNOG / PMNOGBASE)
          + 4.04 * (PMS / PMSBASE)
          + 4.85 * ((PBRENT / RXD) / OILBASE)
          + 1.01 * ((BPAPS / GVA) / TXRATEBASE)
          + 24.72 * (SCOST / 100)
          + 0.47 * (CCOST / 100)
          + 3.43 * (UTCOST / 100)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (36.83 * (v('ULCMS') / v('ULCMSBASE'))
            + 24.64 * (v('PMNOG') / v('PMNOGBASE'))
            + 4.04 * (v('PMS') / v('PMSBASE'))
            + 4.85 * ((v('PBRENT') / v('RXD')) / v('OILBASE'))
            + 1.01 * ((v('BPAPS') / v('GVA')) / v('TXRATEBASE'))
            + 24.72 * (v('SCOST') / 100)
            + 0.47 * (v('CCOST') / 100)
            + 3.43 * (v('UTCOST') / 100))


# --------------------------------------------------------------------------- #
# SCOST  (simultaneous with CCOST, UTCOST)
# --------------------------------------------------------------------------- #
def compute_SCOST(data, t):
    """
    SCOST = 70.54 * (ULCMS / ULCMSBASE)
          + 6.93 * (PMNOG / PMNOGBASE)
          + 6.41 * (PMS / PMSBASE)
          + 0.09 * ((PBRENT / RXD) / OILBASE)
          + 3.52 * ((BPAPS / GVA) / TXRATEBASE)
          + 9.78 * (PPIY / PPIYBASE)
          + 1.64 * (CCOST / 100)
          + 1.09 * (UTCOST / 100)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (70.54 * (v('ULCMS') / v('ULCMSBASE'))
            + 6.93 * (v('PMNOG') / v('PMNOGBASE'))
            + 6.41 * (v('PMS') / v('PMSBASE'))
            + 0.09 * ((v('PBRENT') / v('RXD')) / v('OILBASE'))
            + 3.52 * ((v('BPAPS') / v('GVA')) / v('TXRATEBASE'))
            + 9.78 * (v('PPIY') / v('PPIYBASE'))
            + 1.64 * (v('CCOST') / 100)
            + 1.09 * (v('UTCOST') / 100))


# --------------------------------------------------------------------------- #
# CCOST  (simultaneous with SCOST, UTCOST)
# --------------------------------------------------------------------------- #
def compute_CCOST(data, t):
    """
    CCOST = 40.25 * (ULCMS / ULCMSBASE)
          + 2.80 * (PMNOG / PMNOGBASE)
          + 0.90 * (PMS / PMSBASE)
          + 0.03 * ((PBRENT / RXD) / OILBASE)
          + 0.51 * ((BPAPS / GVA) / TXRATEBASE)
          + 27.06 * (PPIY / PPIYBASE)
          + 28.13 * (SCOST / 100)
          + 0.34 * (UTCOST / 100)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (40.25 * (v('ULCMS') / v('ULCMSBASE'))
            + 2.80 * (v('PMNOG') / v('PMNOGBASE'))
            + 0.90 * (v('PMS') / v('PMSBASE'))
            + 0.03 * ((v('PBRENT') / v('RXD')) / v('OILBASE'))
            + 0.51 * ((v('BPAPS') / v('GVA')) / v('TXRATEBASE'))
            + 27.06 * (v('PPIY') / v('PPIYBASE'))
            + 28.13 * (v('SCOST') / 100)
            + 0.34 * (v('UTCOST') / 100))


# --------------------------------------------------------------------------- #
# UTCOST  (simultaneous with SCOST, CCOST)
# --------------------------------------------------------------------------- #
def compute_UTCOST(data, t):
    """
    UTCOST = 14.85 * (ULCMS / ULCMSBASE)
           + 3.04 * (PMNOG / PMNOGBASE)
           + 0.51 * (PMS / PMSBASE)
           + 51.52 * ((PBRENT / RXD) / OILBASE)
           + 2.90 * ((BPAPS / GVA) / TXRATEBASE)
           + 8.24 * (PPIY / PPIYBASE)
           + 16.00 * (SCOST / 100)
           + 2.95 * (CCOST / 100)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (14.85 * (v('ULCMS') / v('ULCMSBASE'))
            + 3.04 * (v('PMNOG') / v('PMNOGBASE'))
            + 0.51 * (v('PMS') / v('PMSBASE'))
            + 51.52 * ((v('PBRENT') / v('RXD')) / v('OILBASE'))
            + 2.90 * ((v('BPAPS') / v('GVA')) / v('TXRATEBASE'))
            + 8.24 * (v('PPIY') / v('PPIYBASE'))
            + 16.00 * (v('SCOST') / 100)
            + 2.95 * (v('CCOST') / 100))


# --------------------------------------------------------------------------- #
# RPCOST
# --------------------------------------------------------------------------- #
def compute_RPCOST(data, t):
    """
    RPCOST = 13.18 * (PMNOG / PMNOGBASE)
           + 4.07 * (PMS / PMSBASE)
           + 11.56 * ((BPAPS / GVA) / TXRATEBASE)
           + 7.07 * (PPIY / PPIYBASE)
           + 59.96 * (SCOST / 100)
           + 0.92 * (CCOST / 100)
           + 3.24 * (UTCOST / 100)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (13.18 * (v('PMNOG') / v('PMNOGBASE'))
            + 4.07 * (v('PMS') / v('PMSBASE'))
            + 11.56 * ((v('BPAPS') / v('GVA')) / v('TXRATEBASE'))
            + 7.07 * (v('PPIY') / v('PPIYBASE'))
            + 59.96 * (v('SCOST') / 100)
            + 0.92 * (v('CCOST') / 100)
            + 3.24 * (v('UTCOST') / 100))


# --------------------------------------------------------------------------- #
# ICOST
# --------------------------------------------------------------------------- #
def compute_ICOST(data, t):
    """
    ICOST = 18.40 * (PMNOG / PMNOGBASE)
          + 0.41 * (PMS / PMSBASE)
          + 0.19 * ((PBRENT / RXD) / OILBASE)
          + 5.63 * ((BPAPS / MSGVA) / TXRATEBASE)
          + 8.18 * (PPIY / PPIYBASE)
          + 20.76 * (SCOST / 100)
          + 46.42 * (CCOST / 100)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (18.40 * (v('PMNOG') / v('PMNOGBASE'))
            + 0.41 * (v('PMS') / v('PMSBASE'))
            + 0.19 * ((v('PBRENT') / v('RXD')) / v('OILBASE'))
            + 5.63 * ((v('BPAPS') / v('MSGVA')) / v('TXRATEBASE'))
            + 8.18 * (v('PPIY') / v('PPIYBASE'))
            + 20.76 * (v('SCOST') / 100)
            + 46.42 * (v('CCOST') / 100))


# --------------------------------------------------------------------------- #
# XGCOST
# --------------------------------------------------------------------------- #
def compute_XGCOST(data, t):
    """
    XGCOST = 15.77 * (PMNOG / PMNOGBASE)
           + 2.92 * ((BPAPS / MSGVA) / TXRATEBASE)
           + 68.46 * (PPIY / PPIYBASE)
           + 12.80 * (SCOST / 100)
           + 0.05 * (UTCOST / 100)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (15.77 * (v('PMNOG') / v('PMNOGBASE'))
            + 2.92 * ((v('BPAPS') / v('MSGVA')) / v('TXRATEBASE'))
            + 68.46 * (v('PPIY') / v('PPIYBASE'))
            + 12.80 * (v('SCOST') / 100)
            + 0.05 * (v('UTCOST') / 100))


# --------------------------------------------------------------------------- #
# XSCOST
# --------------------------------------------------------------------------- #
def compute_XSCOST(data, t):
    """
    XSCOST = 7.22 * (PMS / PMSBASE)
           + 5.99 * ((BPAPS / MSGVA) / TXRATEBASE)
           + 9.29 * (PPIY / PPIYBASE)
           + 75.39 * (SCOST / 100)
           + 1.90 * (CCOST / 100)
           + 0.21 * (UTCOST / 100)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (7.22 * (v('PMS') / v('PMSBASE'))
            + 5.99 * ((v('BPAPS') / v('MSGVA')) / v('TXRATEBASE'))
            + 9.29 * (v('PPIY') / v('PPIYBASE'))
            + 75.39 * (v('SCOST') / 100)
            + 1.90 * (v('CCOST') / 100)
            + 0.21 * (v('UTCOST') / 100))


# --------------------------------------------------------------------------- #
# MKGW
# --------------------------------------------------------------------------- #
def compute_MKGW(data, t):
    """MKGW = 100 * (PPIY / (MCOST / 100)) / (PPIYBASE)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 100 * (v('PPIY') / (v('MCOST') / 100)) / v('PPIYBASE')


# --------------------------------------------------------------------------- #
# MKR
# --------------------------------------------------------------------------- #
def compute_MKR(data, t):
    """
    dlog(MKR) = (dlog(CPI) - W1 * dlog(CPIRENT)
                - (1 - W1) * dlog(RPCOST)) / (1 - W1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    W1 = config.W1

    dlog_CPI = np.log(v('CPI') / v('CPI', 1))
    dlog_CPIRENT = np.log(v('CPIRENT') / v('CPIRENT', 1))
    dlog_RPCOST = np.log(v('RPCOST') / v('RPCOST', 1))

    rhs = (dlog_CPI - W1 * dlog_CPIRENT - (1 - W1) * dlog_RPCOST) / (1 - W1)

    return v('MKR', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# CPIX
# --------------------------------------------------------------------------- #
def compute_CPIX(data, t):
    """CPIX = (RPCOST / 100) * (MKR / 100) * CPIXBASE"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('RPCOST') / 100) * (v('MKR') / 100) * v('CPIXBASE')


# --------------------------------------------------------------------------- #
# PRENT
# --------------------------------------------------------------------------- #
def compute_PRENT(data, t):
    """
    PRENT = PRENT(-1) * (0.62 * ((WFP / (ETLFS - ESLFS))
            / (WFP(-1) / (ETLFS(-1) - ESLFS(-1))))
            + 0.15 * (HRRPW / HRRPW(-1))
            + 0.23 * (PRP / PRP(-1)))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    # Wage growth: current earnings / lagged earnings
    wage_t = v('WFP') / (v('ETLFS') - v('ESLFS'))
    wage_t1 = v('WFP', 1) / (v('ETLFS', 1) - v('ESLFS', 1))
    wage_growth = wage_t / wage_t1

    # HRRPW growth
    hrrpw_growth = v('HRRPW') / v('HRRPW', 1)

    # PRP growth
    prp_growth = v('PRP') / v('PRP', 1)

    return v('PRENT', 1) * (0.62 * wage_growth
                             + 0.15 * hrrpw_growth
                             + 0.23 * prp_growth)


# --------------------------------------------------------------------------- #
# CPIH
# --------------------------------------------------------------------------- #
def compute_CPIH(data, t):
    """
    CPIH = CPIH(-1) * (CPI^(1 - W5) * OOH^W5)
                     / (CPI(-1)^(1 - W5) * OOH(-1)^W5)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    W5 = config.W5

    return (v('CPIH', 1)
            * (v('CPI') ** (1 - W5) * v('OOH') ** W5)
            / (v('CPI', 1) ** (1 - W5) * v('OOH', 1) ** W5))


# --------------------------------------------------------------------------- #
# CPIRENT
# --------------------------------------------------------------------------- #
def compute_CPIRENT(data, t):
    """
    CPIRENT = CPIRENT(-1) * (0.62 * ((WFP / (ETLFS - ESLFS))
              / (WFP(-1) / (ETLFS(-1) - ESLFS(-1))))
              + 0.15 * (HRRPW / HRRPW(-1))
              + 0.23 * (PRP / PRP(-1)))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    # Wage growth
    wage_t = v('WFP') / (v('ETLFS') - v('ESLFS'))
    wage_t1 = v('WFP', 1) / (v('ETLFS', 1) - v('ESLFS', 1))
    wage_growth = wage_t / wage_t1

    # HRRPW growth
    hrrpw_growth = v('HRRPW') / v('HRRPW', 1)

    # PRP growth
    prp_growth = v('PRP') / v('PRP', 1)

    return v('CPIRENT', 1) * (0.62 * wage_growth
                               + 0.15 * hrrpw_growth
                               + 0.23 * prp_growth)


# --------------------------------------------------------------------------- #
# PRMIP
# --------------------------------------------------------------------------- #
def compute_PRMIP(data, t):
    """
    PRMIP = PRMIP(-1) * (RMORT / RMORT(-1)) * (LHP / LHP(-1)) / (HH / HH(-1))
    @ADD(V) PRMIP  PRMIP_A
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    result = (v('PRMIP', 1)
              * (v('RMORT') / v('RMORT', 1))
              * (v('LHP') / v('LHP', 1))
              / (v('HH') / v('HH', 1)))

    # Additive adjustment
    result += v('PRMIP_A')

    return result


# --------------------------------------------------------------------------- #
# PR
# --------------------------------------------------------------------------- #
def compute_PR(data, t):
    """PR = I7 * ((1 - W4) * PRXMIP / I9 + W4 * PRMIP / I4)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    I4 = config.I4
    I7 = config.I7
    I9 = config.I9
    W4 = config.W4

    return I7 * ((1 - W4) * v('PRXMIP') / I9 + W4 * v('PRMIP') / I4)


# --------------------------------------------------------------------------- #
# RPI
# --------------------------------------------------------------------------- #
def compute_RPI(data, t):
    """RPI = PR / PR(-4) * 100 - 100"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PR') / v('PR', 4) * 100 - 100


# --------------------------------------------------------------------------- #
# PXNOG
# --------------------------------------------------------------------------- #
def compute_PXNOG(data, t):
    """
    dlog(PXNOG) = 0.635957 * dlog(PPIY(-1))
        + 0.102727 * (dlog(WPG) - dlog(RXD))
        - 0.131253 * dlog(RX)
        - 0.000508 * @TREND(1979Q4)
        + 0.100860 * @recode(@date = @dateval("1997:01"), 1, 0)
        - 0.063293 * @recode(@date = @dateval("1998:01"), 1, 0)
        + 0.034519 * @recode(@date = @dateval("1993:01"), 1, 0)
        - 0.161370 * (log(PXNOG(-1))
            + 0.330293 * log(RX(-1))
            - 0.921258 * log(PPIY(-1))
            - (1 - 0.921258) * log(WPG(-1) / RXD(-1)))
        + 0.297153
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    # dlog(PPIY(-1))
    dlog_PPIY_1 = np.log(v('PPIY', 1) / v('PPIY', 2))

    # dlog(WPG) - dlog(RXD)
    dlog_WPG = np.log(v('WPG') / v('WPG', 1))
    dlog_RXD = np.log(v('RXD') / v('RXD', 1))

    # dlog(RX)
    dlog_RX = np.log(v('RX') / v('RX', 1))

    # Trend
    tr = trend(t, '1979Q4')

    # Dummies
    d_1997Q1 = recode_eq(t, "1997:01")
    d_1998Q1 = recode_eq(t, "1998:01")
    d_1993Q1 = recode_eq(t, "1993:01")

    # Error correction term
    ecm = (safe_log(v('PXNOG', 1))
           + 0.330293 * safe_log(v('RX', 1))
           - 0.921258 * safe_log(v('PPIY', 1))
           - (1 - 0.921258) * safe_log(v('WPG', 1) / v('RXD', 1)))

    rhs = (0.635957 * dlog_PPIY_1
           + 0.102727 * (dlog_WPG - dlog_RXD)
           - 0.131253 * dlog_RX
           - 0.000508 * tr
           + 0.100860 * d_1997Q1
           - 0.063293 * d_1998Q1
           + 0.034519 * d_1993Q1
           - 0.161370 * ecm
           + 0.297153)

    return v('PXNOG', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# PXS
# --------------------------------------------------------------------------- #
def compute_PXS(data, t):
    """PXS / PXS(-1) = PXNOG / PXNOG(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PXS', 1) * (v('PXNOG') / v('PXNOG', 1))


# --------------------------------------------------------------------------- #
# PMNOG
# --------------------------------------------------------------------------- #
def compute_PMNOG(data, t):
    """
    dlog(PMNOG) = 0.606452 * dlog(PPIY)
        + 0.230808 * (dlog(WPG) - dlog(RXD))
        - 0.106493 * dlog(RX)
        + 0.066665 * @recode(@date = @dateval("1997:01"), 1, 0)
        - 0.038986 * @recode(@date = @dateval("1998:01"), 1, 0)
        - 0.000538 * @TREND(1979Q4)
        - 0.160709 * (log(PMNOG(-1))
            + 0.139917 * log(RX(-1))
            - 0.552396 * log(PPIY(-1))
            - (1 - 0.552396) * log(WPG(-1) / RXD(-1)))
        + 0.183135
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    # dlog(PPIY) - current period (no lag, unlike PXNOG)
    dlog_PPIY = np.log(v('PPIY') / v('PPIY', 1))

    # dlog(WPG) - dlog(RXD)
    dlog_WPG = np.log(v('WPG') / v('WPG', 1))
    dlog_RXD = np.log(v('RXD') / v('RXD', 1))

    # dlog(RX)
    dlog_RX = np.log(v('RX') / v('RX', 1))

    # Trend
    tr = trend(t, '1979Q4')

    # Dummies
    d_1997Q1 = recode_eq(t, "1997:01")
    d_1998Q1 = recode_eq(t, "1998:01")

    # Error correction term
    ecm = (safe_log(v('PMNOG', 1))
           + 0.139917 * safe_log(v('RX', 1))
           - 0.552396 * safe_log(v('PPIY', 1))
           - (1 - 0.552396) * safe_log(v('WPG', 1) / v('RXD', 1)))

    rhs = (0.606452 * dlog_PPIY
           + 0.230808 * (dlog_WPG - dlog_RXD)
           - 0.106493 * dlog_RX
           + 0.066665 * d_1997Q1
           - 0.038986 * d_1998Q1
           - 0.000538 * tr
           - 0.160709 * ecm
           + 0.183135)

    return v('PMNOG', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# PMS
# --------------------------------------------------------------------------- #
def compute_PMS(data, t):
    """PMS / PMS(-1) = PMNOG / PMNOG(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PMS', 1) * (v('PMNOG') / v('PMNOG', 1))


# --------------------------------------------------------------------------- #
# PINV
# --------------------------------------------------------------------------- #
def compute_PINV(data, t):
    """PINV = 100 * BV / INV"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 100 * v('BV') / v('INV')


# --------------------------------------------------------------------------- #
# PCE
# --------------------------------------------------------------------------- #
def compute_PCE(data, t):
    """PCE / PCE(-4) = CPI / CPI(-4)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PCE', 4) * (v('CPI') / v('CPI', 4))


# --------------------------------------------------------------------------- #
# PIF
# --------------------------------------------------------------------------- #
def compute_PIF(data, t):
    """
    PIF = (GDPMPS - CGGPS - CONSPS - DINVPS - VALPS - XPS + MPS - SDEPS)
          * 100 / IF
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    numerator = (v('GDPMPS') - v('CGGPS') - v('CONSPS') - v('DINVPS')
                 - v('VALPS') - v('XPS') + v('MPS') - v('SDEPS'))

    return numerator * 100 / v('IF')


# --------------------------------------------------------------------------- #
# PCDUR
# --------------------------------------------------------------------------- #
def compute_PCDUR(data, t):
    """PCDUR / PCDUR(-1) = PMNOG / PMNOG(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PCDUR', 1) * (v('PMNOG') / v('PMNOG', 1))


# --------------------------------------------------------------------------- #
# RHF
# --------------------------------------------------------------------------- #
def compute_RHF(data, t):
    """
    RHF = RMORT - (1 - 0.25 * TPBRZ) * (RMORT - RDEP)
                 * (1 - 0.001 * LHP / GPW)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('RMORT')
            - (1 - 0.25 * v('TPBRZ'))
            * (v('RMORT') - v('RDEP'))
            * (1 - 0.001 * v('LHP') / v('GPW')))


# --------------------------------------------------------------------------- #
# HD
# --------------------------------------------------------------------------- #
def compute_HD(data, t):
    """HD / HD(-1) = APH / APH(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('HD', 1) * (v('APH') / v('APH', 1))


# --------------------------------------------------------------------------- #
# PMSGVA
# --------------------------------------------------------------------------- #
def compute_PMSGVA(data, t):
    """PMSGVA = 100 * (MSGVAPS / MSGVA)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 100 * (v('MSGVAPS') / v('MSGVA'))


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        # Oil base index (2009 average)
        ('OILBASE',    compute_OILBASE,    'identity'),

        # Wages equation (ECM)
        ('PSAVEI',     compute_PSAVEI,     'dlog'),

        # Earnings and wage identities
        ('EARN',       compute_EARN,       'identity'),
        ('RPW',        compute_RPW,        'identity'),
        ('RCW',        compute_RCW,        'identity'),

        # Unit labour costs
        ('ULCPS',      compute_ULCPS,      'identity'),
        ('MSGVAPSEMP', compute_MSGVAPSEMP, 'identity'),
        ('FYEMPMS',    compute_FYEMPMS,    'identity'),
        ('ULCMS',      compute_ULCMS,      'identity'),

        # Base period indices (2009 averages)
        ('ULCPSBASE',  compute_ULCPSBASE,  'identity'),
        ('ULCMSBASE',  compute_ULCMSBASE,  'identity'),
        ('PMNOGBASE',  compute_PMNOGBASE,  'identity'),
        ('PMSBASE',    compute_PMSBASE,    'identity'),
        ('TXRATEBASE', compute_TXRATEBASE, 'identity'),
        ('PPIYBASE',   compute_PPIYBASE,   'identity'),
        ('CPIXBASE',   compute_CPIXBASE,   'identity'),

        # Cost indices (simultaneous block: SCOST, CCOST, UTCOST)
        ('SCOST',      compute_SCOST,      'identity'),
        ('CCOST',      compute_CCOST,      'identity'),
        ('UTCOST',     compute_UTCOST,     'identity'),
        ('MCOST',      compute_MCOST,      'identity'),

        # Derived cost indices
        ('RPCOST',     compute_RPCOST,     'identity'),
        ('ICOST',      compute_ICOST,      'identity'),
        ('XGCOST',     compute_XGCOST,     'identity'),
        ('XSCOST',     compute_XSCOST,     'identity'),

        # Markups
        ('MKGW',       compute_MKGW,       'identity'),
        ('MKR',        compute_MKR,        'dlog'),

        # Consumer prices
        ('CPIX',       compute_CPIX,       'identity'),
        ('PRENT',      compute_PRENT,      'identity'),
        ('CPIH',       compute_CPIH,       'identity'),
        ('CPIRENT',    compute_CPIRENT,    'identity'),

        # RPI block
        ('PRMIP',      compute_PRMIP,      'identity'),
        ('PR',         compute_PR,         'identity'),
        ('RPI',        compute_RPI,        'identity'),

        # Trade prices (ECM equations)
        ('PXNOG',      compute_PXNOG,      'dlog'),
        ('PXS',        compute_PXS,        'ratio'),
        ('PMNOG',      compute_PMNOG,      'dlog'),
        ('PMS',        compute_PMS,        'ratio'),

        # Other deflators
        ('PINV',       compute_PINV,       'identity'),
        ('PCE',        compute_PCE,        'ratio'),
        ('PIF',        compute_PIF,        'identity'),
        ('PCDUR',      compute_PCDUR,      'ratio'),

        # Household finance and housing
        ('RHF',        compute_RHF,        'identity'),
        ('HD',         compute_HD,         'ratio'),

        # Market sector deflator
        ('PMSGVA',     compute_PMSGVA,     'identity'),
    ]
