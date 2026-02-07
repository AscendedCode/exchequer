"""
Group 3: Investment
Lines 38-134 of obr_model.txt

Equations (42 total):
    DB, DP, DV         - Discount factors for tax depreciation allowances
    WB, WP, WV         - Weights for asset types (constants)
    TAFB, TAFP, TAFV   - Tax-adjusted factors by asset type
    TAF                 - Weighted tax-adjustment factor
    WG                  - Growth rate weight (constant)
    CDEBT, CEQUITY     - Cost of debt / cost of equity
    RWACC              - Real weighted average cost of capital
    RDELTA             - Depreciation rate (constant)
    COCU, COC          - Cost of capital (unadjusted / adjusted)
    KSTAR              - Desired capital stock
    KMSXH              - Market-sector capital stock (excl. housing)
    KGAP               - Capital stock gap
    TQ                 - Tobin's Q proxy
    PKMSXHB            - Price of capital stock
    IBUS               - Business investment
    IBUSX              - Business investment (excl. one-off)
    GGIPS, GGI, GGIX   - General government investment
    GGIDEF             - GGI deflator (ratio equation)
    HIMPROV            - Home improvements (dlog equation)
    PCIH               - Public corporations investment in housing (ratio)
    VALPS, VALHH       - Valuables
    IFPS               - Total investment in current prices
    PIPRL              - Private-sector rental investment deflator
    IHPS               - Housing investment in current prices
    IHHPS              - Household investment in current prices (ratio)
    PIBUS              - Business investment deflator
    ICCPS              - Corporate-sector capital spending
    IPCPS              - Public corporations investment current prices
    IFCPS              - Financial-corporations capital spending
    NETAD              - Net additions to housing stock
    HSALL              - Total housing stock
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# --------------------------------------------------------------------------- #
# DB  (discount factor - buildings)
# --------------------------------------------------------------------------- #
def compute_DB(data, t):
    """
    DB = @recode(@date <= @dateval("2011:02"), 1, 0)
         * 1 / (1 + DISCO)
         * (IIB + (SIB / DISCO)
            * (1 - (1 + DISCO)^((-1) * (1 - IIB)
              / (SIB + 0.1 * @recode(@date >= @dateval("2011:03"), 1, 0)))))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    pre_2011q2 = recode_leq(t, "2011:02")
    post_2011q3 = recode_geq(t, "2011:03")

    DISCO = v('DISCO')
    IIB = v('IIB')
    SIB = v('SIB')

    # Guard the denominator with the dummy to avoid division when SIB might
    # combine with the 0.1 adjustment
    sib_adj = SIB + 0.1 * post_2011q3
    exponent = (-1) * (1 - IIB) / sib_adj

    result = (pre_2011q2
              * (1 / (1 + DISCO))
              * (IIB + (SIB / DISCO) * (1 - (1 + DISCO) ** exponent)))

    return result


# --------------------------------------------------------------------------- #
# DP  (discount factor - plant & machinery)
# --------------------------------------------------------------------------- #
def compute_DP(data, t):
    """DP = 1 / (1 + DISCO) * ((DISCO * FP + SP) / (DISCO + SP))"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    DISCO = v('DISCO')
    return (1 / (1 + DISCO)) * ((DISCO * v('FP') + v('SP')) / (DISCO + v('SP')))


# --------------------------------------------------------------------------- #
# DV  (discount factor - vehicles)
# --------------------------------------------------------------------------- #
def compute_DV(data, t):
    """DV = SV / (DISCO + SV)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('SV') / (v('DISCO') + v('SV'))


# --------------------------------------------------------------------------- #
# WB, WP, WV  (asset-type weights - constants)
# --------------------------------------------------------------------------- #
def compute_WB(data, t):
    """WB = 0.31"""
    return 0.31


def compute_WP(data, t):
    """WP = 0.54"""
    return 0.54


def compute_WV(data, t):
    """WV = 0.14"""
    return 0.14


# --------------------------------------------------------------------------- #
# TAFB, TAFP, TAFV  (tax-adjusted factors)
# --------------------------------------------------------------------------- #
def compute_TAFB(data, t):
    """TAFB = (1 - TCPRO * DB) / (1 - TCPRO)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (1 - v('TCPRO') * v('DB')) / (1 - v('TCPRO'))


def compute_TAFP(data, t):
    """TAFP = (1 - TCPRO * DP) / (1 - TCPRO)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (1 - v('TCPRO') * v('DP')) / (1 - v('TCPRO'))


def compute_TAFV(data, t):
    """TAFV = (1 - TCPRO * DV) / (1 - TCPRO)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (1 - v('TCPRO') * v('DV')) / (1 - v('TCPRO'))


# --------------------------------------------------------------------------- #
# TAF  (weighted tax-adjustment factor)
# --------------------------------------------------------------------------- #
def compute_TAF(data, t):
    """TAF = WB * TAFB + WP * TAFP + WV * TAFV"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('WB') * v('TAFB') + v('WP') * v('TAFP') + v('WV') * v('TAFV')


# --------------------------------------------------------------------------- #
# WG  (growth rate weight - constant)
# --------------------------------------------------------------------------- #
def compute_WG(data, t):
    """WG = 0.03"""
    return 0.03


# --------------------------------------------------------------------------- #
# CDEBT  (cost of debt)
# --------------------------------------------------------------------------- #
def compute_CDEBT(data, t):
    """CDEBT = CDEBT(-1) + d(RIC)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_RIC = v('RIC') - v('RIC', 1)
    return v('CDEBT', 1) + d_RIC


# --------------------------------------------------------------------------- #
# CEQUITY  (cost of equity)
# --------------------------------------------------------------------------- #
def compute_CEQUITY(data, t):
    """CEQUITY = NDIV * (1 + WG) + 100 * WG"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('NDIV') * (1 + v('WG')) + 100 * v('WG')


# --------------------------------------------------------------------------- #
# RWACC  (real weighted average cost of capital)
# --------------------------------------------------------------------------- #
def compute_RWACC(data, t):
    """RWACC = DEBTW * CDEBT + (1 - DEBTW) * CEQUITY"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DEBTW') * v('CDEBT') + (1 - v('DEBTW')) * v('CEQUITY')


# --------------------------------------------------------------------------- #
# RDELTA  (depreciation rate - constant)
# --------------------------------------------------------------------------- #
def compute_RDELTA(data, t):
    """RDELTA = 0.022"""
    return 0.022


# --------------------------------------------------------------------------- #
# COCU  (unadjusted cost of capital)
# --------------------------------------------------------------------------- #
def compute_COCU(data, t):
    """
    COCU = PIBUS / PGDP * @elem(PGDP, "1970Q1") / @elem(PIBUS, "1970Q1")
           * (DELTA + RWACC)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    pgdp_base = elem(data, 'PGDP', '1970Q1')
    pibus_base = elem(data, 'PIBUS', '1970Q1')

    return (v('PIBUS') / v('PGDP')) * (pgdp_base / pibus_base) * (v('DELTA') + v('RWACC'))


# --------------------------------------------------------------------------- #
# COC  (tax-adjusted cost of capital)
# --------------------------------------------------------------------------- #
def compute_COC(data, t):
    """COC = TAF * COCU"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('TAF') * v('COCU')


# --------------------------------------------------------------------------- #
# KSTAR  (desired capital stock)
# --------------------------------------------------------------------------- #
def compute_KSTAR(data, t):
    """KSTAR = exp(log(MSGVA) - 0.4 * log(COC) + 2.434202655)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return np.exp(safe_log(v('MSGVA')) - 0.4 * safe_log(v('COC')) + 2.434202655)


# --------------------------------------------------------------------------- #
# KMSXH  (market-sector capital stock excl. housing)
# --------------------------------------------------------------------------- #
def compute_KMSXH(data, t):
    """KMSXH = (IBUSX / 1000) + KMSXH(-1) * (1 - RDELTA)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('IBUSX') / 1000) + v('KMSXH', 1) * (1 - v('RDELTA'))


# --------------------------------------------------------------------------- #
# KGAP  (capital-stock gap)
# --------------------------------------------------------------------------- #
def compute_KGAP(data, t):
    """KGAP = log(KMSXH * 1000) - log(KSTAR)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return safe_log(v('KMSXH') * 1000) - safe_log(v('KSTAR'))


# --------------------------------------------------------------------------- #
# TQ  (Tobin's Q proxy)
# --------------------------------------------------------------------------- #
def compute_TQ(data, t):
    """TQ = -(NWIC / 1000) / (KMSXH * (PKMSXHB / 100))"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return -(v('NWIC') / 1000) / (v('KMSXH') * (v('PKMSXHB') / 100))


# --------------------------------------------------------------------------- #
# PKMSXHB  (price of capital stock)
# --------------------------------------------------------------------------- #
def compute_PKMSXHB(data, t):
    """PKMSXHB = PIBUS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PIBUS')


# --------------------------------------------------------------------------- #
# IBUS  (business investment)
# --------------------------------------------------------------------------- #
def compute_IBUS(data, t):
    """IBUS = IF - GGI - PCIH - PCLEB - IH - IPRL"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('IF') - v('GGI') - v('PCIH') - v('PCLEB') - v('IH') - v('IPRL')


# --------------------------------------------------------------------------- #
# IBUSX  (business investment excl. one-off)
# --------------------------------------------------------------------------- #
def compute_IBUSX(data, t):
    """IBUSX = IBUS - 17394 * @recode(@date = @dateval("2005:02"), 1, 0)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('IBUS') - 17394 * recode_eq(t, "2005:02")


# --------------------------------------------------------------------------- #
# GGIPS  (general government investment, current prices)
# --------------------------------------------------------------------------- #
def compute_GGIPS(data, t):
    """GGIPS = CGIPS + LAIPS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGIPS') + v('LAIPS')


# --------------------------------------------------------------------------- #
# GGI  (general government investment, volume)
# --------------------------------------------------------------------------- #
def compute_GGI(data, t):
    """GGI = 100 * GGIPS / GGIDEF"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 100 * v('GGIPS') / v('GGIDEF')


# --------------------------------------------------------------------------- #
# GGIX  (general government investment excl. one-off)
# --------------------------------------------------------------------------- #
def compute_GGIX(data, t):
    """GGIX = GGI + 17394 * @recode(@date = @dateval("2005:02"), 1, 0)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GGI') + 17394 * recode_eq(t, "2005:02")


# --------------------------------------------------------------------------- #
# GGIDEF  (GGI deflator - ratio equation)
# --------------------------------------------------------------------------- #
def compute_GGIDEF(data, t):
    """GGIDEF / GGIDEF(-1) = PIF / PIF(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GGIDEF', 1) * (v('PIF') / v('PIF', 1))


# --------------------------------------------------------------------------- #
# HIMPROV  (home improvements - dlog equation)
# --------------------------------------------------------------------------- #
def compute_HIMPROV(data, t):
    """
    dlog(HIMPROV) = -1.936849
                    + 0.0467091 * d(RMORT)
                    - 0.09652566 * dlog(PD(-1))
                    - 0.5129925 * (log(HIMPROV(-1)) - 1.00768 * log(CONSPS(-1)))
                    - 0.0834384 * @recode(@date = @dateval("2003:01"), 1, 0)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_RMORT = v('RMORT') - v('RMORT', 1)
    dlog_PD_1 = np.log(v('PD', 1) / v('PD', 2))

    ecm = safe_log(v('HIMPROV', 1)) - 1.00768 * safe_log(v('CONSPS', 1))

    dummy = recode_eq(t, "2003:01")

    rhs = (-1.936849
           + 0.0467091 * d_RMORT
           - 0.09652566 * dlog_PD_1
           - 0.5129925 * ecm
           - 0.0834384 * dummy)

    return v('HIMPROV', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# PCIH  (public corporations investment in housing - ratio equation)
# --------------------------------------------------------------------------- #
def compute_PCIH(data, t):
    """PCIH / PCIH(-1) = IH / IH(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PCIH', 1) * (v('IH') / v('IH', 1))


# --------------------------------------------------------------------------- #
# VALPS  (valuables, current prices)
# --------------------------------------------------------------------------- #
def compute_VALPS(data, t):
    """VALPS = VAL * PIF / 100"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('VAL') * v('PIF') / 100


# --------------------------------------------------------------------------- #
# VALHH  (household valuables)
# --------------------------------------------------------------------------- #
def compute_VALHH(data, t):
    """VALHH = 0.25 * VALPS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.25 * v('VALPS')


# --------------------------------------------------------------------------- #
# IFPS  (total investment, current prices)
# --------------------------------------------------------------------------- #
def compute_IFPS(data, t):
    """IFPS = IF * PIF / 100"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('IF') * v('PIF') / 100


# --------------------------------------------------------------------------- #
# PIPRL  (private rental investment deflator)
# --------------------------------------------------------------------------- #
def compute_PIPRL(data, t):
    """PIPRL = 100 * IPRLPS / IPRL"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 100 * v('IPRLPS') / v('IPRL')


# --------------------------------------------------------------------------- #
# IHPS  (housing investment, current prices)
# --------------------------------------------------------------------------- #
def compute_IHPS(data, t):
    """IHPS = IH * PIH / 100"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('IH') * v('PIH') / 100


# --------------------------------------------------------------------------- #
# IHHPS  (household investment, current prices - ratio equation)
# --------------------------------------------------------------------------- #
def compute_IHHPS(data, t):
    """
    IHHPS = IHHPS(-1) * (0.8456 * IHPS + 0.5674 * IPRLPS
            + 0.0803 * (PIBUS / 100) * IBUS)
            / (0.8456 * IHPS(-1) + 0.5674 * IPRLPS(-1)
               + 0.0803 * (PIBUS(-1) / 100) * IBUS(-1))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    numerator = (0.8456 * v('IHPS')
                 + 0.5674 * v('IPRLPS')
                 + 0.0803 * (v('PIBUS') / 100) * v('IBUS'))

    denominator = (0.8456 * v('IHPS', 1)
                   + 0.5674 * v('IPRLPS', 1)
                   + 0.0803 * (v('PIBUS', 1) / 100) * v('IBUS', 1))

    return v('IHHPS', 1) * numerator / denominator


# --------------------------------------------------------------------------- #
# PIBUS  (business investment deflator)
# --------------------------------------------------------------------------- #
def compute_PIBUS(data, t):
    """
    PIBUS = 100 * (IFPS - IHPS - IPRLPS
            - (PIF * 0.9828 / 100) * (PCIH + PCLEB) - GGIPS) / IBUS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    numerator = (v('IFPS')
                 - v('IHPS')
                 - v('IPRLPS')
                 - (v('PIF') * 0.9828 / 100) * (v('PCIH') + v('PCLEB'))
                 - v('GGIPS'))

    return 100 * numerator / v('IBUS')


# --------------------------------------------------------------------------- #
# ICCPS  (corporate-sector capital spending)
# --------------------------------------------------------------------------- #
def compute_ICCPS(data, t):
    """ICCPS = 0.1543 * IHPS + 0.4204 * IPRLPS + 0.8331 * (PIBUS / 100) * IBUS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (0.1543 * v('IHPS')
            + 0.4204 * v('IPRLPS')
            + 0.8331 * (v('PIBUS') / 100) * v('IBUS'))


# --------------------------------------------------------------------------- #
# IPCPS  (public corporations investment, current prices)
# --------------------------------------------------------------------------- #
def compute_IPCPS(data, t):
    """IPCPS = (PIF * 0.9828 / 100) * (PCIH + PCLEB) + 0.0456 * (PIBUS / 100) * IBUS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return ((v('PIF') * 0.9828 / 100) * (v('PCIH') + v('PCLEB'))
            + 0.0456 * (v('PIBUS') / 100) * v('IBUS'))


# --------------------------------------------------------------------------- #
# IFCPS  (financial-corporations capital spending)
# --------------------------------------------------------------------------- #
def compute_IFCPS(data, t):
    """IFCPS = IFPS - IHHPS - ICCPS - LAIPS - CGIPS - IPCPS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('IFPS') - v('IHHPS') - v('ICCPS')
            - v('LAIPS') - v('CGIPS') - v('IPCPS'))


# --------------------------------------------------------------------------- #
# NETAD  (net additions to housing stock)
# --------------------------------------------------------------------------- #
def compute_NETAD(data, t):
    """NETAD = (PEHC / 1000) * 1.5166"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('PEHC') / 1000) * 1.5166


# --------------------------------------------------------------------------- #
# HSALL  (total housing stock)
# --------------------------------------------------------------------------- #
def compute_HSALL(data, t):
    """HSALL = HSALL(-1) + NETAD"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('HSALL', 1) + v('NETAD')


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        # Discount factors
        ('DB',       compute_DB,       'identity'),
        ('DP',       compute_DP,       'identity'),
        ('DV',       compute_DV,       'identity'),

        # Weights (constants)
        ('WB',       compute_WB,       'identity'),
        ('WP',       compute_WP,       'identity'),
        ('WV',       compute_WV,       'identity'),

        # Tax-adjusted factors
        ('TAFB',     compute_TAFB,     'identity'),
        ('TAFP',     compute_TAFP,     'identity'),
        ('TAFV',     compute_TAFV,     'identity'),
        ('TAF',      compute_TAF,      'identity'),

        # Cost of capital components
        ('WG',       compute_WG,       'identity'),
        ('CDEBT',    compute_CDEBT,    'identity'),
        ('CEQUITY',  compute_CEQUITY,  'identity'),
        ('RWACC',    compute_RWACC,    'identity'),
        ('RDELTA',   compute_RDELTA,   'identity'),
        ('COCU',     compute_COCU,     'identity'),
        ('COC',      compute_COC,      'identity'),

        # Capital stock
        ('KSTAR',    compute_KSTAR,    'identity'),
        ('KMSXH',    compute_KMSXH,    'identity'),
        ('KGAP',     compute_KGAP,     'identity'),
        ('TQ',       compute_TQ,       'identity'),
        ('PKMSXHB',  compute_PKMSXHB,  'identity'),

        # Business investment
        ('IBUS',     compute_IBUS,     'identity'),
        ('IBUSX',    compute_IBUSX,    'identity'),

        # General government investment
        ('GGIPS',    compute_GGIPS,    'identity'),
        ('GGI',      compute_GGI,      'identity'),
        ('GGIX',     compute_GGIX,     'identity'),
        ('GGIDEF',   compute_GGIDEF,   'ratio'),

        # Housing & home improvements
        ('HIMPROV',  compute_HIMPROV,  'dlog'),
        ('PCIH',     compute_PCIH,     'ratio'),

        # Valuables
        ('VALPS',    compute_VALPS,    'identity'),
        ('VALHH',    compute_VALHH,    'identity'),

        # Investment current-price aggregates and deflators
        ('IFPS',     compute_IFPS,     'identity'),
        ('PIPRL',    compute_PIPRL,    'identity'),
        ('IHPS',     compute_IHPS,     'identity'),
        ('IHHPS',    compute_IHHPS,    'ratio'),
        ('PIBUS',    compute_PIBUS,    'identity'),

        # Sectoral capital spending
        ('ICCPS',    compute_ICCPS,    'identity'),
        ('IPCPS',    compute_IPCPS,    'identity'),
        ('IFCPS',    compute_IFCPS,    'identity'),

        # Housing stock
        ('NETAD',    compute_NETAD,    'identity'),
        ('HSALL',    compute_HSALL,    'identity'),
    ]
