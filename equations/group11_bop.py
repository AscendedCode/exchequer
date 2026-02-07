"""
Group 11: Balance of Payments
Lines 437-497 of obr_model.txt

Equations:
    RXD      - Real exchange rate (ratio)
    ECUPO    - ECU/Sterling parity (ratio)
    DRES     - Reserves change (d)
    SRES     - Sterling reserves (identity)
    CIPD     - Credits on investment income (identity)
    REXC     - Return on overseas assets (identity)
    DIPD     - Debits on investment income (identity)
    CGCBOP   - Central government BOP contribution (ratio)
    NIPD     - Net investment income (identity)
    EECOMPD  - Compensation of employees debits (dlog)
    EECOMPC  - Compensation of employees credits (ratio)
    EUSUBP   - EU subsidies payments (identity)
    EUSUBPR  - EU subsidies receipts (identity)
    EUSF     - EU social fund (identity)
    ECNET    - EC net contribution (identity)
    GNP4     - GNP adjustment (identity)
    EUVAT    - EU VAT contribution (identity)
    BENAB    - Benefits abroad (identity)
    CGITFA   - CG international transfers (identity)
    ITA      - International transfers (identity)
    HHTFA    - Household transfers from abroad (identity)
    HHTA     - Household transfers abroad (ratio)
    TRANC    - Transfer credits (identity)
    TRAND    - Transfer debits (identity)
    TRANB    - Transfer balance (identity)
    CGKTA    - CG capital transfers abroad (identity)
    TB       - Trade balance (identity)
    CB       - Current balance (identity)
    CBPCNT   - Current balance as % of GDP (identity)
    NAFROW   - Net acquisition of financial assets ROW (identity)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# --------------------------------------------------------------------------- #
# RXD = RXD(-1) * RX / RX(-1)
# --------------------------------------------------------------------------- #
def compute_RXD(data, t):
    """RXD = RXD(-1) * RX / RX(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('RXD', 1) * v('RX') / v('RX', 1)


# --------------------------------------------------------------------------- #
# ECUPO = ECUPO(-1) * RX / RX(-1)
# --------------------------------------------------------------------------- #
def compute_ECUPO(data, t):
    """ECUPO = (ECUPO(-1) * RX / RX(-1))"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('ECUPO', 1) * v('RX') / v('RX', 1)


# --------------------------------------------------------------------------- #
# d(DRES) = 0
# --------------------------------------------------------------------------- #
def compute_DRES(data, t):
    """d(DRES) = 0"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DRES', 1) + 0


# --------------------------------------------------------------------------- #
# SRES = -DRES + (1 + 0.227*(RXD(-1)/RXD - 1) + 0.364*(RX(-1)/RX - 1)) * SRES(-1)
# --------------------------------------------------------------------------- #
def compute_SRES(data, t):
    """SRES = -DRES + (1 + 0.227*(RXD(-1)/RXD - 1) + 0.364*(RX(-1)/RX - 1)) * SRES(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (-v('DRES')
            + (1 + 0.227 * (v('RXD', 1) / v('RXD') - 1)
               + 0.364 * (v('RX', 1) / v('RX') - 1))
            * v('SRES', 1))


# --------------------------------------------------------------------------- #
# CIPD = (0.7173 * CIPD(-1)/LROW(-2) + (1-0.7173) * REXC/100) * LROW(-1)
# --------------------------------------------------------------------------- #
def compute_CIPD(data, t):
    """CIPD = (0.7173 * CIPD(-1)/LROW(-2) + (1-0.7173) * REXC/100) * LROW(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (0.7173 * v('CIPD', 1) / v('LROW', 2)
            + (1 - 0.7173) * v('REXC') / 100) * v('LROW', 1)


# --------------------------------------------------------------------------- #
# REXC = portfolio return on overseas assets
# --------------------------------------------------------------------------- #
def compute_REXC(data, t):
    """
    REXC = (DLROW(-1)/LROW(-1)) * (1.24 + 1.91*(log(WEQPR) - log(WEQPR(-4))) + 0.57*R/4)
         + (EQLROW(-1)/LROW(-1)) * (0.41 + 0.17*(log(WEQPR) - log(WEQPR(-4))))
         + (BLROW(-1)/LROW(-1)) * (0.30 + 0.82*(ROLT/4))
         + (OTLROW(-1)/LROW(-1)) * (0.09 + 0.8*ROCB/4)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    dlog_WEQPR = safe_log(v('WEQPR')) - safe_log(v('WEQPR', 4))

    rexc = ((v('DLROW', 1) / v('LROW', 1))
            * (1.24 + 1.91 * dlog_WEQPR + 0.57 * v('R') / 4)
            + (v('EQLROW', 1) / v('LROW', 1))
            * (0.41 + 0.17 * dlog_WEQPR)
            + (v('BLROW', 1) / v('LROW', 1))
            * (0.30 + 0.82 * (v('ROLT') / 4))
            + (v('OTLROW', 1) / v('LROW', 1))
            * (0.09 + 0.8 * v('ROCB') / 4))

    return rexc


# --------------------------------------------------------------------------- #
# DIPD = (0.6283 * DIPD(-1)/AROW(-2) + (1-0.6283) * REXD_RETURN/100) * AROW(-1)
# where REXD_RETURN is a return on domestic assets computed inline.
# --------------------------------------------------------------------------- #
def compute_DIPD(data, t):
    """
    DIPD = (0.6283 * DIPD(-1)/AROW(-2) + (1-0.6283) * REXD/100) * AROW(-1)

    where REXD (return on domestic assets, distinct from the exchange rate RXD) is:
    REXD = (DAROW(-1)/AROW(-1)) * (0.62 + 2.36*FYCPR/GDPMPS
                - 1.64*@recode(@date=@dateval("1998:03"),1,0))
         + (EQAROW(-1)/AROW(-1)) * (0.57 + 15.33*NDIVHH/EQHH)
         + (BAROW(-1)/AROW(-1)) * (0.23 + 1.04*RL/4)
         + (OTAROW(-1)/AROW(-1)) * (0.18 + 0.14*R/4 + 0.78*ROCB/4)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    # Compute REXD_RETURN (return on domestic assets) inline
    dummy_1998q3 = recode_eq(t, "1998:03")

    rexd_return = ((v('DAROW', 1) / v('AROW', 1))
                   * (0.62 + 2.36 * v('FYCPR') / v('GDPMPS')
                      - 1.64 * dummy_1998q3)
                   + (v('EQAROW', 1) / v('AROW', 1))
                   * (0.57 + 15.33 * v('NDIVHH') / v('EQHH'))
                   + (v('BAROW', 1) / v('AROW', 1))
                   * (0.23 + 1.04 * v('RL') / 4)
                   + (v('OTAROW', 1) / v('AROW', 1))
                   * (0.18 + 0.14 * v('R') / 4 + 0.78 * v('ROCB') / 4))

    return (0.6283 * v('DIPD', 1) / v('AROW', 2)
            + (1 - 0.6283) * rexd_return / 100) * v('AROW', 1)


# --------------------------------------------------------------------------- #
# d(CGCBOP) / CGCBOP(-1) = d(CGC) / CGC(-1)
# -> CGCBOP = CGCBOP(-1) + CGCBOP(-1) * (CGC - CGC(-1)) / CGC(-1)
# --------------------------------------------------------------------------- #
def compute_CGCBOP(data, t):
    """d(CGCBOP) / CGCBOP(-1) = d(CGC) / CGC(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGCBOP', 1) + v('CGCBOP', 1) * (v('CGC') - v('CGC', 1)) / v('CGC', 1)


# --------------------------------------------------------------------------- #
# NIPD = CIPD - DIPD + CGCBOP
# --------------------------------------------------------------------------- #
def compute_NIPD(data, t):
    """NIPD = CIPD - DIPD + CGCBOP"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CIPD') - v('DIPD') + v('CGCBOP')


# --------------------------------------------------------------------------- #
# dlog(EECOMPD) = -0.492198*log(EECOMPD(-1)) + 0.693337*log(FYEMP(-1))
#   + 2.148955*dlog(FYEMP) + 0.107609*@recode(@date>=@dateval("2005:01"),1,0)
#   - 0.004629*@TREND(1979Q4) - 5.105951
# --------------------------------------------------------------------------- #
def compute_EECOMPD(data, t):
    """
    dlog(EECOMPD) = -0.492198*log(EECOMPD(-1)) + 0.693337*log(FYEMP(-1))
        + 2.148955*dlog(FYEMP)
        + 0.107609*@recode(@date >= @dateval("2005:01"), 1, 0)
        - 0.004629*@TREND(1979Q4) - 5.105951
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    dlog_FYEMP = np.log(v('FYEMP') / v('FYEMP', 1))
    dummy_2005 = recode_geq(t, "2005:01")
    t_trend = trend(t, "1979Q4")

    rhs = (-0.492198 * safe_log(v('EECOMPD', 1))
           + 0.693337 * safe_log(v('FYEMP', 1))
           + 2.148955 * dlog_FYEMP
           + 0.107609 * dummy_2005
           - 0.004629 * t_trend
           - 5.105951)

    return v('EECOMPD', 1) * np.exp(rhs)


# --------------------------------------------------------------------------- #
# EECOMPC / EECOMPC(-1) = MAJGDP / MAJGDP(-1)
# --------------------------------------------------------------------------- #
def compute_EECOMPC(data, t):
    """EECOMPC / EECOMPC(-1) = MAJGDP / MAJGDP(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EECOMPC', 1) * v('MAJGDP') / v('MAJGDP', 1)


# --------------------------------------------------------------------------- #
# EUSUBP = 0
# --------------------------------------------------------------------------- #
def compute_EUSUBP(data, t):
    """EUSUBP = 0"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0


# --------------------------------------------------------------------------- #
# EUSUBPR = EUSUBPR(-1) * ECUPO(-1) / ECUPO
# --------------------------------------------------------------------------- #
def compute_EUSUBPR(data, t):
    """EUSUBPR = EUSUBPR(-1) * ECUPO(-1) / ECUPO"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EUSUBPR', 1) * v('ECUPO', 1) / v('ECUPO')


# --------------------------------------------------------------------------- #
# EUSF = EUSF(-1) * ECUPO(-1) / ECUPO
# --------------------------------------------------------------------------- #
def compute_EUSF(data, t):
    """EUSF = EUSF(-1) * ECUPO(-1) / ECUPO"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EUSF', 1) * v('ECUPO', 1) / v('ECUPO')


# --------------------------------------------------------------------------- #
# ECNET = (1 - 0.5*(ECUPO(-1)/ECUPO - 1)) * ECNET(-1)
# --------------------------------------------------------------------------- #
def compute_ECNET(data, t):
    """ECNET = (1 - 0.5*(ECUPO(-1)/ECUPO - 1)) * ECNET(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (1 - 0.5 * (v('ECUPO', 1) / v('ECUPO') - 1)) * v('ECNET', 1)


# --------------------------------------------------------------------------- #
# GNP4 = 0.010 * ((GDPMPS + NIPD + EECOMPC - EECOMPD) / ECUPO(-4))
# --------------------------------------------------------------------------- #
def compute_GNP4(data, t):
    """GNP4 = 0.010 * ((GDPMPS + NIPD + EECOMPC - EECOMPD) / ECUPO(-4))"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.010 * ((v('GDPMPS') + v('NIPD') + v('EECOMPC') - v('EECOMPD'))
                    / v('ECUPO', 4))


# --------------------------------------------------------------------------- #
# EUVAT = 0.0325 * VREC / (0.8267 * ECUPO(-4))
# --------------------------------------------------------------------------- #
def compute_EUVAT(data, t):
    """EUVAT = 0.0325 * VREC / (0.8267 * ECUPO(-4))"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.0325 * v('VREC') / (0.8267 * v('ECUPO', 4))


# --------------------------------------------------------------------------- #
# BENAB = 0.012 * CGSB
# --------------------------------------------------------------------------- #
def compute_BENAB(data, t):
    """BENAB = 0.012 * CGSB"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.012 * v('CGSB')


# --------------------------------------------------------------------------- #
# CGITFA = ITA
# --------------------------------------------------------------------------- #
def compute_CGITFA(data, t):
    """CGITFA = ITA"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('ITA')


# --------------------------------------------------------------------------- #
# ITA = 0.001115 * WFP
# --------------------------------------------------------------------------- #
def compute_ITA(data, t):
    """ITA = 0.001115 * WFP"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.001115 * v('WFP')


# --------------------------------------------------------------------------- #
# log(HHTFA) = log(HHTFA(-1) * MAJGDP / MAJGDP(-1))
# -> HHTFA = HHTFA(-1) * MAJGDP / MAJGDP(-1)
# --------------------------------------------------------------------------- #
def compute_HHTFA(data, t):
    """log(HHTFA) = log(HHTFA(-1) * MAJGDP / MAJGDP(-1))"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('HHTFA', 1) * v('MAJGDP') / v('MAJGDP', 1)


# --------------------------------------------------------------------------- #
# HHTA / HHTA(-1) = WFP / WFP(-1)
# --------------------------------------------------------------------------- #
def compute_HHTA(data, t):
    """HHTA / HHTA(-1) = WFP / WFP(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('HHTA', 1) * v('WFP') / v('WFP', 1)


# --------------------------------------------------------------------------- #
# TRANC = EUSUBP + HHTFA + EUSF + CGITFA + EUSUBPR + INSURE
# --------------------------------------------------------------------------- #
def compute_TRANC(data, t):
    """TRANC = EUSUBP + HHTFA + EUSF + CGITFA + EUSUBPR + INSURE"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('EUSUBP') + v('HHTFA') + v('EUSF')
            + v('CGITFA') + v('EUSUBPR') + v('INSURE'))


# --------------------------------------------------------------------------- #
# TRAND = TROD + ECNET + EUVAT + EUOT + HHTA + GNP4 + BENAB + ITA + INSURE
# --------------------------------------------------------------------------- #
def compute_TRAND(data, t):
    """TRAND = TROD + ECNET + EUVAT + EUOT + HHTA + GNP4 + BENAB + ITA + INSURE"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('TROD') + v('ECNET') + v('EUVAT') + v('EUOT')
            + v('HHTA') + v('GNP4') + v('BENAB') + v('ITA') + v('INSURE'))


# --------------------------------------------------------------------------- #
# TRANB = TRANC - TRAND
# --------------------------------------------------------------------------- #
def compute_TRANB(data, t):
    """TRANB = TRANC - TRAND"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('TRANC') - v('TRAND')


# --------------------------------------------------------------------------- #
# CGKTA = 0.02351 * KCGPSO
# --------------------------------------------------------------------------- #
def compute_CGKTA(data, t):
    """CGKTA = 0.02351 * KCGPSO"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0.02351 * v('KCGPSO')


# --------------------------------------------------------------------------- #
# TB = XPS - MPS
# --------------------------------------------------------------------------- #
def compute_TB(data, t):
    """TB = XPS - MPS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('XPS') - v('MPS')


# --------------------------------------------------------------------------- #
# CB = TB + (EECOMPC - EECOMPD) + NIPD + TRANC - TRAND
# --------------------------------------------------------------------------- #
def compute_CB(data, t):
    """CB = TB + (EECOMPC - EECOMPD) + NIPD + TRANC - TRAND"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('TB') + (v('EECOMPC') - v('EECOMPD'))
            + v('NIPD') + v('TRANC') - v('TRAND'))


# --------------------------------------------------------------------------- #
# CBPCNT = (CB / GDPMPS) * 100
# --------------------------------------------------------------------------- #
def compute_CBPCNT(data, t):
    """CBPCNT = (CB / GDPMPS) * 100"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('CB') / v('GDPMPS')) * 100


# --------------------------------------------------------------------------- #
# NAFROW = -(CB + EUKT - (CGKTA + OPSKTA) + NPAA)
# --------------------------------------------------------------------------- #
def compute_NAFROW(data, t):
    """NAFROW = -(CB + (EUKT) - (CGKTA + OPSKTA) + NPAA)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return -(v('CB') + v('EUKT') - (v('CGKTA') + v('OPSKTA')) + v('NPAA'))


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('RXD',      compute_RXD,      'identity'),
        ('ECUPO',    compute_ECUPO,    'identity'),
        ('DRES',     compute_DRES,     'd'),
        ('SRES',     compute_SRES,     'identity'),
        ('REXC',     compute_REXC,     'identity'),
        ('CIPD',     compute_CIPD,     'identity'),
        ('DIPD',     compute_DIPD,     'identity'),
        ('CGCBOP',   compute_CGCBOP,   'identity'),
        ('NIPD',     compute_NIPD,     'identity'),
        ('EECOMPD',  compute_EECOMPD,  'dlog'),
        ('EECOMPC',  compute_EECOMPC,  'ratio'),
        ('EUSUBP',   compute_EUSUBP,   'identity'),
        ('EUSUBPR',  compute_EUSUBPR,  'identity'),
        ('EUSF',     compute_EUSF,     'identity'),
        ('ECNET',    compute_ECNET,    'identity'),
        ('GNP4',     compute_GNP4,     'identity'),
        ('EUVAT',    compute_EUVAT,    'identity'),
        ('BENAB',    compute_BENAB,    'identity'),
        ('ITA',      compute_ITA,      'identity'),
        ('CGITFA',   compute_CGITFA,   'identity'),
        ('HHTFA',    compute_HHTFA,    'identity'),
        ('HHTA',     compute_HHTA,     'ratio'),
        ('TRANC',    compute_TRANC,    'identity'),
        ('TRAND',    compute_TRAND,    'identity'),
        ('TRANB',    compute_TRANB,    'identity'),
        ('CGKTA',    compute_CGKTA,    'identity'),
        ('TB',       compute_TB,       'identity'),
        ('CB',       compute_CB,       'identity'),
        ('CBPCNT',   compute_CBPCNT,   'identity'),
        ('NAFROW',   compute_NAFROW,   'identity'),
    ]
