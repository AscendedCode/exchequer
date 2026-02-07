"""
Group 16: Gross Domestic Product
Lines 681-751 of obr_model.txt

Group 17: Market Sector GVA Satellite
Lines 756-762 of obr_model.txt

Equations (Group 16):
    TFEPS    - Total final expenditure, current prices (identity)
    SDEPS    - Statistical discrepancy, current prices (identity)
    GDPMPS   - GDP at market prices, current prices (identity)
    MGDPNSA  - Monitored GDP NSA (identity)
    BPAPS    - Basic price adjustment, current prices (identity)
    GVAPS    - GVA at current prices (identity)
    TFE      - Total final expenditure, volume (identity)
    BPA      - Basic price adjustment, volume (ratio)
    GVA      - Gross value added, volume (identity)
    PGVA     - GVA deflator (identity)
    TPRODPS  - Taxes on production, current prices (identity)
    SDI      - Statistical discrepancy, income (identity)
    OS       - Operating surplus (identity)
    RENTCO   - Rental income of corporates (ratio)
    IROO     - Imputed rent on owner-occupied housing (identity)
    OSHH     - Operating surplus, households (identity)
    FISIMGG  - FISIM general government (identity)
    FISIMPS  - FISIM total, current prices (identity)
    FYCPR    - Full year corporate profits (identity)
    OSCO     - Operating surplus, corporates (identity)
    GTPFC    - Gross trading profits, financial corporates (identity)
    FC       - Financial corporates (identity)
    GNIPS    - GNI at current prices (identity)
    NNSGVA   - Non-North-Sea GVA (identity)
    GAP      - Output gap (identity)
    GDPMAL   - GDP per person of working age (identity)
    TRGDPAL  - Trend GDP per person of working age (identity)
    GDPM16   - GDP per person 16+ (identity)
    TRGDP16  - Trend GDP per person 16+ (identity)

Equations (Group 17 - Market Sector GVA Satellite):
    GGVAPS   - General government GVA, current prices (identity)
    MSGVAPS  - Market sector GVA, current prices (identity)
    GGVA     - General government GVA, volume (ratio)
    MSGVA    - Market sector GVA, volume (identity)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# =========================================================================== #
# Group 16: Gross Domestic Product
# =========================================================================== #


# --------------------------------------------------------------------------- #
# TFEPS
# --------------------------------------------------------------------------- #
def compute_TFEPS(data, t):
    """
    TFEPS = CGGPS + CONSPS + DINVPS + VALPS + IFPS + XPS

    EViews original (line 681):
    TFEPS = CGGPS + CONSPS + DINVPS + VALPS + IFPS + XPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('CGGPS') + v('CONSPS') + v('DINVPS')
            + v('VALPS') + v('IFPS') + v('XPS'))


# --------------------------------------------------------------------------- #
# SDEPS
# --------------------------------------------------------------------------- #
def compute_SDEPS(data, t):
    """
    SDEPS = PGDP * SDE / 100

    EViews original (line 683):
    SDEPS = PGDP * SDE / 100
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PGDP') * v('SDE') / 100


# --------------------------------------------------------------------------- #
# GDPMPS
# --------------------------------------------------------------------------- #
def compute_GDPMPS(data, t):
    """
    GDPMPS = TFEPS - MPS + SDEPS

    EViews original (line 688):
    GDPMPS = TFEPS - MPS + SDEPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('TFEPS') - v('MPS') + v('SDEPS')


# --------------------------------------------------------------------------- #
# MGDPNSA
# --------------------------------------------------------------------------- #
def compute_MGDPNSA(data, t):
    """
    MGDPNSA = GDPMPS

    EViews original (line 692):
    MGDPNSA = GDPMPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GDPMPS')


# --------------------------------------------------------------------------- #
# BPAPS
# --------------------------------------------------------------------------- #
def compute_BPAPS(data, t):
    """
    BPAPS = (CETAX - BETPRF) + EXDUTAC + XLAVAT + LAVAT + TSD + TXMIS + ROCS
            - (EUSUBP + LASUBP + CGSUBP + CCLACA) + BANKROLL + BLEVY

    EViews original (line 695):
    BPAPS = (CETAX - BETPRF) + EXDUTAC + XLAVAT + LAVAT + TSD + TXMIS + ROCS
            - (EUSUBP + LASUBP + CGSUBP + CCLACA) + BANKROLL + BLEVY
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return ((v('CETAX') - v('BETPRF'))
            + v('EXDUTAC') + v('XLAVAT') + v('LAVAT')
            + v('TSD') + v('TXMIS') + v('ROCS')
            - (v('EUSUBP') + v('LASUBP') + v('CGSUBP') + v('CCLACA'))
            + v('BANKROLL') + v('BLEVY'))


# --------------------------------------------------------------------------- #
# GVAPS
# --------------------------------------------------------------------------- #
def compute_GVAPS(data, t):
    """
    GVAPS = GDPMPS - BPAPS

    EViews original (line 697):
    GVAPS = GDPMPS - BPAPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GDPMPS') - v('BPAPS')


# --------------------------------------------------------------------------- #
# TFE
# --------------------------------------------------------------------------- #
def compute_TFE(data, t):
    """
    TFE = CGG + CONS + DINV + VAL + IF + X

    EViews original (line 699):
    TFE = CGG + CONS + DINV + VAL + IF + X
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('CGG') + v('CONS') + v('DINV')
            + v('VAL') + v('IF') + v('X'))


# --------------------------------------------------------------------------- #
# BPA
# --------------------------------------------------------------------------- #
def compute_BPA(data, t):
    """
    BPA / BPA(-1) = GDPM / GDPM(-1)

    EViews original (line 705):
    BPA / BPA(-1) = GDPM / GDPM(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('BPA', 1) * (v('GDPM') / v('GDPM', 1))


# --------------------------------------------------------------------------- #
# GVA
# --------------------------------------------------------------------------- #
def compute_GVA(data, t):
    """
    GVA = GDPM - BPA

    EViews original (line 707):
    GVA = GDPM - BPA
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GDPM') - v('BPA')


# --------------------------------------------------------------------------- #
# PGVA
# --------------------------------------------------------------------------- #
def compute_PGVA(data, t):
    """
    PGVA = 100 * GVAPS / GVA

    EViews original (line 709):
    PGVA = 100 * GVAPS / GVA
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 100 * v('GVAPS') / v('GVA')


# --------------------------------------------------------------------------- #
# TPRODPS
# --------------------------------------------------------------------------- #
def compute_TPRODPS(data, t):
    """
    TPRODPS = NNDRA + NIS + VEDCO + OPT + LAPT + EUETS
              - CGSUBPR - LASUBPR - EUSUBPR

    EViews original (line 715):
    TPRODPS = NNDRA + NIS + VEDCO + OPT + LAPT + EUETS
              - CGSUBPR - LASUBPR - EUSUBPR
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('NNDRA') + v('NIS') + v('VEDCO')
            + v('OPT') + v('LAPT') + v('EUETS')
            - v('CGSUBPR') - v('LASUBPR') - v('EUSUBPR'))


# --------------------------------------------------------------------------- #
# SDI
# --------------------------------------------------------------------------- #
def compute_SDI(data, t):
    """
    SDI = SDI(-1)

    EViews original (line 717):
    SDI = SDI(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('SDI', 1)


# --------------------------------------------------------------------------- #
# OS
# --------------------------------------------------------------------------- #
def compute_OS(data, t):
    """
    OS = GDPMPS - FYEMP - MI - BPAPS - TPRODPS - SDI

    EViews original (line 719):
    OS = GDPMPS - FYEMP - MI - BPAPS - TPRODPS - SDI
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('GDPMPS') - v('FYEMP') - v('MI')
            - v('BPAPS') - v('TPRODPS') - v('SDI'))


# --------------------------------------------------------------------------- #
# RENTCO
# --------------------------------------------------------------------------- #
def compute_RENTCO(data, t):
    """
    RENTCO / RENTCO(-1) = GDPMPS / GDPMPS(-1)

    EViews original (line 721):
    RENTCO / RENTCO(-1) = GDPMPS / GDPMPS(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('RENTCO', 1) * (v('GDPMPS') / v('GDPMPS', 1))


# --------------------------------------------------------------------------- #
# IROO
# --------------------------------------------------------------------------- #
def compute_IROO(data, t):
    """
    IROO = (PRENT * POP16) / 1000

    EViews original (line 723):
    IROO = (PRENT * POP16) / 1000
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('PRENT') * v('POP16')) / 1000


# --------------------------------------------------------------------------- #
# OSHH
# --------------------------------------------------------------------------- #
def compute_OSHH(data, t):
    """
    OSHH = (12874 + 0.85 * IROO - DIPHHmf)

    EViews original (line 725):
    OSHH = (12874 + 0.85 * IROO - DIPHHmf)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 12874 + 0.85 * v('IROO') - v('DIPHHmf')


# --------------------------------------------------------------------------- #
# FISIMGG
# --------------------------------------------------------------------------- #
def compute_FISIMGG(data, t):
    """
    FISIMGG = 0

    EViews original (line 727):
    FISIMGG = 0
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0


# --------------------------------------------------------------------------- #
# FISIMPS
# --------------------------------------------------------------------------- #
def compute_FISIMPS(data, t):
    """
    FISIMPS = (DIRHHf + DIPHHuf + DIPHHmf) + (DIRICf + DIPICf)
              + FISIMGG + FISIMROW

    EViews original (line 729):
    FISIMPS = (DIRHHf + DIPHHuf + DIPHHmf) + (DIRICf + DIPICf)
              + FISIMGG + FISIMROW
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return ((v('DIRHHf') + v('DIPHHuf') + v('DIPHHmf'))
            + (v('DIRICf') + v('DIPICf'))
            + v('FISIMGG') + v('FISIMROW'))


# --------------------------------------------------------------------------- #
# FYCPR
# --------------------------------------------------------------------------- #
def compute_FYCPR(data, t):
    """
    FYCPR = OS - OSHH - OSGG - OSPC - RENTCO + SA - FISIMPS

    EViews original (line 731):
    FYCPR = OS - OSHH - OSGG - OSPC - RENTCO + SA - FISIMPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('OS') - v('OSHH') - v('OSGG') - v('OSPC')
            - v('RENTCO') + v('SA') - v('FISIMPS'))


# --------------------------------------------------------------------------- #
# OSCO
# --------------------------------------------------------------------------- #
def compute_OSCO(data, t):
    """
    OSCO = OS - OSHH - OSGG - OSPC

    EViews original (line 733):
    OSCO = OS - OSHH - OSGG - OSPC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('OS') - v('OSHH') - v('OSGG') - v('OSPC')


# --------------------------------------------------------------------------- #
# GTPFC
# --------------------------------------------------------------------------- #
def compute_GTPFC(data, t):
    """
    GTPFC = FYCPR - NNSGTP - NSGTP

    EViews original (line 735):
    GTPFC = FYCPR - NNSGTP - NSGTP
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('FYCPR') - v('NNSGTP') - v('NSGTP')


# --------------------------------------------------------------------------- #
# FC
# --------------------------------------------------------------------------- #
def compute_FC(data, t):
    """
    FC = FISIMPS + GTPFC

    EViews original (line 737):
    FC = FISIMPS + GTPFC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('FISIMPS') + v('GTPFC')


# --------------------------------------------------------------------------- #
# GNIPS
# --------------------------------------------------------------------------- #
def compute_GNIPS(data, t):
    """
    GNIPS = GDPMPS + NIPD + (EECOMPC - EECOMPD) + (EUSUBPR + EUSUBP)
            - (EUOT + EUVAT)

    EViews original (line 739):
    GNIPS = GDPMPS + NIPD + (EECOMPC - EECOMPD) + (EUSUBPR + EUSUBP)
            - (EUOT + EUVAT)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('GDPMPS') + v('NIPD')
            + (v('EECOMPC') - v('EECOMPD'))
            + (v('EUSUBPR') + v('EUSUBP'))
            - (v('EUOT') + v('EUVAT')))


# --------------------------------------------------------------------------- #
# NNSGVA
# --------------------------------------------------------------------------- #
def compute_NNSGVA(data, t):
    """
    NNSGVA = GVA - NSGVA

    EViews original (line 741):
    NNSGVA = GVA - NSGVA
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GVA') - v('NSGVA')


# --------------------------------------------------------------------------- #
# GAP
# --------------------------------------------------------------------------- #
def compute_GAP(data, t):
    """
    GAP = GDPM / TRGDP * 100 - 100

    EViews original (line 743):
    GAP = GDPM / TRGDP * 100 - 100
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GDPM') / v('TRGDP') * 100 - 100


# --------------------------------------------------------------------------- #
# GDPMAL
# --------------------------------------------------------------------------- #
def compute_GDPMAL(data, t):
    """
    GDPMAL = GDPM / POPAL

    EViews original (line 745):
    GDPMAL = GDPM / POPAL
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GDPM') / v('POPAL')


# --------------------------------------------------------------------------- #
# TRGDPAL
# --------------------------------------------------------------------------- #
def compute_TRGDPAL(data, t):
    """
    TRGDPAL = TRGDP / POPAL

    EViews original (line 747):
    TRGDPAL = TRGDP / POPAL
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('TRGDP') / v('POPAL')


# --------------------------------------------------------------------------- #
# GDPM16
# --------------------------------------------------------------------------- #
def compute_GDPM16(data, t):
    """
    GDPM16 = GDPM / POP16

    EViews original (line 749):
    GDPM16 = GDPM / POP16
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GDPM') / v('POP16')


# --------------------------------------------------------------------------- #
# TRGDP16
# --------------------------------------------------------------------------- #
def compute_TRGDP16(data, t):
    """
    TRGDP16 = TRGDP / POP16

    EViews original (line 751):
    TRGDP16 = TRGDP / POP16
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('TRGDP') / v('POP16')


# =========================================================================== #
# Group 17: Market Sector GVA Satellite (lines 756-762)
# =========================================================================== #


# --------------------------------------------------------------------------- #
# GGVAPS
# --------------------------------------------------------------------------- #
def compute_GGVAPS(data, t):
    """
    GGVAPS = CGWS + LAWS + OSGG

    EViews original (line 756):
    GGVAPS = CGWS + LAWS + OSGG
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGWS') + v('LAWS') + v('OSGG')


# --------------------------------------------------------------------------- #
# MSGVAPS
# --------------------------------------------------------------------------- #
def compute_MSGVAPS(data, t):
    """
    MSGVAPS = GVAPS - GGVAPS

    EViews original (line 758):
    MSGVAPS = GVAPS - GGVAPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GVAPS') - v('GGVAPS')


# --------------------------------------------------------------------------- #
# GGVA
# --------------------------------------------------------------------------- #
def compute_GGVA(data, t):
    """
    GGVA / GGVA(-1) = CGG / CGG(-1)

    EViews original (line 760):
    GGVA / GGVA(-1) = CGG / CGG(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GGVA', 1) * (v('CGG') / v('CGG', 1))


# --------------------------------------------------------------------------- #
# MSGVA
# --------------------------------------------------------------------------- #
def compute_MSGVA(data, t):
    """
    MSGVA = GVA - GGVA

    EViews original (line 762):
    MSGVA = GVA - GGVA
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GVA') - v('GGVA')


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        # Group 16: GDP
        ('TFEPS',   compute_TFEPS,   'identity'),
        ('SDEPS',   compute_SDEPS,   'identity'),
        ('GDPMPS',  compute_GDPMPS,  'identity'),
        ('MGDPNSA', compute_MGDPNSA, 'identity'),
        ('BPAPS',   compute_BPAPS,   'identity'),
        ('GVAPS',   compute_GVAPS,   'identity'),
        ('TFE',     compute_TFE,     'identity'),
        ('BPA',     compute_BPA,     'ratio'),
        ('GVA',     compute_GVA,     'identity'),
        ('PGVA',    compute_PGVA,    'identity'),
        ('TPRODPS', compute_TPRODPS, 'identity'),
        ('SDI',     compute_SDI,     'identity'),
        ('OS',      compute_OS,      'identity'),
        ('RENTCO',  compute_RENTCO,  'ratio'),
        ('IROO',    compute_IROO,    'identity'),
        ('OSHH',    compute_OSHH,    'identity'),
        ('FISIMGG', compute_FISIMGG, 'identity'),
        ('FISIMPS', compute_FISIMPS, 'identity'),
        ('FYCPR',   compute_FYCPR,   'identity'),
        ('OSCO',    compute_OSCO,    'identity'),
        ('GTPFC',   compute_GTPFC,   'identity'),
        ('FC',      compute_FC,      'identity'),
        ('GNIPS',   compute_GNIPS,   'identity'),
        ('NNSGVA',  compute_NNSGVA,  'identity'),
        ('GAP',     compute_GAP,     'identity'),
        ('GDPMAL',  compute_GDPMAL,  'identity'),
        ('TRGDPAL', compute_TRGDPAL, 'identity'),
        ('GDPM16',  compute_GDPM16,  'identity'),
        ('TRGDP16', compute_TRGDP16, 'identity'),
        # Group 17: Market Sector GVA Satellite
        ('GGVAPS',  compute_GGVAPS,  'identity'),
        ('MSGVAPS', compute_MSGVAPS, 'identity'),
        ('GGVA',    compute_GGVA,    'ratio'),
        ('MSGVA',   compute_MSGVA,   'identity'),
    ]
