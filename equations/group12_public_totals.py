"""
Group 12: Public Sector Totals
Lines 504-571 of obr_model.txt

Equations:
    CGSUBP   - CG subsidies payments (identity)
    DEP      - Depreciation (identity)
    PSCB     - Public sector current budget (identity)
    NPACG    - Net purchases of assets CG (4Q average)
    NPALA    - Net purchases of assets LA (4Q average)
    PSGI     - Public sector gross investment (identity)
    TME      - Total managed expenditure (identity)
    CGNB     - Central government net borrowing (identity)
    LANB     - Local authority net borrowing (identity)
    GGNB     - General government net borrowing (identity)
    GGNBCY   - GGNB calendar year (identity)
    PCNB     - Public corporations net borrowing (identity)
    PCNBCY   - PCNB calendar year (identity)
    PSNBNSA  - Public sector net borrowing NSA (identity)
    PSNBCY   - PSNB calendar year (identity)
    SWAPS    - Swaps (identity)
    TDEF     - Total deficit (identity)
    CGLSFA   - CG lending and sale of financial assets (identity)
    PSLSFA   - PS lending and sale of financial assets (identity)
    CGACADJ  - CG accruals adjustment (identity)
    PSACADJ  - PS accruals adjustment (identity)
    PSFL     - PS financial liabilities (identity)
    PSTA     - PS tangible assets (dynamic)
    PSNW     - PS net worth (identity)
    LABRO    - LA borrowing requirement (identity)
    CGNCR    - CG net cash requirement (identity)
    PSNCR    - PS net cash requirement (identity)
    COIN     - Coin (ratio, 4Q lag)
    PSND     - PS net debt (d)
    GGLIQ    - GG liquid assets (identity)
    GGGD     - GG gross debt (d)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# --------------------------------------------------------------------------- #
# CGSUBP = PSCE - (CGWS + CGP + RCGIM + LAWS + LAPR + RLAIM) - LATSUB
#          - (CGSB + LASBHH) - CGNCGA - ECNET - LANCGA
#          - (CGOTR + LAOTRHH) - (DICGOP + DILAPR + DIPCOP)
#          - EUVAT - GNP4 - CGSUBPR
# --------------------------------------------------------------------------- #
def compute_CGSUBP(data, t):
    """
    CGSUBP = PSCE - (CGWS + CGP + RCGIM + LAWS + LAPR + RLAIM) - LATSUB
             - (CGSB + LASBHH) - CGNCGA - ECNET - LANCGA
             - (CGOTR + LAOTRHH) - (DICGOP + DILAPR + DIPCOP)
             - EUVAT - GNP4 - CGSUBPR
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('PSCE')
            - (v('CGWS') + v('CGP') + v('RCGIM') + v('LAWS') + v('LAPR') + v('RLAIM'))
            - v('LATSUB')
            - (v('CGSB') + v('LASBHH'))
            - v('CGNCGA') - v('ECNET') - v('LANCGA')
            - (v('CGOTR') + v('LAOTRHH'))
            - (v('DICGOP') + v('DILAPR') + v('DIPCOP'))
            - v('EUVAT') - v('GNP4') - v('CGSUBPR'))


# --------------------------------------------------------------------------- #
# DEP = RCGIM + RLAIM + PCCON
# --------------------------------------------------------------------------- #
def compute_DEP(data, t):
    """DEP = RCGIM + RLAIM + PCCON"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('RCGIM') + v('RLAIM') + v('PCCON')


# --------------------------------------------------------------------------- #
# PSCB = PSCR - PSCE - DEP
# --------------------------------------------------------------------------- #
def compute_PSCB(data, t):
    """PSCB = PSCR - PSCE - DEP"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PSCR') - v('PSCE') - v('DEP')


# --------------------------------------------------------------------------- #
# NPACG = (NPACG(-1) + NPACG(-2) + NPACG(-3) + NPACG(-4)) / 4
# --------------------------------------------------------------------------- #
def compute_NPACG(data, t):
    """NPACG = (NPACG(-1) + NPACG(-2) + NPACG(-3) + NPACG(-4)) / 4"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('NPACG', 1) + v('NPACG', 2) + v('NPACG', 3) + v('NPACG', 4)) / 4


# --------------------------------------------------------------------------- #
# NPALA = (NPALA(-1) + NPALA(-2) + NPALA(-3) + NPALA(-4)) / 4
# --------------------------------------------------------------------------- #
def compute_NPALA(data, t):
    """NPALA = (NPALA(-1) + NPALA(-2) + NPALA(-3) + NPALA(-4)) / 4"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('NPALA', 1) + v('NPALA', 2) + v('NPALA', 3) + v('NPALA', 4)) / 4


# --------------------------------------------------------------------------- #
# PSGI = CGIPS + LAIPS + IPCPS + IBPC + DINVCG + (NPACG + NPALA)
#      + (KCGPSO - KPSCG) + (KLA - KGLAPC - KGLA) + (KPCPS - KPSPC) + ASSETSA
# --------------------------------------------------------------------------- #
def compute_PSGI(data, t):
    """
    PSGI = CGIPS + LAIPS + IPCPS + IBPC + DINVCG + (NPACG + NPALA)
         + (KCGPSO - KPSCG) + (KLA - KGLAPC - KGLA) + (KPCPS - KPSPC) + ASSETSA
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('CGIPS') + v('LAIPS') + v('IPCPS') + v('IBPC') + v('DINVCG')
            + (v('NPACG') + v('NPALA'))
            + (v('KCGPSO') - v('KPSCG'))
            + (v('KLA') - v('KGLAPC') - v('KGLA'))
            + (v('KPCPS') - v('KPSPC'))
            + v('ASSETSA'))


# --------------------------------------------------------------------------- #
# TME = PSCE + DEP + PSNI
# --------------------------------------------------------------------------- #
def compute_TME(data, t):
    """TME = PSCE + DEP + PSNI"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PSCE') + v('DEP') + v('PSNI')


# --------------------------------------------------------------------------- #
# CGNB = (CGWS + CGP) + CGTSUB + CGSB + CGNCGA + CGCGLA + CGOTR + GNP4
#      + EUVAT + DICGOP + (CGIPS + NPACG) + DINVCG + (KCGLA + KCGPC) + KCGPSO
#      - KPSCG - (PUBSTIW + TYPCO) - (PUBSTPD - LAPT)
#      - (OCT + LANNDR) - (INHT + LAEPS + SWISSCAP)
#      - (EMPNIC + EENIC) - CGNDIV - CGINTRA - (RNCG + HHTCG + BLEVY)
# --------------------------------------------------------------------------- #
def compute_CGNB(data, t):
    """
    CGNB = (CGWS + CGP) + CGTSUB + CGSB + CGNCGA + CGCGLA + CGOTR + GNP4
         + EUVAT + DICGOP + (CGIPS + NPACG) + DINVCG + (KCGLA + KCGPC)
         + KCGPSO - KPSCG - (PUBSTIW + TYPCO) - (PUBSTPD - LAPT)
         - (OCT + LANNDR) - (INHT + LAEPS + SWISSCAP)
         - (EMPNIC + EENIC) - CGNDIV - CGINTRA - (RNCG + HHTCG + BLEVY)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return ((v('CGWS') + v('CGP'))
            + v('CGTSUB') + v('CGSB') + v('CGNCGA') + v('CGCGLA')
            + v('CGOTR') + v('GNP4') + v('EUVAT') + v('DICGOP')
            + (v('CGIPS') + v('NPACG'))
            + v('DINVCG')
            + (v('KCGLA') + v('KCGPC'))
            + v('KCGPSO') - v('KPSCG')
            - (v('PUBSTIW') + v('TYPCO'))
            - (v('PUBSTPD') - v('LAPT'))
            - (v('OCT') + v('LANNDR'))
            - (v('INHT') + v('LAEPS') + v('SWISSCAP'))
            - (v('EMPNIC') + v('EENIC'))
            - v('CGNDIV') - v('CGINTRA')
            - (v('RNCG') + v('HHTCG') + v('BLEVY')))


# --------------------------------------------------------------------------- #
# LANB = (LAWS + LAPR) + LATSUB + LASBHH + LANCGA - CGCGLA + LAOTRHH
#       + DILAPR + (LAIPS + NPALA) - KCGLA + (KLA - KGLAPC) - KGLA
#       - LAPT - (CC - LANNDR) - LAINTRA - LANDIV - LARENT - CIL
# --------------------------------------------------------------------------- #
def compute_LANB(data, t):
    """
    LANB = (LAWS + LAPR) + LATSUB + LASBHH + LANCGA - CGCGLA + LAOTRHH
         + DILAPR + (LAIPS + NPALA) - KCGLA + (KLA - KGLAPC) - KGLA
         - LAPT - (CC - LANNDR) - LAINTRA - LANDIV - LARENT - CIL
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return ((v('LAWS') + v('LAPR'))
            + v('LATSUB') + v('LASBHH') + v('LANCGA')
            - v('CGCGLA') + v('LAOTRHH')
            + v('DILAPR')
            + (v('LAIPS') + v('NPALA'))
            - v('KCGLA')
            + (v('KLA') - v('KGLAPC'))
            - v('KGLA')
            - v('LAPT')
            - (v('CC') - v('LANNDR'))
            - v('LAINTRA') - v('LANDIV') - v('LARENT') - v('CIL'))


# --------------------------------------------------------------------------- #
# GGNB = CGNB + LANB
# --------------------------------------------------------------------------- #
def compute_GGNB(data, t):
    """GGNB = CGNB + LANB"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGNB') + v('LANB')


# --------------------------------------------------------------------------- #
# GGNBCY = GGNB
# --------------------------------------------------------------------------- #
def compute_GGNBCY(data, t):
    """GGNBCY = GGNB"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('GGNB')


# --------------------------------------------------------------------------- #
# PCNB = DIPCOP + IPCPS + IBPC - (KCGPC + KGLAPC) + (KPCPS - KPSPC)
#       + TYPCO - OSPC - PCNDIV - PCINTRA - PCRENT
# --------------------------------------------------------------------------- #
def compute_PCNB(data, t):
    """
    PCNB = DIPCOP + IPCPS + IBPC - (KCGPC + KGLAPC) + (KPCPS - KPSPC)
         + TYPCO - OSPC - PCNDIV - PCINTRA - PCRENT
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('DIPCOP') + v('IPCPS') + v('IBPC')
            - (v('KCGPC') + v('KGLAPC'))
            + (v('KPCPS') - v('KPSPC'))
            + v('TYPCO')
            - v('OSPC') - v('PCNDIV') - v('PCINTRA') - v('PCRENT'))


# --------------------------------------------------------------------------- #
# PCNBCY = PCNB
# --------------------------------------------------------------------------- #
def compute_PCNBCY(data, t):
    """PCNBCY = PCNB"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PCNB')


# --------------------------------------------------------------------------- #
# PSNBNSA = -PSCB + PSNI
# --------------------------------------------------------------------------- #
def compute_PSNBNSA(data, t):
    """PSNBNSA = -PSCB + PSNI"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return -v('PSCB') + v('PSNI')


# --------------------------------------------------------------------------- #
# PSNBCY = PSNBNSA
# --------------------------------------------------------------------------- #
def compute_PSNBCY(data, t):
    """PSNBCY = PSNBNSA"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PSNBNSA')


# --------------------------------------------------------------------------- #
# SWAPS = 0
# --------------------------------------------------------------------------- #
def compute_SWAPS(data, t):
    """SWAPS = 0"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 0


# --------------------------------------------------------------------------- #
# TDEF = CGNB + LANB + SWAPS
# --------------------------------------------------------------------------- #
def compute_TDEF(data, t):
    """TDEF = CGNB + LANB + SWAPS"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGNB') + v('LANB') + v('SWAPS')


# --------------------------------------------------------------------------- #
# CGLSFA = (LCGOS + LCGPR) + CGMISP
# --------------------------------------------------------------------------- #
def compute_CGLSFA(data, t):
    """CGLSFA = (LCGOS + LCGPR) + (CGMISP)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('LCGOS') + v('LCGPR')) + v('CGMISP')


# --------------------------------------------------------------------------- #
# PSLSFA = CGLSFA + (LALEND + LAMISE) + (PCLEND + PCMISE)
# --------------------------------------------------------------------------- #
def compute_PSLSFA(data, t):
    """PSLSFA = CGLSFA + (LALEND + LAMISE) + (PCLEND + PCMISE)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('CGLSFA')
            + (v('LALEND') + v('LAMISE'))
            + (v('PCLEND') + v('PCMISE')))


# --------------------------------------------------------------------------- #
# CGACADJ = (EXDUTAC + NICAC + INCTAC) + FCACA + CGACRES + (ILGAC + CONACC)
#          + MFTRAN
# --------------------------------------------------------------------------- #
def compute_CGACADJ(data, t):
    """CGACADJ = (EXDUTAC + NICAC + INCTAC) + FCACA + CGACRES + (ILGAC + CONACC) + MFTRAN"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return ((v('EXDUTAC') + v('NICAC') + v('INCTAC'))
            + v('FCACA') + v('CGACRES')
            + (v('ILGAC') + v('CONACC'))
            + v('MFTRAN'))


# --------------------------------------------------------------------------- #
# PSACADJ = CGACADJ + LAAC + LAMFT + PCAC + PCGILT + MFTPC
# --------------------------------------------------------------------------- #
def compute_PSACADJ(data, t):
    """PSACADJ = CGACADJ + LAAC + LAMFT + PCAC + PCGILT + MFTPC"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('CGACADJ') + v('LAAC') + v('LAMFT')
            + v('PCAC') + v('PCGILT') + v('MFTPC'))


# --------------------------------------------------------------------------- #
# PSFL = CGGILTS + OFLPS + NATSAV + MKTIG
# --------------------------------------------------------------------------- #
def compute_PSFL(data, t):
    """PSFL = CGGILTS + OFLPS + NATSAV + MKTIG"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGGILTS') + v('OFLPS') + v('NATSAV') + v('MKTIG')


# --------------------------------------------------------------------------- #
# PSTA = PSTA(-1) * PIF/PIF(-1)
#      + 0.5*(PSNI + KCGPC + KGLAPC - KLA - KCGPSO) * (1 + GGIDEF/GGIDEF(-1))
# --------------------------------------------------------------------------- #
def compute_PSTA(data, t):
    """
    PSTA = PSTA(-1) * PIF/PIF(-1)
         + 0.5*(PSNI + KCGPC + KGLAPC - KLA - KCGPSO) * (1 + GGIDEF/GGIDEF(-1))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('PSTA', 1) * v('PIF') / v('PIF', 1)
            + 0.5 * (v('PSNI') + v('KCGPC') + v('KGLAPC') - v('KLA') - v('KCGPSO'))
            * (1 + v('GGIDEF') / v('GGIDEF', 1)))


# --------------------------------------------------------------------------- #
# PSNW = PSTA + PSFA - PSFL
# --------------------------------------------------------------------------- #
def compute_PSNW(data, t):
    """PSNW = PSTA + PSFA - PSFL"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PSTA') + v('PSFA') - v('PSFL')


# --------------------------------------------------------------------------- #
# LABRO = LANB + LALEND + LAMISE + LAAC + LAGILT + LAMFT - LCGLA
# --------------------------------------------------------------------------- #
def compute_LABRO(data, t):
    """LABRO = LANB + LALEND + LAMISE + LAAC + LAGILT + LAMFT - LCGLA"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('LANB') + v('LALEND') + v('LAMISE')
            + v('LAAC') + v('LAGILT') + v('LAMFT') - v('LCGLA'))


# --------------------------------------------------------------------------- #
# CGNCR = CGNB + CGLSFA + CGACADJ + LCGLA + LCGPC
# --------------------------------------------------------------------------- #
def compute_CGNCR(data, t):
    """CGNCR = CGNB + CGLSFA + CGACADJ + LCGLA + LCGPC"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGNB') + v('CGLSFA') + v('CGACADJ') + v('LCGLA') + v('LCGPC')


# --------------------------------------------------------------------------- #
# PSNCR = PSNBNSA + PSLSFA + PSACADJ
# --------------------------------------------------------------------------- #
def compute_PSNCR(data, t):
    """PSNCR = PSNBNSA + PSLSFA + PSACADJ"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('PSNBNSA') + v('PSLSFA') + v('PSACADJ')


# --------------------------------------------------------------------------- #
# COIN / COIN(-4) = M0 / M0(-4)
# -> COIN = COIN(-4) * M0 / M0(-4)
# --------------------------------------------------------------------------- #
def compute_COIN(data, t):
    """COIN / COIN(-4) = M0 / M0(-4)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('COIN', 4) * v('M0') / v('M0', 4)


# --------------------------------------------------------------------------- #
# d(PSND) = PSNCR - ILGAC + d(FLEASGG) + d(FLEASPC) + PSNDRES
# --------------------------------------------------------------------------- #
def compute_PSND(data, t):
    """d(PSND) = PSNCR - ILGAC + d(FLEASGG) + d(FLEASPC) + PSNDRES"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_FLEASGG = v('FLEASGG') - v('FLEASGG', 1)
    d_FLEASPC = v('FLEASPC') - v('FLEASPC', 1)

    rhs = v('PSNCR') - v('ILGAC') + d_FLEASGG + d_FLEASPC + v('PSNDRES')

    return v('PSND', 1) + rhs


# --------------------------------------------------------------------------- #
# GGLIQ = CGLIQ + LALIQ
# --------------------------------------------------------------------------- #
def compute_GGLIQ(data, t):
    """GGLIQ = CGLIQ + LALIQ"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('CGLIQ') + v('LALIQ')


# --------------------------------------------------------------------------- #
# d(GGGD) = CGNCR + LABRO - ILGAC + d(SRES) + d(GGLIQ) + GGGDRES
# --------------------------------------------------------------------------- #
def compute_GGGD(data, t):
    """d(GGGD) = CGNCR + LABRO - ILGAC + d(SRES) + d(GGLIQ) + GGGDRES"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    d_SRES = v('SRES') - v('SRES', 1)
    d_GGLIQ = v('GGLIQ') - v('GGLIQ', 1)

    rhs = (v('CGNCR') + v('LABRO') - v('ILGAC')
           + d_SRES + d_GGLIQ + v('GGGDRES'))

    return v('GGGD', 1) + rhs


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('CGSUBP',   compute_CGSUBP,   'identity'),
        ('DEP',      compute_DEP,      'identity'),
        ('PSCB',     compute_PSCB,     'identity'),
        ('NPACG',    compute_NPACG,    'identity'),
        ('NPALA',    compute_NPALA,    'identity'),
        ('PSGI',     compute_PSGI,     'identity'),
        ('TME',      compute_TME,      'identity'),
        ('CGNB',     compute_CGNB,     'identity'),
        ('LANB',     compute_LANB,     'identity'),
        ('GGNB',     compute_GGNB,     'identity'),
        ('GGNBCY',   compute_GGNBCY,   'identity'),
        ('PCNB',     compute_PCNB,     'identity'),
        ('PCNBCY',   compute_PCNBCY,   'identity'),
        ('PSNBNSA',  compute_PSNBNSA,  'identity'),
        ('PSNBCY',   compute_PSNBCY,   'identity'),
        ('SWAPS',    compute_SWAPS,    'identity'),
        ('TDEF',     compute_TDEF,     'identity'),
        ('CGLSFA',   compute_CGLSFA,   'identity'),
        ('PSLSFA',   compute_PSLSFA,   'identity'),
        ('CGACADJ',  compute_CGACADJ,  'identity'),
        ('PSACADJ',  compute_PSACADJ,  'identity'),
        ('PSFL',     compute_PSFL,     'identity'),
        ('PSTA',     compute_PSTA,     'identity'),
        ('PSNW',     compute_PSNW,     'identity'),
        ('LABRO',    compute_LABRO,    'identity'),
        ('CGNCR',    compute_CGNCR,    'identity'),
        ('PSNCR',    compute_PSNCR,    'identity'),
        ('COIN',     compute_COIN,     'ratio'),
        ('PSND',     compute_PSND,     'd'),
        ('GGLIQ',    compute_GGLIQ,    'identity'),
        ('GGGD',     compute_GGGD,     'd'),
    ]
