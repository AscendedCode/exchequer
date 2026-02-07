"""
Group 4: The Labour Market
OBR macroeconomic model equations (lines 139-178).
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# ---------------------------------------------------------------------------
# ECG / ECG(-1) = EGG / EGG(-1)
# ---------------------------------------------------------------------------
def compute_ECG(data, t):
    """ECG / ECG(-1) = EGG / EGG(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    rhs = v('EGG') / v('EGG', 1)
    return v('ECG', 1) * rhs


# ---------------------------------------------------------------------------
# ELA / ELA(-1) = EGG / EGG(-1)
# ---------------------------------------------------------------------------
def compute_ELA(data, t):
    """ELA / ELA(-1) = EGG / EGG(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    rhs = v('EGG') / v('EGG', 1)
    return v('ELA', 1) * rhs


# ---------------------------------------------------------------------------
# dlog(EPS) = log((ET - ECG - ELA) / (ET(-1) - ECG(-1) - ELA(-1)))
# ---------------------------------------------------------------------------
def compute_EPS(data, t):
    """dlog(EPS) = log((ET - ECG - ELA) / (ET(-1) - ECG(-1) - ELA(-1)))"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    rhs = safe_log(
        (v('ET') - v('ECG') - v('ELA'))
        / (v('ET', 1) - v('ECG', 1) - v('ELA', 1))
    )
    return v('EPS', 1) * np.exp(rhs)


# ---------------------------------------------------------------------------
# dlog(EMS) = -0.0113474 + 0.4369834*dlog(EMS(-1))
#   + 0.1932386*dlog(EMS(-2)) + 0.1713792*dlog(MSGVA(-1))
#   - 0.0062207*(log(EMS(-1)/MSGVA(-1)) + 0.4*(log(PSAVEI(-1)/PMSGVA(-1))))
#   - 0.0103188*@recode(@date = @dateval("2010:04"), 1, 0)
# ---------------------------------------------------------------------------
def compute_EMS(data, t):
    """dlog(EMS) = -0.0113474 + 0.4369834*dlog(EMS(-1)) + 0.1932386*dlog(EMS(-2))
    + 0.1713792*dlog(MSGVA(-1))
    - 0.0062207*(log(EMS(-1)/MSGVA(-1)) + 0.4*(log(PSAVEI(-1)/PMSGVA(-1))))
    - 0.0103188*@recode(@date = @dateval("2010:04"), 1, 0)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    dlog_EMS_1 = safe_log(v('EMS', 1)) - safe_log(v('EMS', 2))
    dlog_EMS_2 = safe_log(v('EMS', 2)) - safe_log(v('EMS', 3))
    dlog_MSGVA_1 = safe_log(v('MSGVA', 1)) - safe_log(v('MSGVA', 2))

    ecm = (safe_log(v('EMS', 1) / v('MSGVA', 1))
           + 0.4 * safe_log(v('PSAVEI', 1) / v('PMSGVA', 1)))

    dummy_2010Q4 = recode_eq(t, "2010:04")

    rhs = (-0.0113474
           + 0.4369834 * dlog_EMS_1
           + 0.1932386 * dlog_EMS_2
           + 0.1713792 * dlog_MSGVA_1
           - 0.0062207 * ecm
           - 0.0103188 * dummy_2010Q4)

    return v('EMS', 1) * np.exp(rhs)


# ---------------------------------------------------------------------------
# ET = ET(-1) * ETLFS / ETLFS(-1)
#   (written in the model as: ET / ET(-1) = ETLFS / ETLFS(-1))
# ---------------------------------------------------------------------------
def compute_ET(data, t):
    """ET / ET(-1) = ETLFS / ETLFS(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    rhs = v('ETLFS') / v('ETLFS', 1)
    return v('ET', 1) * rhs


# ---------------------------------------------------------------------------
# WRGTP / WRGTP(-1) = ET / ET(-1)
# ---------------------------------------------------------------------------
def compute_WRGTP(data, t):
    """WRGTP / WRGTP(-1) = ET / ET(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    rhs = v('ET') / v('ET', 1)
    return v('WRGTP', 1) * rhs


# ---------------------------------------------------------------------------
# WFJ = ET + WRGTP
# ---------------------------------------------------------------------------
def compute_WFJ(data, t):
    """WFJ = ET + WRGTP"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return v('ET') + v('WRGTP')


# ---------------------------------------------------------------------------
# ETLFS = 1000 * (HWA / AVH)
# ---------------------------------------------------------------------------
def compute_ETLFS(data, t):
    """ETLFS = 1000 * (HWA / AVH)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return 1000.0 * (v('HWA') / v('AVH'))


# ---------------------------------------------------------------------------
# ES / ES(-1) = ET / ET(-1)
# ---------------------------------------------------------------------------
def compute_ES(data, t):
    """ES / ES(-1) = ET / ET(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    rhs = v('ET') / v('ET', 1)
    return v('ES', 1) * rhs


# ---------------------------------------------------------------------------
# ESLFS / ESLFS(-1) = ES / ES(-1)
# ---------------------------------------------------------------------------
def compute_ESLFS(data, t):
    """ESLFS / ESLFS(-1) = ES / ES(-1)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    rhs = v('ES') / v('ES', 1)
    return v('ESLFS', 1) * rhs


# ---------------------------------------------------------------------------
# GAD = GAD1 + GAD2 + GAD3
# ---------------------------------------------------------------------------
def compute_GAD(data, t):
    """GAD = GAD1 + GAD2 + GAD3"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return v('GAD1') + v('GAD2') + v('GAD3')


# ---------------------------------------------------------------------------
# POP16 / POP16(-1) = (GAD2 + GAD3) / (GAD2(-1) + GAD3(-1))
# ---------------------------------------------------------------------------
def compute_POP16(data, t):
    """POP16 / POP16(-1) = (GAD2 + GAD3) / (GAD2(-1) + GAD3(-1))"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    rhs = (v('GAD2') + v('GAD3')) / (v('GAD2', 1) + v('GAD3', 1))
    return v('POP16', 1) * rhs


# ---------------------------------------------------------------------------
# ULFS = ((POP16 * PART16 / 100) - ETLFS)
# ---------------------------------------------------------------------------
def compute_ULFS(data, t):
    """ULFS = ((POP16 * PART16 / 100) - ETLFS)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return (v('POP16') * v('PART16') / 100.0) - v('ETLFS')


# ---------------------------------------------------------------------------
# LFSUR = 100 * ULFS / (ETLFS + ULFS)
# ---------------------------------------------------------------------------
def compute_LFSUR(data, t):
    """LFSUR = 100 * ULFS / (ETLFS + ULFS)"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return 100.0 * v('ULFS') / (v('ETLFS') + v('ULFS'))


# ---------------------------------------------------------------------------
# @IDENTITY PRODH = GDPM / HWA
# ---------------------------------------------------------------------------
def compute_PRODH(data, t):
    """@IDENTITY PRODH = GDPM / HWA"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return v('GDPM') / v('HWA')


# ---------------------------------------------------------------------------
# PART16 = 100 * (ULFS + ETLFS) / POP16
# ---------------------------------------------------------------------------
def compute_PART16(data, t):
    """PART16 = 100 * (ULFS + ETLFS) / POP16"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return 100.0 * (v('ULFS') + v('ETLFS')) / v('POP16')


# ---------------------------------------------------------------------------
# ER = 100 * ETLFS / POP16
# ---------------------------------------------------------------------------
def compute_ER(data, t):
    """ER = 100 * ETLFS / POP16"""
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]
    return 100.0 * v('ETLFS') / v('POP16')


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('ECG',    compute_ECG,    'ratio'),
        ('ELA',    compute_ELA,    'ratio'),
        ('EPS',    compute_EPS,    'dlog'),
        ('EMS',    compute_EMS,    'dlog'),
        ('ET',     compute_ET,     'ratio'),
        ('WRGTP',  compute_WRGTP,  'ratio'),
        ('WFJ',    compute_WFJ,    'identity'),
        ('ETLFS',  compute_ETLFS,  'identity'),
        ('ES',     compute_ES,     'ratio'),
        ('ESLFS',  compute_ESLFS,  'ratio'),
        ('GAD',    compute_GAD,    'identity'),
        ('POP16',  compute_POP16,  'ratio'),
        ('ULFS',   compute_ULFS,   'identity'),
        ('LFSUR',  compute_LFSUR,  'identity'),
        ('PRODH',  compute_PRODH,  'identity'),
        ('PART16', compute_PART16, 'identity'),
        ('ER',     compute_ER,     'identity'),
    ]
