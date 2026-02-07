"""
Group 15: Income Account
Lines 589-677 of obr_model.txt

Equations:
    WFP      - Wages, fees and profits (identity)
    MI       - Mixed income (ratio)
    EMPSC    - Employers' social contributions (identity)
    FYEMP    - Full year employment income (identity)
    EMPISC   - Employers' insurance social contributions (identity)
    EMPASC   - Employers' actual social contributions (identity)
    EMPISCPP - Employers' ISC pension provision (ratio)
    HHISC    - Household insurance social contributions (ratio)
    HHSB     - Household social benefits (identity)
    OSB      - Other social benefits (ratio)
    SBHH     - Social benefits to households (identity)
    TYWHH    - Taxes on income and wealth, households (identity)
    NMTRHH   - Net miscellaneous transfers, households (identity)
    DIPHHx   - Debt interest payments, households extended (identity)
    DIPHHmf  - Debt interest payments, HH mortgage FISIM (identity)
    DIPHH    - Debt interest payments, households (identity)
    DIRHHx   - Debt interest receipts, HH extended (identity)
    DIRHHf   - Debt interest receipts, HH FISIM (identity)
    DIRICx   - Debt interest receipts, IC extended (identity)
    DIRICf   - Debt interest receipts, IC FISIM (d)
    DIRIC    - Debt interest receipts, IC (d)
    DIPICx   - Debt interest payments, IC extended (identity)
    DIPICf   - Debt interest payments, IC FISIM (d)
    DIPIC    - Debt interest payments, IC (d)
    WYQC     - Withdrawals from quasi-corporates (ratio)
    NDIVHH   - Net dividends, households (log)
    PIRHH    - Property income received, households (identity)
    PIPHH    - Property income paid, households (identity)
    EECPP    - Employers' contributions to pension provision (identity)
    EESC     - Employers' extra-statutory contributions (identity)
    HHDI     - Household disposable income (identity)
    RHHDI    - Real household disposable income (identity)
    EMPCPP   - Employers' contributions to pension provision (ratio)
    NEAHH    - Net equity of HH in pension fund reserves adjustment (identity)
    SVHH     - Household saving (identity)
    SY       - Saving ratio (identity)
    KGHH     - Capital grants to households (identity)
    NAFHH    - Net acquisition of financial assets, households (identity)
    NAFCO    - Net acquisition of financial assets, corporations (identity)
    NAFFC    - Net acquisition of financial assets, financial corps (identity)
    NAFIC    - Net acquisition of financial assets, industrial & commercial (identity)
    SAVCO    - Corporate saving (identity)
"""

import numpy as np
from ..eviews_functions import recode_eq, recode_geq, recode_leq, elem, trend, safe_log


# --------------------------------------------------------------------------- #
# WFP
# --------------------------------------------------------------------------- #
def compute_WFP(data, t):
    """
    WFP = ADJW * PSAVEI * (EMS - ESLFS) + (52/4000) * CGWADJ * ERCG * ECG
          + (52/4000) * LAWADJ * ERLA * ELA

    EViews original (line 589):
    WFP = ADJW * PSAVEI * (EMS - ESLFS) + (52 / 4000) * CGWADJ * ERCG * ECG
          + (52 / 4000) * LAWADJ * ERLA * ELA
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('ADJW') * v('PSAVEI') * (v('EMS') - v('ESLFS'))
            + (52 / 4000) * v('CGWADJ') * v('ERCG') * v('ECG')
            + (52 / 4000) * v('LAWADJ') * v('ERLA') * v('ELA'))


# --------------------------------------------------------------------------- #
# MI
# --------------------------------------------------------------------------- #
def compute_MI(data, t):
    """
    MI / MI(-1) = WFP / WFP(-1)

    EViews original (line 591):
    MI / MI(-1) = WFP / WFP(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('MI', 1) * (v('WFP') / v('WFP', 1))


# --------------------------------------------------------------------------- #
# EMPSC
# --------------------------------------------------------------------------- #
def compute_EMPSC(data, t):
    """
    EMPSC = EMPISC + CGASC + EMPNIC + EMPCPP

    EViews original (line 593):
    EMPSC = EMPISC + CGASC + EMPNIC + EMPCPP
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EMPISC') + v('CGASC') + v('EMPNIC') + v('EMPCPP')


# --------------------------------------------------------------------------- #
# FYEMP
# --------------------------------------------------------------------------- #
def compute_FYEMP(data, t):
    """
    FYEMP = WFP + EMPSC

    EViews original (line 595):
    FYEMP = WFP + EMPSC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('WFP') + v('EMPSC')


# --------------------------------------------------------------------------- #
# EMPISC
# --------------------------------------------------------------------------- #
def compute_EMPISC(data, t):
    """
    EMPISC = HHISC + LASC + CGISC

    EViews original (line 597):
    EMPISC = HHISC + LASC + CGISC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('HHISC') + v('LASC') + v('CGISC')


# --------------------------------------------------------------------------- #
# EMPASC
# --------------------------------------------------------------------------- #
def compute_EMPASC(data, t):
    """
    EMPASC = EMPSC - EMPISC

    EViews original (line 599):
    EMPASC = EMPSC - EMPISC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EMPSC') - v('EMPISC')


# --------------------------------------------------------------------------- #
# EMPISCPP
# --------------------------------------------------------------------------- #
def compute_EMPISCPP(data, t):
    """
    EMPISCPP / EMPISCPP(-1) = EMPISC / EMPISC(-1)

    EViews original (line 601):
    EMPISCPP / EMPISCPP(-1) = EMPISC / EMPISC(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EMPISCPP', 1) * (v('EMPISC') / v('EMPISC', 1))


# --------------------------------------------------------------------------- #
# HHISC
# --------------------------------------------------------------------------- #
def compute_HHISC(data, t):
    """
    HHISC / HHISC(-1) = WFP / WFP(-1)

    EViews original (line 603):
    HHISC / HHISC(-1) = WFP / WFP(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('HHISC', 1) * (v('WFP') / v('WFP', 1))


# --------------------------------------------------------------------------- #
# HHSB
# --------------------------------------------------------------------------- #
def compute_HHSB(data, t):
    """
    HHSB = 2 * HHISC

    EViews original (line 605):
    HHSB = 2 * HHISC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 2 * v('HHISC')


# --------------------------------------------------------------------------- #
# OSB
# --------------------------------------------------------------------------- #
def compute_OSB(data, t):
    """
    OSB / OSB(-1) = PCE/PCE(-1) * GAD3/GAD3(-1)

    EViews original (line 607):
    OSB / OSB(-1) = PCE / PCE(-1) * GAD3 / GAD3(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('OSB', 1) * (v('PCE') / v('PCE', 1)) * (v('GAD3') / v('GAD3', 1))


# --------------------------------------------------------------------------- #
# SBHH
# --------------------------------------------------------------------------- #
def compute_SBHH(data, t):
    """
    SBHH = EMPISC + OSB + (HHSB - HHISC - EMPISCPP) + CGSB + LASBHH
           + EESCLA + EESCCG + CGASC - BENAB

    EViews original (line 609):
    SBHH = EMPISC + OSB + (HHSB - HHISC - EMPISCPP) + CGSB + LASBHH
           + EESCLA + EESCCG + CGASC - BENAB
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('EMPISC') + v('OSB')
            + (v('HHSB') - v('HHISC') - v('EMPISCPP'))
            + v('CGSB') + v('LASBHH')
            + v('EESCLA') + v('EESCCG') + v('CGASC') - v('BENAB'))


# --------------------------------------------------------------------------- #
# TYWHH
# --------------------------------------------------------------------------- #
def compute_TYWHH(data, t):
    """
    TYWHH = TYEM + TSEOP + CC + CGT + OCT - NPISHTC

    EViews original (line 612):
    TYWHH = TYEM + TSEOP + CC + CGT + OCT - NPISHTC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('TYEM') + v('TSEOP') + v('CC') + v('CGT') + v('OCT') - v('NPISHTC')


# --------------------------------------------------------------------------- #
# NMTRHH
# --------------------------------------------------------------------------- #
def compute_NMTRHH(data, t):
    """
    NMTRHH = LAOTRHH + (CGOTR - HHTCG) + (HHTFA - HHTA) + (EUSF) + 100

    EViews original (line 615):
    NMTRHH = LAOTRHH + (CGOTR - HHTCG) + (HHTFA - HHTA) + (EUSF) + 100
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('LAOTRHH')
            + (v('CGOTR') - v('HHTCG'))
            + (v('HHTFA') - v('HHTA'))
            + v('EUSF') + 100)


# --------------------------------------------------------------------------- #
# DIPHHx
# --------------------------------------------------------------------------- #
def compute_DIPHHx(data, t):
    """
    DIPHHx = DIPHH + DIPHHmf + DIPHHuf

    EViews original (line 617):
    DIPHHx = DIPHH + DIPHHmf + DIPHHuf
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DIPHH') + v('DIPHHmf') + v('DIPHHuf')


# --------------------------------------------------------------------------- #
# DIPHHmf
# --------------------------------------------------------------------------- #
def compute_DIPHHmf(data, t):
    """
    DIPHHmf = LHP(-1) * ((1 + (RMORT - R) / 100)^0.25 - 1)

    EViews original (line 619):
    DIPHHmf = LHP(-1) * ((1 + (RMORT - R) / 100)^0.25 - 1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('LHP', 1) * ((1 + (v('RMORT') - v('R')) / 100) ** 0.25 - 1)


# --------------------------------------------------------------------------- #
# DIPHH
# --------------------------------------------------------------------------- #
def compute_DIPHH(data, t):
    """
    DIPHH = (LHP(-1) + OLPE(-1)) * ((1 + (0.9 * R + 0.2) / 100)^0.25 - 1)

    EViews original (line 623):
    DIPHH = (LHP(-1) + OLPE(-1)) * ((1 + (0.9 * R + 0.2) / 100)^0.25 - 1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return ((v('LHP', 1) + v('OLPE', 1))
            * ((1 + (0.9 * v('R') + 0.2) / 100) ** 0.25 - 1))


# --------------------------------------------------------------------------- #
# DIRHHx
# --------------------------------------------------------------------------- #
def compute_DIRHHx(data, t):
    """
    DIRHHx = DIRHH - DIRHHf

    EViews original (line 625):
    DIRHHx = DIRHH - DIRHHf
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DIRHH') - v('DIRHHf')


# --------------------------------------------------------------------------- #
# DIRHHf
# --------------------------------------------------------------------------- #
def compute_DIRHHf(data, t):
    """
    DIRHHf = -(0.75 * DEPHH(-1) * ((1 + (RDEP - R) / 100)^0.25 - 1))

    EViews original (line 627):
    DIRHHf = -(0.75 * DEPHH(-1) * ((1 + (RDEP - R) / 100)^.25 - 1))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return -(0.75 * v('DEPHH', 1) * ((1 + (v('RDEP') - v('R')) / 100) ** 0.25 - 1))


# --------------------------------------------------------------------------- #
# DIRICx
# --------------------------------------------------------------------------- #
def compute_DIRICx(data, t):
    """
    DIRICx = DIRIC - DIRICf

    EViews original (line 629):
    DIRICx = DIRIC - DIRICf
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DIRIC') - v('DIRICf')


# --------------------------------------------------------------------------- #
# DIRICf
# --------------------------------------------------------------------------- #
def compute_DIRICf(data, t):
    """
    d(DIRICf) = -((2.75) * M4IC(-1) * (((1 + (0.9*R - 0.2 - R)/100)^0.25) - 1))
                + ((2.75) * M4IC(-2) * (((1 + (0.9*R(-1) - 0.2 - R(-1))/100)^0.25) - 1))

    EViews original (line 631):
    d(DIRICf) = -((2.75) * M4IC(-1) * (((1 + (0.9 * R - 0.2 - R) / 100)^0.25) - 1))
                + ((2.75) * M4IC(-2) * (((1 + (0.9 * R(-1) - 0.2 - R(-1)) / 100)^0.25) - 1))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    # 0.9*R - 0.2 - R = -0.1*R - 0.2
    term_t = -2.75 * v('M4IC', 1) * (((1 + (0.9 * v('R') - 0.2 - v('R')) / 100) ** 0.25) - 1)
    term_t1 = 2.75 * v('M4IC', 2) * (((1 + (0.9 * v('R', 1) - 0.2 - v('R', 1)) / 100) ** 0.25) - 1)

    rhs = term_t + term_t1

    return v('DIRICf', 1) + rhs


# --------------------------------------------------------------------------- #
# DIRIC
# --------------------------------------------------------------------------- #
def compute_DIRIC(data, t):
    """
    d(DIRIC) = (M4IC(-1) * (((1+R/100)^0.25) - 1)
                - M4IC(-2) * (((1+R(-1)/100)^0.25) - 1)) * 1.3
               + (M4IC(-1) * (((1+ROCB/100)^0.25) - 1)
                  - M4IC(-2) * (((1+ROCB(-1)/100)^0.25) - 1)) * 0.6

    EViews original (line 633):
    d(DIRIC) = (M4IC(-1) * (((1 + R / 100)^0.25) - 1) - M4IC(-2) * (((1 + R(-1) / 100)^0.25) - 1)) * 1.3
               + (M4IC(-1) * (((1 + ROCB / 100)^0.25) - 1) - M4IC(-2) * (((1 + ROCB(-1) / 100)^0.25) - 1)) * 0.6
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    part1 = (v('M4IC', 1) * (((1 + v('R') / 100) ** 0.25) - 1)
             - v('M4IC', 2) * (((1 + v('R', 1) / 100) ** 0.25) - 1)) * 1.3

    part2 = (v('M4IC', 1) * (((1 + v('ROCB') / 100) ** 0.25) - 1)
             - v('M4IC', 2) * (((1 + v('ROCB', 1) / 100) ** 0.25) - 1)) * 0.6

    rhs = part1 + part2

    return v('DIRIC', 1) + rhs


# --------------------------------------------------------------------------- #
# DIPICx
# --------------------------------------------------------------------------- #
def compute_DIPICx(data, t):
    """
    DIPICx = DIPIC + DIPICf

    EViews original (line 635):
    DIPICx = DIPIC + DIPICf
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DIPIC') + v('DIPICf')


# --------------------------------------------------------------------------- #
# DIPICf
# --------------------------------------------------------------------------- #
def compute_DIPICf(data, t):
    """
    d(DIPICf) = STLIC * (((1 + (RIC - R)/100)^0.25) - 1)
                + FXLIC * (((1 + 2.9/100)^0.25) - 1)
                - STLIC(-1) * (((1 + (RIC(-1) - R(-1))/100)^0.25) - 1)
                + FXLIC(-1) * (((1 + 2.9/100)^0.25) - 1)

    EViews original (line 637):
    d(DIPICf) = STLIC * (((1 + (RIC - R) / 100)^0.25) - 1) + FXLIC * (((1 + 2.9 / 100)^0.25) - 1)
                - STLIC(-1) * (((1 + (RIC(-1) - R(-1)) / 100)^0.25) - 1)
                + FXLIC(-1) * (((1 + 2.9 / 100)^0.25) - 1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    fxlic_const = ((1 + 2.9 / 100) ** 0.25) - 1

    rhs = (v('STLIC') * (((1 + (v('RIC') - v('R')) / 100) ** 0.25) - 1)
           + v('FXLIC') * fxlic_const
           - v('STLIC', 1) * (((1 + (v('RIC', 1) - v('R', 1)) / 100) ** 0.25) - 1)
           + v('FXLIC', 1) * fxlic_const)

    return v('DIPICf', 1) + rhs


# --------------------------------------------------------------------------- #
# DIPIC
# --------------------------------------------------------------------------- #
def compute_DIPIC(data, t):
    """
    d(DIPIC) = (STLIC(-1) * (((1+R/100)^0.25) - 1)
                - STLIC(-2) * (((1+R(-1)/100)^0.25) - 1))
               + (FXLIC(-1) * (((1+ROCB/100)^0.25) - 1)
                  - FXLIC(-2) * (((1+ROCB(-1)/100)^0.25) - 1))
               + (BLIC(-1) * (((1+RL/100)^0.25) - 1)
                  - BLIC(-2) * (((1+RL(-1)/100)^0.25) - 1))

    EViews original (line 639):
    d(DIPIC) = (STLIC(-1) * (((1 + R / 100)^0.25) - 1) - STLIC(-2) * (((1 + R(-1) / 100)^0.25) - 1))
               + (FXLIC(-1) * (((1 + ROCB / 100)^0.25) - 1) - FXLIC(-2) * (((1 + ROCB(-1) / 100)^0.25) - 1))
               + (BLIC(-1) * (((1 + RL / 100)^0.25) - 1) - BLIC(-2) * (((1 + RL(-1) / 100)^0.25) - 1))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    part1 = (v('STLIC', 1) * (((1 + v('R') / 100) ** 0.25) - 1)
             - v('STLIC', 2) * (((1 + v('R', 1) / 100) ** 0.25) - 1))

    part2 = (v('FXLIC', 1) * (((1 + v('ROCB') / 100) ** 0.25) - 1)
             - v('FXLIC', 2) * (((1 + v('ROCB', 1) / 100) ** 0.25) - 1))

    part3 = (v('BLIC', 1) * (((1 + v('RL') / 100) ** 0.25) - 1)
             - v('BLIC', 2) * (((1 + v('RL', 1) / 100) ** 0.25) - 1))

    rhs = part1 + part2 + part3

    return v('DIPIC', 1) + rhs


# --------------------------------------------------------------------------- #
# WYQC
# --------------------------------------------------------------------------- #
def compute_WYQC(data, t):
    """
    WYQC / WYQC(-1) = FYCPR / FYCPR(-1)

    EViews original (line 641):
    WYQC / WYQC(-1) = FYCPR / FYCPR(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('WYQC', 1) * (v('FYCPR') / v('FYCPR', 1))


# --------------------------------------------------------------------------- #
# NDIVHH
# --------------------------------------------------------------------------- #
def compute_NDIVHH(data, t):
    """
    log(NDIVHH) = -8.605599 + 0.8092696 * log(FYCPR(-4)) + 0.6597959 * log(CORP)

    EViews original (line 643):
    log(NDIVHH) = -8.605599 + 0.8092696 * log(FYCPR(-4)) + 0.6597959 * log(CORP)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return np.exp(-8.605599
                  + 0.8092696 * safe_log(v('FYCPR', 4))
                  + 0.6597959 * safe_log(v('CORP')))


# --------------------------------------------------------------------------- #
# PIRHH
# --------------------------------------------------------------------------- #
def compute_PIRHH(data, t):
    """
    PIRHH = NDIVHH + APIIH + DIRHH + WYQC

    EViews original (line 645):
    PIRHH = NDIVHH + APIIH + DIRHH + WYQC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('NDIVHH') + v('APIIH') + v('DIRHH') + v('WYQC')


# --------------------------------------------------------------------------- #
# PIPHH
# --------------------------------------------------------------------------- #
def compute_PIPHH(data, t):
    """
    PIPHH = DIPHH

    EViews original (line 647):
    PIPHH = DIPHH
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('DIPHH')


# --------------------------------------------------------------------------- #
# EECPP
# --------------------------------------------------------------------------- #
def compute_EECPP(data, t):
    """
    EECPP = ((1 + (RL/100))^0.25 - 1) * (PIHH(-1) * 0.729)
            + ((1 + 0.05)^0.25 - 1) * (PIHH(-1) * 0.271)

    EViews original (line 649):
    EECPP = ((1 + (RL / 100))^0.25 - 1) * (PIHH(-1) * 0.729)
            + ((1 + 0.05)^0.25 - 1) * (PIHH(-1) * 0.271)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (((1 + v('RL') / 100) ** 0.25 - 1) * (v('PIHH', 1) * 0.729)
            + ((1 + 0.05) ** 0.25 - 1) * (v('PIHH', 1) * 0.271))


# --------------------------------------------------------------------------- #
# EESC
# --------------------------------------------------------------------------- #
def compute_EESC(data, t):
    """
    EESC = EESCLA + EENIC + EECPP + EESCCG

    EViews original (line 651):
    EESC = EESCLA + EENIC + EECPP + EESCCG
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EESCLA') + v('EENIC') + v('EECPP') + v('EESCCG')


# --------------------------------------------------------------------------- #
# HHDI
# --------------------------------------------------------------------------- #
def compute_HHDI(data, t):
    """
    HHDI = MI + FYEMP - EMPSC - EESC - TYWHH + NMTRHH + SBHH
           + (PIRHH - PIPHH + FSMADJ) - HHSB + HHISC
           + (EECOMPC - EECOMPD) + OSHH

    EViews original (line 654):
    HHDI = MI + FYEMP - EMPSC - EESC - TYWHH + NMTRHH + SBHH
           + (PIRHH - PIPHH + FSMADJ) - HHSB + HHISC
           + (EECOMPC - EECOMPD) + OSHH
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('MI') + v('FYEMP') - v('EMPSC') - v('EESC')
            - v('TYWHH') + v('NMTRHH') + v('SBHH')
            + (v('PIRHH') - v('PIPHH') + v('FSMADJ'))
            - v('HHSB') + v('HHISC')
            + (v('EECOMPC') - v('EECOMPD'))
            + v('OSHH'))


# --------------------------------------------------------------------------- #
# RHHDI
# --------------------------------------------------------------------------- #
def compute_RHHDI(data, t):
    """
    RHHDI = 100 * HHDI / PCE

    EViews original (line 656):
    RHHDI = 100 * HHDI / PCE
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 100 * v('HHDI') / v('PCE')


# --------------------------------------------------------------------------- #
# EMPCPP
# --------------------------------------------------------------------------- #
def compute_EMPCPP(data, t):
    """
    EMPCPP / EMPCPP(-1) = WFP / WFP(-1)

    EViews original (line 658):
    EMPCPP / EMPCPP(-1) = WFP / WFP(-1)
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EMPCPP', 1) * (v('WFP') / v('WFP', 1))


# --------------------------------------------------------------------------- #
# NEAHH
# --------------------------------------------------------------------------- #
def compute_NEAHH(data, t):
    """
    NEAHH = EMPCPP + EECPP + EMPISCPP - OSB

    EViews original (line 660):
    NEAHH = EMPCPP + EECPP + EMPISCPP - OSB
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('EMPCPP') + v('EECPP') + v('EMPISCPP') - v('OSB')


# --------------------------------------------------------------------------- #
# SVHH
# --------------------------------------------------------------------------- #
def compute_SVHH(data, t):
    """
    SVHH = HHDI + NEAHH - CONSPS

    EViews original (line 662):
    SVHH = HHDI + NEAHH - CONSPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('HHDI') + v('NEAHH') - v('CONSPS')


# --------------------------------------------------------------------------- #
# SY
# --------------------------------------------------------------------------- #
def compute_SY(data, t):
    """
    SY = 100 * (SVHH / (NEAHH + HHDI))

    EViews original (line 664):
    SY = 100 * (SVHH / (NEAHH + HHDI))
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return 100 * (v('SVHH') / (v('NEAHH') + v('HHDI')))


# --------------------------------------------------------------------------- #
# KGHH
# --------------------------------------------------------------------------- #
def compute_KGHH(data, t):
    """
    KGHH = -INHT + 0.95 * KLA + 0.55 * KCGPSO + 0.4 * EUKT

    EViews original (line 666):
    KGHH = -INHT + 0.95 * KLA + 0.55 * KCGPSO + 0.4 * EUKT
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return -v('INHT') + 0.95 * v('KLA') + 0.55 * v('KCGPSO') + 0.4 * v('EUKT')


# --------------------------------------------------------------------------- #
# NAFHH
# --------------------------------------------------------------------------- #
def compute_NAFHH(data, t):
    """
    NAFHH = SVHH + KGHH - DINVHH - VALHH - NPAHH - IHHPS

    EViews original (line 668):
    NAFHH = SVHH + KGHH - DINVHH - VALHH - NPAHH - IHHPS
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('SVHH') + v('KGHH') - v('DINVHH') - v('VALHH') - v('NPAHH') - v('IHHPS')


# --------------------------------------------------------------------------- #
# NAFCO
# --------------------------------------------------------------------------- #
def compute_NAFCO(data, t):
    """
    NAFCO = -NAFHH + CB + EUKT - CGKTA - OPSKTA + NPAA + SDEPS - SDI + PSNBCY

    EViews original (line 670):
    NAFCO = -NAFHH + CB + EUKT - CGKTA - OPSKTA + NPAA + SDEPS - SDI + PSNBCY
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (-v('NAFHH') + v('CB') + v('EUKT')
            - v('CGKTA') - v('OPSKTA') + v('NPAA')
            + v('SDEPS') - v('SDI') + v('PSNBCY'))


# --------------------------------------------------------------------------- #
# NAFFC
# --------------------------------------------------------------------------- #
def compute_NAFFC(data, t):
    """
    NAFFC = -12012 + FISIMPS - NEAHH - BLEVY

    EViews original (line 672):
    NAFFC = -12012 + FISIMPS - NEAHH - BLEVY
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return -12012 + v('FISIMPS') - v('NEAHH') - v('BLEVY')


# --------------------------------------------------------------------------- #
# NAFIC
# --------------------------------------------------------------------------- #
def compute_NAFIC(data, t):
    """
    NAFIC = NAFCO - NAFFC

    EViews original (line 674):
    NAFIC = NAFCO - NAFFC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return v('NAFCO') - v('NAFFC')


# --------------------------------------------------------------------------- #
# SAVCO
# --------------------------------------------------------------------------- #
def compute_SAVCO(data, t):
    """
    SAVCO = NAFCO + KGHH - DINVHH + DINVPS - DINVCG + VALPS - VALHH
            - NPAHH + IFPS - IHHPS - NPACG - CGIPS - KLA - KCGPSO
            - LAIPS - NPALA + INHT + KGLA - EUKT + CGKTA + OPSKTA
            - NPAA - IPCPS - IBPC

    EViews original (line 676):
    SAVCO = NAFCO + KGHH - DINVHH + DINVPS - DINVCG + VALPS - VALHH
            - NPAHH + IFPS - IHHPS - NPACG - CGIPS - KLA - KCGPSO
            - LAIPS - NPALA + INHT + KGLA - EUKT + CGKTA + OPSKTA
            - NPAA - IPCPS - IBPC
    """
    v = lambda var, lag=0: data.at[t - lag, var] if lag > 0 else data.at[t, var]

    return (v('NAFCO')
            + v('KGHH') - v('DINVHH') + v('DINVPS') - v('DINVCG')
            + v('VALPS') - v('VALHH') - v('NPAHH')
            + v('IFPS') - v('IHHPS')
            - v('NPACG') - v('CGIPS') - v('KLA') - v('KCGPSO')
            - v('LAIPS') - v('NPALA')
            + v('INHT') + v('KGLA')
            - v('EUKT') + v('CGKTA') + v('OPSKTA')
            - v('NPAA') - v('IPCPS') - v('IBPC'))


# --------------------------------------------------------------------------- #
# get_equations
# --------------------------------------------------------------------------- #
def get_equations():
    """Return list of (variable_name, compute_function, equation_type) tuples."""
    return [
        ('WFP',      compute_WFP,      'identity'),
        ('MI',       compute_MI,       'ratio'),
        ('EMPSC',    compute_EMPSC,    'identity'),
        ('FYEMP',    compute_FYEMP,    'identity'),
        ('EMPISC',   compute_EMPISC,   'identity'),
        ('EMPASC',   compute_EMPASC,   'identity'),
        ('EMPISCPP', compute_EMPISCPP, 'ratio'),
        ('HHISC',    compute_HHISC,    'ratio'),
        ('HHSB',     compute_HHSB,     'identity'),
        ('OSB',      compute_OSB,      'ratio'),
        ('SBHH',     compute_SBHH,     'identity'),
        ('TYWHH',    compute_TYWHH,    'identity'),
        ('NMTRHH',   compute_NMTRHH,   'identity'),
        ('DIPHHx',   compute_DIPHHx,   'identity'),
        ('DIPHHmf',  compute_DIPHHmf,  'identity'),
        ('DIPHH',    compute_DIPHH,    'identity'),
        ('DIRHHx',   compute_DIRHHx,   'identity'),
        ('DIRHHf',   compute_DIRHHf,   'identity'),
        ('DIRICx',   compute_DIRICx,   'identity'),
        ('DIRICf',   compute_DIRICf,   'd'),
        ('DIRIC',    compute_DIRIC,    'd'),
        ('DIPICx',   compute_DIPICx,   'identity'),
        ('DIPICf',   compute_DIPICf,   'd'),
        ('DIPIC',    compute_DIPIC,    'd'),
        ('WYQC',     compute_WYQC,     'ratio'),
        ('NDIVHH',   compute_NDIVHH,   'log'),
        ('PIRHH',    compute_PIRHH,    'identity'),
        ('PIPHH',    compute_PIPHH,    'identity'),
        ('EECPP',    compute_EECPP,    'identity'),
        ('EESC',     compute_EESC,     'identity'),
        ('HHDI',     compute_HHDI,     'identity'),
        ('RHHDI',    compute_RHHDI,    'identity'),
        ('EMPCPP',   compute_EMPCPP,   'ratio'),
        ('NEAHH',    compute_NEAHH,    'identity'),
        ('SVHH',     compute_SVHH,     'identity'),
        ('SY',       compute_SY,       'identity'),
        ('KGHH',     compute_KGHH,     'identity'),
        ('NAFHH',    compute_NAFHH,    'identity'),
        ('NAFCO',    compute_NAFCO,    'identity'),
        ('NAFFC',    compute_NAFFC,    'identity'),
        ('NAFIC',    compute_NAFIC,    'identity'),
        ('SAVCO',    compute_SAVCO,    'identity'),
    ]
