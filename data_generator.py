"""
Synthetic data generator for the OBR macroeconomic model.

Generates plausible placeholder data for all ~600 variables so the model
structure can be tested. Values are calibrated to approximate UK magnitudes
but are NOT real data.
"""

import warnings
import numpy as np
import pandas as pd
from . import config

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def _ar1(n, mean, persistence, vol, seed=None):
    """Generate an AR(1) process."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = mean
    for i in range(1, n):
        x[i] = mean * (1 - persistence) + persistence * x[i - 1] + vol * rng.standard_normal()
    return x


def _gbm(n, start, drift_q, vol_q, seed=None):
    """Generate geometric Brownian motion (quarterly)."""
    rng = np.random.default_rng(seed)
    log_returns = drift_q + vol_q * rng.standard_normal(n)
    log_returns[0] = np.log(start)
    return np.exp(np.cumsum(log_returns))


def _trend(n, start, growth_q):
    """Simple exponential trend."""
    return start * (1 + growth_q) ** np.arange(n)


def generate_synthetic_data():
    """Generate a full DataFrame of synthetic data for all model variables."""
    periods = pd.period_range(config.HISTORY_START, config.FORECAST_END, freq='Q')
    n = len(periods)
    data = pd.DataFrame(index=periods, dtype=float)
    quarters = np.arange(n)
    rng = np.random.default_rng(42)

    # ===================================================================
    # EXOGENOUS VARIABLES
    # ===================================================================

    # Interest rates (percent)
    data['R'] = _ar1(n, 4.0, 0.95, 0.3, seed=1)
    data['RL'] = data['R'] + 0.8 + rng.standard_normal(n) * 0.1
    data['RMORT'] = data['R'] + 1.8 + rng.standard_normal(n) * 0.1
    data['RDEP'] = np.maximum(data['R'] - 1.2, 0.1)
    data['ROCB'] = data['R'] + 0.3 + rng.standard_normal(n) * 0.05
    data['ROLT'] = data['RL'] + 0.2
    data['DISCO'] = data['R'] / 100

    # Exchange rates
    data['RX'] = _gbm(n, 1.0, 0.0, 0.015, seed=2)
    data['RXD'] = data['RX'] * 0.85

    # Oil price (USD/barrel)
    data['PBRENT'] = _gbm(n, 20.0, 0.005, 0.06, seed=3)

    # World variables
    data['WPG'] = _trend(n, 50.0, 0.005) * (1 + rng.standard_normal(n) * 0.01)
    data['MAJGDP'] = _trend(n, 200000.0, 0.006) * (1 + rng.standard_normal(n) * 0.005)
    data['WEQPR'] = _trend(n, 50.0, 0.01) * (1 + rng.standard_normal(n) * 0.03)

    # Population (thousands)
    data['POP16'] = _trend(n, 40000.0, 0.001)
    data['POPAL'] = data['POP16'] * 1.22
    data['GAD1'] = data['POP16'] * 0.18
    data['GAD2'] = data['POP16'] * 0.62
    data['GAD3'] = data['POP16'] * 0.20

    # Labour force parameters
    data['PART16'] = 63.0 + rng.standard_normal(n) * 0.3
    data['AVH'] = 32.0 + rng.standard_normal(n) * 0.1
    data['HH'] = _trend(n, 22000.0, 0.0008)

    # Tax rates
    data['TCPRO'] = 0.25
    data['TPBRZ'] = 0.20
    data['NIS'] = _trend(n, 3000.0, 0.005)

    # Investment parameters
    data['IIB'] = 0.05
    data['SIB'] = 0.04
    data['FP'] = 0.03
    data['SP'] = 0.05
    data['SV'] = 0.03
    data['DELTA'] = 0.02
    data['DEBTW'] = 0.35
    data['NDIV'] = _trend(n, 3.0, 0.005)

    # Government/policy (exogenous paths)
    data['EGG'] = _trend(n, 5500.0, 0.002) * (1 + rng.standard_normal(n) * 0.002)
    data['CGWADJ'] = 1.0
    data['LAWADJ'] = 1.0
    data['ERCG'] = _trend(n, 400.0, 0.005)
    data['ERLA'] = _trend(n, 350.0, 0.005)
    data['ADJW'] = 1.0

    # North Sea
    data['NSGVA'] = _trend(n, 8000.0, -0.01) * (1 + rng.standard_normal(n) * 0.02)
    data['NSGVA'] = np.maximum(data['NSGVA'], 1000.0)
    data['XOIL'] = 1.37 * data['NSGVA']

    # Fiscal exogenous
    data['CGGPSPSF'] = _trend(n, 50000.0, 0.005)
    data['PSCE'] = _trend(n, 180000.0, 0.005)
    data['PSNI'] = _trend(n, 15000.0, 0.005)
    data['CGIPS'] = _trend(n, 5000.0, 0.005)
    data['LAIPS'] = _trend(n, 3000.0, 0.005)

    # Housing
    data['LHP'] = _trend(n, 800000.0, 0.01)
    data['OLPE'] = _trend(n, 100000.0, 0.008)
    data['STUDENT'] = _trend(n, 50000.0, 0.02)
    data['HRRPW'] = _trend(n, 100.0, 0.005)
    data['PRP'] = _trend(n, 200.0, 0.01)

    # Various exogenous fiscal/benefit items
    for var in ['VREC', 'TXFUEL', 'TXTOB', 'TXALC', 'CUST', 'CCL', 'AL',
                'TXCUS', 'VEDHH', 'BBC', 'PASSPORT', 'OHT', 'VEDCO',
                'NNDRA', 'LAPT', 'OPT', 'EUETS', 'CIL', 'ENVLEVY',
                'BANKROLL', 'RULC', 'BLEVY', 'SWISSCAP', 'BETPRF',
                'BETLEVY', 'OFGEM', 'EXDUTAC', 'XLAVAT', 'LAVAT',
                'TSD', 'ROCS', 'TXMIS', 'RFP', 'EUOT', 'NPISHTC',
                'PRT', 'CGT', 'FCACA', 'PROV', 'TYPCO', 'LAEPS',
                'INHT', 'CC', 'EENIC', 'EMPNIC', 'OSPC',
                'LARENT', 'PCRENT', 'INCTAC', 'NICAC', 'CGACRES',
                'ILGAC', 'CONACC', 'MFTRAN', 'LAAC', 'LAMFT', 'PCAC',
                'PCGILT', 'MFTPC', 'CGGILTS', 'OFLPS', 'NATSAV',
                'MKTIG', 'LAGILT', 'LCGLA', 'LCGPC',
                'CGLIQ', 'LALIQ', 'PSNDRES', 'GGGDRES',
                'FLEASGG', 'FLEASPC', 'LCGOS', 'LCGPR', 'CGMISP',
                'LALEND', 'LAMISE', 'PCLEND', 'PCMISE',
                'IBPC', 'KCGPSO', 'KPSCG', 'KLA', 'KGLAPC', 'KGLA',
                'KPCPS', 'KPSPC', 'ASSETSA', 'NPACG', 'NPALA',
                'NPAHH', 'NPAA', 'EUKT', 'OPSKTA',
                'PCCON', 'PCIH', 'PCLEB', 'IPRL', 'IPRLPS',
                'CGSB', 'LASBHH', 'LANCGA', 'LAOTRHH', 'CGOTR',
                'DICGOP', 'DILAPR', 'DIPCOP', 'LAPR',
                'LASUBP', 'CGSUBPR', 'CCLACA',
                'RCGIM', 'RLAIM', 'TROD',
                'CGCGLA', 'CGINTRA', 'LAINTRA', 'PCINTRA',
                'LANNDR', 'KCGLA', 'KCGPC',
                'RNCG', 'HHTCG', 'LANDIV', 'PCNDIV', 'CGNDIV',
                'IPCPS', 'KGLAPC_f', 'NSCTP', 'NNSCTP',
                'M4OFC', 'COIN', 'M0',
                'APIIH', 'DIPHH', 'DIPHHuf', 'DIRHH', 'FSMADJ',
                'OSHH', 'SIPT',
                'INSURE', 'ITA_exog',
                'FISIMROW', 'FISIMGG',
                'TRGDP', 'SDE', 'SDEPS',
                'SPECX', 'ALAD', 'CORP',
                'STLIC', 'FXLIC', 'BLIC',
                'PRXMIP', 'OOH', 'HWA', 'H16',
                'MILAPM', 'CTC', 'TSEOP', 'TCINV',
                'XS', 'IH']:
        data[var] = _trend(n, 500.0 + rng.random() * 5000, 0.005) * (1 + rng.standard_normal(n) * 0.01)

    # ===================================================================
    # ENDOGENOUS VARIABLES - Historical seed values
    # ===================================================================

    # GDP and components (GBP millions, quarterly, constant prices)
    base_gdp = 300000.0
    gdp = _trend(n, base_gdp, 0.005) * (1 + rng.standard_normal(n) * 0.003)
    data['GDPM'] = gdp
    data['CONS'] = 0.62 * gdp
    data['IF'] = 0.17 * gdp
    data['CGG'] = 0.20 * gdp
    data['X'] = 0.30 * gdp
    data['M'] = 0.31 * gdp
    data['DINV'] = rng.standard_normal(n) * 500
    data['VAL'] = 0.01 * gdp
    data['SDE'] = 0.01 * gdp
    data['TFE'] = data['CGG'] + data['CONS'] + data['DINV'] + data['VAL'] + data['IF'] + data['X']
    data['GVA'] = 0.90 * gdp
    data['BPA'] = gdp - data['GVA']
    data['GGVA'] = 0.18 * data['GVA']
    data['MSGVA'] = data['GVA'] - data['GGVA']
    data['NNSGVA'] = data['GVA'] - data['NSGVA']

    # Nominal GDP and components
    price_level = _trend(n, 40.0, 0.005)
    data['PGDP'] = price_level * 100 / base_gdp * gdp / gdp  # normalise
    data['PGDP'] = _trend(n, 60.0, 0.005)
    data['GDPMPS'] = data['GDPM'] * data['PGDP'] / 100
    data['CONSPS'] = data['CONS'] * _trend(n, 65.0, 0.005) / 100
    data['IFPS'] = data['IF'] * _trend(n, 70.0, 0.005) / 100

    # Price indices (base ≈ 100 around 2019)
    q2019 = 196  # approximate index of 2019Q1
    data['PCE'] = _trend(n, 30.0, 0.005)
    data['CPI'] = _trend(n, 30.0, 0.005)
    data['CPIH'] = data['CPI'] * 1.005
    data['CPIX'] = data['CPI'] * 0.98
    data['CPIRENT'] = _trend(n, 28.0, 0.006)
    data['PIF'] = _trend(n, 32.0, 0.005)
    data['PIBUS'] = _trend(n, 33.0, 0.005)
    data['PIH'] = _trend(n, 31.0, 0.006)
    data['PINV'] = _trend(n, 80.0, 0.003)
    data['PXNOG'] = _trend(n, 35.0, 0.004)
    data['PXS'] = data['PXNOG'] * 1.02
    data['PXOIL'] = data['PBRENT'] * 0.8
    data['PMNOG'] = _trend(n, 34.0, 0.004)
    data['PMS'] = data['PMNOG'] * 1.01
    data['PMOIL'] = data['PXOIL'] * 1.0
    data['PPIY'] = _trend(n, 35.0, 0.004)
    data['PMSGVA'] = _trend(n, 60.0, 0.005)
    data['PGVA'] = _trend(n, 62.0, 0.005)
    data['PCDUR'] = _trend(n, 33.0, 0.003)
    data['PRENT'] = _trend(n, 80.0, 0.006)
    data['PDINV'] = _trend(n, 70.0, 0.004)
    data['GGFCD'] = _trend(n, 70.0, 0.005)
    data['GGIDEF'] = _trend(n, 72.0, 0.005)
    data['PKMSXHB'] = data['PIBUS']

    # Employment (thousands)
    data['ET'] = _trend(n, 25000.0, 0.002) * (1 + rng.standard_normal(n) * 0.002)
    data['ETLFS'] = data['ET'] * 1.02
    data['EMS'] = data['ET'] * 0.78
    data['ECG'] = data['ET'] * 0.08
    data['ELA'] = data['ET'] * 0.07
    data['ES'] = data['ET'] * 0.12
    data['ESLFS'] = data['ES'] * 1.02
    data['ULFS'] = data['POP16'] * data['PART16'] / 100 - data['ETLFS']
    data['LFSUR'] = 100 * data['ULFS'] / (data['ETLFS'] + data['ULFS'])

    # Hours and productivity
    data['HWA'] = data['ETLFS'] * data['AVH'] / 1000
    data['APH'] = _trend(n, 25.0, 0.003)
    data['PRODH'] = data['GDPM'] / data['HWA']
    data['HD'] = data['APH']

    # Wages and income (GBP millions)
    data['PSAVEI'] = _trend(n, 500.0, 0.008)
    data['WFP'] = _trend(n, 120000.0, 0.006)
    data['FYEMP'] = data['WFP'] * 1.15
    data['EARN'] = data['WFP'] / (data['ETLFS'] - data['ESLFS'])
    data['EMPSC'] = data['FYEMP'] - data['WFP']
    data['EMPISC'] = data['EMPSC'] * 0.4
    data['HHISC'] = data['EMPISC'] * 0.6
    data['MI'] = _trend(n, 15000.0, 0.005)

    # Income flows
    data['HHDI'] = _trend(n, 250000.0, 0.005)
    data['RHHDI'] = 100 * data['HHDI'] / data['PCE']

    # Consumption sub-components
    data['CDUR'] = data['CONS'] * 0.12
    data['CDURPS'] = data['CDUR'] * data['PCDUR'] / 100
    data['PD'] = _trend(n, 150.0, 0.005)

    # Trade sub-components
    data['XNOG'] = data['X'] * 0.55
    data['XS'] = data['X'] * 0.35
    data['XPS'] = _trend(n, 90000.0, 0.006)
    data['MNOG'] = data['M'] * 0.55
    data['MS'] = data['M'] * 0.30
    data['MOIL'] = data['M'] * 0.05
    data['MPS'] = _trend(n, 95000.0, 0.006)
    data['TDOIL'] = data['MOIL'] + data['NSGVA'] - data['XOIL']

    # Investment components
    data['IBUS'] = data['IF'] * 0.55
    data['IBUSX'] = data['IBUS']
    data['GGI'] = data['IF'] * 0.18
    data['GGIPS'] = data['CGIPS'] + data['LAIPS']
    data['GGIX'] = data['GGI']
    data['HIMPROV'] = _trend(n, 10000.0, 0.005)
    data['IH'] = data['IF'] * 0.12
    data['IHPS'] = data['IH'] * data['PIH'] / 100
    data['IHHPS'] = data['IHPS'] * 0.9
    data['ICCPS'] = _trend(n, 30000.0, 0.005)
    data['IFCPS'] = _trend(n, 10000.0, 0.005)
    data['VALPS'] = data['VAL'] * data['PIF'] / 100
    data['VALHH'] = 0.25 * data['VALPS']

    # Capital stock
    data['KMSXH'] = _trend(n, 1000.0, 0.005)
    data['KSTAR'] = data['KMSXH'] * 1000
    data['KGAP'] = np.log(data['KMSXH'] * 1000) - np.log(data['KSTAR'])

    # Cost of capital components
    data['DB'] = 0.5
    data['DP'] = 0.5
    data['DV'] = 0.3
    data['WB'] = 0.31
    data['WP'] = 0.54
    data['WV'] = 0.14
    data['WG'] = 0.03
    data['TAFB'] = 1.0
    data['TAFP'] = 1.0
    data['TAFV'] = 1.0
    data['TAF'] = 1.0
    data['CDEBT'] = _ar1(n, 5.0, 0.95, 0.2, seed=10)
    data['CEQUITY'] = 8.0
    data['RWACC'] = 6.0
    data['RDELTA'] = 0.022
    data['COCU'] = 0.08
    data['COC'] = 0.08
    data['TQ'] = 0.1
    data['RIC'] = _ar1(n, 5.0, 0.95, 0.3, seed=11)

    # Housing
    data['HSALL'] = _trend(n, 20000.0, 0.002)
    data['NETAD'] = 50.0
    data['PEHC'] = _trend(n, 40.0, 0.003)
    data['GPW'] = _trend(n, 150.0, 0.006)

    # Inventories
    data['INV'] = _trend(n, 100000.0, 0.003)
    data['BV'] = _trend(n, 80000.0, 0.004)
    data['SA'] = rng.standard_normal(n) * 200
    data['DINVPS'] = data['DINV'] * data['PDINV'] / 100
    data['DINVHH'] = 0.07 * data['DINVPS']
    data['DINVCG'] = rng.standard_normal(n) * 100

    # Public expenditure
    data['CGWS'] = _trend(n, 20000.0, 0.005)
    data['LAWS'] = _trend(n, 15000.0, 0.005)
    data['CGP'] = _trend(n, 8000.0, 0.005)
    data['OSGG'] = data['RCGIM'] + data['RLAIM'] + 100
    data['CGGPS'] = data['CGWS'] + data['LAWS'] + data['CGP'] + data['LAPR'] + data['RCGIM'] + data['RLAIM']
    data['CGGPSPSF'] = data['CGGPS']

    # Fiscal totals
    data['BPAPS'] = _trend(n, 40000.0, 0.005)
    data['GVAPS'] = data['GDPMPS'] - data['BPAPS']
    data['GGVAPS'] = data['CGWS'] + data['LAWS'] + data['OSGG']
    data['MSGVAPS'] = data['GVAPS'] - data['GGVAPS']

    # Tax revenues
    data['CT'] = _trend(n, 8000.0, 0.005)
    data['CETAX'] = _trend(n, 12000.0, 0.005)
    data['TYEM'] = _trend(n, 40000.0, 0.005)
    data['INCTAXG'] = _trend(n, 42000.0, 0.005)
    data['PSCR'] = _trend(n, 170000.0, 0.005)
    data['NATAXES'] = _trend(n, 160000.0, 0.005)

    # Public sector totals
    data['PSCB'] = rng.standard_normal(n) * 5000
    data['PSNBCY'] = _trend(n, 10000.0, 0.003) + rng.standard_normal(n) * 3000
    data['PSND'] = _trend(n, 500000.0, 0.01)
    data['GGGD'] = data['PSND'] * 0.95
    data['TME'] = _trend(n, 200000.0, 0.005)
    data['DEP'] = data['RCGIM'] + data['RLAIM'] + data['PCCON']

    # Balance of payments
    data['TB'] = rng.standard_normal(n) * 5000
    data['CB'] = rng.standard_normal(n) * 5000
    data['CBPCNT'] = data['CB'] / data['GDPMPS'] * 100
    data['NIPD'] = _trend(n, -2000.0, 0.005)
    data['CIPD'] = _trend(n, 20000.0, 0.005)
    data['DIPD'] = _trend(n, 22000.0, 0.005)
    data['EECOMPD'] = _trend(n, 1500.0, 0.005)
    data['EECOMPC'] = _trend(n, 800.0, 0.005)
    data['ECUPO'] = data['RX'] * 1.15
    data['SRES'] = _trend(n, 20000.0, 0.003)
    data['DRES'] = 0.0
    data['CGCBOP'] = _trend(n, 500.0, 0.003)
    data['CGC'] = _trend(n, 2000.0, 0.003)

    # Financial: households
    data['NFWPE'] = _trend(n, 1500000.0, 0.008)
    data['GFWPE'] = _trend(n, 3000000.0, 0.008)
    data['DEPHH'] = _trend(n, 800000.0, 0.008)
    data['DEPHHx'] = data['DEPHH']
    data['EQHH'] = _trend(n, 500000.0, 0.01)
    data['PIHH'] = _trend(n, 1500000.0, 0.008)
    data['OAHH'] = _trend(n, 200000.0, 0.007)
    data['OAHHx'] = data['OAHH']
    data['OLPEx'] = _trend(n, 100000.0, 0.008)
    data['EQPR'] = _trend(n, 50.0, 0.01)
    data['DBR'] = 0.5
    data['NAEQHH'] = rng.standard_normal(n) * 5000
    data['NAEQHHx'] = data['NAEQHH']
    data['NAPEN'] = _trend(n, 5000.0, 0.005)
    data['NAINS'] = rng.standard_normal(n) * 2000
    data['NAINSx'] = data['NAINS']
    data['NAOLPE'] = rng.standard_normal(n) * 3000
    data['NAOLPEx'] = data['NAOLPE']
    data['DEBTU'] = 0.02
    data['GMF'] = 0.01
    data['SVHH'] = _trend(n, 10000.0, 0.005)
    data['NAFHH'] = rng.standard_normal(n) * 5000
    data['NAFHHNSA'] = data['NAFHH']
    data['NLHH'] = data['NAFHHNSA']
    data['KGHH'] = _trend(n, 2000.0, 0.005)
    data['SY'] = 5.0 + rng.standard_normal(n) * 1.0
    data['NEAHH'] = _trend(n, 5000.0, 0.005)
    data['DEPHHADJ'] = 0.0
    data['NAEQHHADJ'] = 0.0
    data['NAINSADJ'] = 0.0
    data['NAOLPEADJ'] = 0.0

    # Financial: rest of world
    data['AROW'] = _trend(n, 4000000.0, 0.01)
    data['LROW'] = _trend(n, 4500000.0, 0.01)
    data['DAROW'] = data['AROW'] * 0.3
    data['EQAROW'] = data['AROW'] * 0.3
    data['BAROW'] = data['AROW'] * 0.25
    data['OTAROW'] = data['AROW'] * 0.15
    data['DLROW'] = data['LROW'] * 0.25
    data['EQLROW'] = data['LROW'] * 0.3
    data['BLROW'] = data['LROW'] * 0.25
    data['OTLROW'] = data['LROW'] * 0.2
    data['NIIP'] = data['LROW'] + data['SRES'] - data['AROW']
    data['NAFROW'] = rng.standard_normal(n) * 5000
    data['NAFROWNSA'] = data['NAFROW']

    # Financial: corporate
    data['NWIC'] = _trend(n, 100000.0, 0.005)
    data['AIC'] = _trend(n, 500000.0, 0.008)
    data['LIC'] = _trend(n, 400000.0, 0.008)
    data['EQLIC'] = data['LIC'] * 0.4
    data['BLIC'] = data['LIC'] * 0.15
    data['STLIC'] = data['LIC'] * 0.1
    data['FXLIC'] = data['LIC'] * 0.08
    data['OLIC'] = data['LIC'] * 0.05

    # Operating surplus components
    data['OS'] = _trend(n, 80000.0, 0.005)
    data['FYCPR'] = _trend(n, 50000.0, 0.005)
    data['GTPFC'] = _trend(n, 40000.0, 0.005)
    data['FC'] = _trend(n, 45000.0, 0.005)
    data['OSCO'] = data['OS'] * 0.7
    data['RENTCO'] = _trend(n, 5000.0, 0.005)
    data['IROO'] = data['PRENT'] * data['POP16'] / 1000
    data['FISIMPS'] = _trend(n, 10000.0, 0.005)

    # National income
    data['GNIPS'] = data['GDPMPS'] * 1.01
    data['GAP'] = data['GDPM'] / data['TRGDP'] * 100 - 100

    # Wage bill components
    data['CGASC'] = _trend(n, 2000.0, 0.005)
    data['CGISC'] = _trend(n, 1000.0, 0.005)
    data['LASC'] = _trend(n, 1500.0, 0.005)
    data['EESCCG'] = _trend(n, 3000.0, 0.005)
    data['EESCLA'] = _trend(n, 2500.0, 0.005)
    data['EMPCPP'] = _trend(n, 4000.0, 0.005)
    data['EMPISCPP'] = _trend(n, 2000.0, 0.005)
    data['OSB'] = _trend(n, 3000.0, 0.005)
    data['HHSB'] = 2 * data['HHISC']
    data['EMPASC'] = data['EMPSC'] - data['EMPISC']

    # Benefits and transfers
    data['SBHH'] = _trend(n, 60000.0, 0.005)
    data['TYWHH'] = _trend(n, 50000.0, 0.005)
    data['NMTRHH'] = _trend(n, 3000.0, 0.003)
    data['PIRHH'] = _trend(n, 30000.0, 0.006)
    data['PIPHH'] = _trend(n, 15000.0, 0.006)
    data['NDIVHH'] = _trend(n, 8000.0, 0.006)
    data['WYQC'] = _trend(n, 2000.0, 0.005)
    data['EECPP'] = _trend(n, 5000.0, 0.005)
    data['EESC'] = _trend(n, 12000.0, 0.005)

    # Interest income/payments
    data['DIRHH'] = _trend(n, 8000.0, 0.005)
    data['DIRHHf'] = _trend(n, -2000.0, 0.005)
    data['DIRHHx'] = data['DIRHH'] - data['DIRHHf']
    data['DIPHHx'] = _trend(n, 12000.0, 0.005)
    data['DIPHHmf'] = _trend(n, 3000.0, 0.005)
    data['DIRIC'] = _trend(n, 5000.0, 0.005)
    data['DIRICf'] = _trend(n, -1000.0, 0.005)
    data['DIRICx'] = data['DIRIC'] - data['DIRICf']
    data['DIPIC'] = _trend(n, 4000.0, 0.005)
    data['DIPICf'] = _trend(n, 1000.0, 0.005)
    data['DIPICx'] = data['DIPIC'] + data['DIPICf']

    # Sector net lending
    data['NAFCO'] = rng.standard_normal(n) * 5000
    data['NAFFC'] = rng.standard_normal(n) * 3000
    data['NAFIC'] = data['NAFCO'] - data['NAFFC']
    data['SAVCO'] = _trend(n, 20000.0, 0.005)

    # ULC and cost indices
    data['ULCPS'] = _trend(n, 60.0, 0.005)
    data['ULCMS'] = _trend(n, 65.0, 0.005)
    data['MCOST'] = 100.0
    data['SCOST'] = 100.0
    data['CCOST'] = 100.0
    data['UTCOST'] = 100.0
    data['RPCOST'] = 100.0
    data['ICOST'] = 100.0
    data['XGCOST'] = 100.0
    data['XSCOST'] = 100.0
    data['MKGW'] = 100.0
    data['MKR'] = 100.0

    # RPI components
    data['PRMIP'] = _trend(n, 200.0, 0.005)
    data['PR'] = _trend(n, 250.0, 0.005)
    data['RPI'] = 3.0

    # Misc identities
    data['RPRICE'] = _trend(n, 1.0, 0.001)
    data['RPW'] = _trend(n, 30.0, 0.003)
    data['RCW'] = _trend(n, 25.0, 0.003)
    data['WRGTP'] = _trend(n, 1000.0, 0.002)
    data['WFJ'] = data['ET'] + data['WRGTP']
    data['ER'] = 100 * data['ETLFS'] / data['POP16']

    # Mortgage and housing rates
    data['RHF'] = _trend(n, 4.0, 0.001)
    data['NSGTP'] = _trend(n, 3000.0, 0.005)
    data['NNSGTP'] = _trend(n, 40000.0, 0.005)

    # Public sector lending
    data['CGNB'] = rng.standard_normal(n) * 3000
    data['LANB'] = rng.standard_normal(n) * 2000
    data['GGNB'] = data['CGNB'] + data['LANB']
    data['GGNBCY'] = data['GGNB']
    data['PCNB'] = rng.standard_normal(n) * 1000
    data['PCNBCY'] = data['PCNB']
    data['TDEF'] = data['CGNB'] + data['LANB']
    data['PSGI'] = _trend(n, 18000.0, 0.005)
    data['PSNBNSA'] = data['PSNBCY']

    data['PSTA'] = _trend(n, 400000.0, 0.005)
    data['PSFA'] = _trend(n, 100000.0, 0.005)
    data['PSNW'] = data['PSTA'] + data['PSFA'] - _trend(n, 500000.0, 0.01)

    data['CGNCR'] = rng.standard_normal(n) * 3000
    data['PSNCR'] = rng.standard_normal(n) * 5000
    data['LABRO'] = rng.standard_normal(n) * 2000

    data['SWAPS'] = 0.0

    # Government totals
    data['CGTSUB'] = _trend(n, 5000.0, 0.005)
    data['LATSUB'] = _trend(n, 3000.0, 0.005)
    data['LASUBPR'] = _trend(n, 1500.0, 0.005)
    data['CGSUBP'] = _trend(n, 3000.0, 0.005)
    data['CGNCGA'] = data['TROD']

    # Public corp
    data['PUBSTIW'] = _trend(n, 50000.0, 0.005)
    data['PUBSTPD'] = _trend(n, 50000.0, 0.005)

    # Transfers
    data['TRANC'] = _trend(n, 5000.0, 0.005)
    data['TRAND'] = _trend(n, 6000.0, 0.005)
    data['TRANB'] = data['TRANC'] - data['TRAND']

    # EU variables
    data['EUSUBP'] = 0.0
    data['EUSUBPR'] = _trend(n, 500.0, 0.003)
    data['EUSF'] = _trend(n, 300.0, 0.003)
    data['ECNET'] = _trend(n, 2000.0, 0.003)
    data['GNP4'] = _trend(n, 1000.0, 0.005)
    data['EUVAT'] = _trend(n, 500.0, 0.005)
    data['BENAB'] = _trend(n, 200.0, 0.005)
    data['CGITFA'] = _trend(n, 100.0, 0.005)
    data['ITA'] = _trend(n, 150.0, 0.005)
    data['HHTFA'] = _trend(n, 300.0, 0.005)
    data['HHTA'] = _trend(n, 200.0, 0.005)
    data['CGKTA'] = _trend(n, 400.0, 0.005)

    # Imports sub-components
    data['MC'] = 0.257 * data['CONS']
    data['MCGG'] = 0.094 * data['CGG']
    data['MIF'] = 0.234 * data['IF']
    data['MDINV'] = 0.106 * data['DINV']
    data['MXS'] = 0.142 * data['XS']
    data['MXG'] = 0.376 * (data['XOIL'] + data['XNOG'])
    data['MTFE'] = data['MC'] + data['MCGG'] + data['MIF'] + data['MDINV'] + data['MXS'] + data['MXG']
    data['MINTY'] = 100.0
    data['MGTFE'] = _trend(n, 50000.0, 0.005)
    data['PMGREL'] = 1.0
    data['MSTFE'] = _trend(n, 15000.0, 0.005)
    data['PMSREL'] = 1.0

    # Financial sector
    data['M4IC'] = _trend(n, 500000.0, 0.008)
    data['M4'] = data['DEPHH'] + data['M4IC'] + data['M4OFC']

    # Asset base year constants
    data['OILBASE'] = 50.0
    data['ULCPSBASE'] = 60.0
    data['ULCMSBASE'] = 65.0
    data['PMNOGBASE'] = 34.0
    data['PMSBASE'] = 35.0
    data['TXRATEBASE'] = 0.15
    data['PPIYBASE'] = 35.0
    data['CPIXBASE'] = 30.0

    # Export returns
    data['REXC'] = 3.0
    data['REXD'] = 3.0

    # Misc ratios
    data['SDLHH'] = 0.0
    data['SDLROW'] = 0.0
    data['SDI'] = _trend(n, 2000.0, 0.003)

    # Housing completions
    data['PCLEB'] = _trend(n, 2000.0, 0.003)

    # Financial transactions
    for var in ['NAEQAROW', 'NABAROW', 'NAOTAROW', 'NAOTLROW',
                'NADLROW', 'NAEQLROW', 'NABLROW',
                'AAROW', 'ALROW_calc',
                'NABLIC', 'NAFXLIC', 'NAEQLIC', 'NALIC',
                'NAAIC', 'AAHH', 'ALHH', 'HHRES',
                'OAHHADJ', 'GGLIQ', 'PSLSFA', 'PSACADJ',
                'CGACADJ', 'PSFL', 'CGLSFA']:
        data[var] = rng.standard_normal(n) * 1000

    # Public sector receipts detail
    data['TAXCRED'] = _trend(n, 5000.0, 0.005)
    data['PSINTR'] = _trend(n, 1000.0, 0.005)
    data['CGRENT'] = _trend(n, 500.0, 0.005)
    data['VED'] = _trend(n, 1500.0, 0.005)
    data['OCT'] = _trend(n, 2000.0, 0.005)
    data['TPRODPS'] = _trend(n, 5000.0, 0.005)

    # Demand weighted averages
    data['MSGVAPSEMP'] = data['MSGVAPS'] - data['MI']
    data['FYEMPMS'] = data['FYEMP'] - data['CGWS'] - data['LAWS']

    # EPS (private sector employment)
    data['EPS'] = data['ET'] - data['ECG'] - data['ELA']

    # GAD (total population in demographic groups)
    data['GAD'] = data['GAD1'] + data['GAD2'] + data['GAD3']

    # NFWPE
    data['NFWPE'] = data['GFWPE'] - data['LHP'] - data['OLPE']

    # CGGPS (government consumption spending)
    data['CGGPS'] = data['CGWS'] + data['LAWS'] + data['CGP'] + data['LAPR'] + data['RCGIM'] + data['RLAIM']

    # GDPM from nominal (ensure consistency)
    data['GDPM'] = gdp

    # ===================================================================
    # ADDITIVE ADJUSTMENTS (all zero)
    # ===================================================================
    for adj in config.ADDITIVE_ADJUSTMENTS.values():
        data[adj] = 0.0

    # ===================================================================
    # Defragment the DataFrame and ensure all values are float
    # ===================================================================
    data = data.copy()
    data = data.astype(float)

    return data
