"""
Translation of EViews functions and syntax into Python equivalents.

EViews Construct -> Python Translation:
    dlog(X) on LHS   -> X = X(-1) * exp(RHS)
    d(X) on LHS      -> X = X(-1) + RHS
    X / X(-1) = RHS  -> X = X(-1) * RHS
    @recode           -> date comparison returning 0.0 or 1.0
    @elem             -> cached value lookup at specific date
    @TREND            -> integer quarter count from base date
    @ADD(V)           -> post-convergence additive adjustment
"""

import numpy as np
import pandas as pd


def dlog_solve(rhs_value, lag1_value):
    """Given dlog(X) = rhs, solve for X: X = X(-1) * exp(rhs)."""
    return lag1_value * np.exp(rhs_value)


def d_solve(rhs_value, lag1_value):
    """Given d(X) = rhs, solve for X: X = X(-1) + rhs."""
    return lag1_value + rhs_value


def ratio_solve(rhs_value, lag1_value):
    """Given X / X(-1) = rhs, solve for X: X = X(-1) * rhs."""
    return lag1_value * rhs_value


def safe_log(x):
    """np.log with guard against zero/negative values."""
    return np.log(max(x, 1e-10))


def parse_eviews_date(date_str):
    """Convert EViews date format '2009:04' to pd.Period('2009Q4')."""
    date_str = date_str.strip().strip('"').strip("'")
    year, q = date_str.split(':')
    return pd.Period(f"{year}Q{int(q)}", freq='Q')


def recode_eq(current_period, date_str):
    """@recode(@date = @dateval("2009:04"), 1, 0)"""
    target = parse_eviews_date(date_str)
    return 1.0 if current_period == target else 0.0


def recode_geq(current_period, date_str):
    """@recode(@date >= @dateval("2005:01"), 1, 0)"""
    target = parse_eviews_date(date_str)
    return 1.0 if current_period >= target else 0.0


def recode_leq(current_period, date_str):
    """@recode(@date <= @dateval("2011:02"), 1, 0)"""
    target = parse_eviews_date(date_str)
    return 1.0 if current_period <= target else 0.0


def elem(data, variable, date_str):
    """@elem(X, "2009Q1") - retrieve value at a specific date."""
    period = pd.Period(date_str.strip().strip('"'), freq='Q')
    return data.at[period, variable]


def trend(current_period, base_date_str):
    """@TREND(1979Q4) - number of quarters since base date."""
    base = pd.Period(base_date_str.strip(), freq='Q')
    return int((current_period - base).n)


def v(data, t, var, lag=0):
    """Read variable value at period t - lag quarters."""
    period = t - lag if lag > 0 else t
    return data.at[period, var]
