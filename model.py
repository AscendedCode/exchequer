"""
Main OBRModel orchestrator class.

Brings together data, equations, and solver to run the complete
OBR macroeconomic model.
"""

import pandas as pd
from .solver import GaussSeidelSolver
from . import config


class OBRModel:
    """
    OBR Macroeconomic Model.

    Usage:
        model = OBRModel()
        model.generate_synthetic_data()
        model.solve('2025Q1', '2030Q4')
        model.export_results('obr_results.xlsx')
    """

    def __init__(self, max_iter=None, tolerance=None, damping=None):
        self.data = None
        self.equations = self._register_all_equations()
        self.solver = GaussSeidelSolver(
            self.equations,
            max_iter=max_iter,
            tolerance=tolerance,
            damping=damping,
        )

    def _register_all_equations(self):
        """Collect all equations from all group modules in evaluation order."""
        from .equations import get_all_equations
        return get_all_equations()

    def generate_synthetic_data(self):
        """Create placeholder data for structural testing."""
        from .data_generator import generate_synthetic_data
        self.data = generate_synthetic_data()
        print(f"Generated synthetic data: {len(self.data)} quarters, "
              f"{len(self.data.columns)} variables")

    def load_data(self, data):
        """Load a pre-built DataFrame of data."""
        if not isinstance(data.index, pd.PeriodIndex):
            raise ValueError("Data must have a PeriodIndex with freq='Q'")
        self.data = data

    def solve(self, start=None, end=None, verbose=True):
        """
        Run the model solver.

        Args:
            start: Start period (str like '2025Q1' or pd.Period). Defaults to FORECAST_START.
            end: End period. Defaults to FORECAST_END.
            verbose: Print progress information.
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call generate_synthetic_data() or load_data() first.")

        start_p = pd.Period(start, freq='Q') if start else config.FORECAST_START
        end_p = pd.Period(end, freq='Q') if end else config.FORECAST_END

        if verbose:
            n_quarters = (end_p - start_p).n + 1
            print(f"Solving {n_quarters} quarters from {start_p} to {end_p}")
            print(f"  Equations: {len(self.equations)}")
            print(f"  Max iterations: {self.solver.max_iter}")
            print(f"  Tolerance: {self.solver.tol}")
            print(f"  Damping: {self.solver.damping}")
            print()

        results = self.solver.solve_range(self.data, start_p, end_p, verbose=verbose)
        return results

    def export_results(self, filepath, variables=None):
        """
        Export results to Excel or CSV.

        Args:
            filepath: Output path (.xlsx or .csv)
            variables: Optional list of variable names to export. Exports all if None.
        """
        if self.data is None:
            raise RuntimeError("No data to export.")

        export_data = self.data if variables is None else self.data[variables]

        if filepath.endswith('.xlsx'):
            export_data.to_excel(filepath)
        elif filepath.endswith('.csv'):
            export_data.to_csv(filepath)
        else:
            export_data.to_csv(filepath)

        print(f"Results exported to {filepath}")

    def get_variable(self, name, start=None, end=None):
        """Get a single variable's time series."""
        if self.data is None:
            raise RuntimeError("No data loaded.")
        series = self.data[name]
        if start:
            series = series[series.index >= pd.Period(start, freq='Q')]
        if end:
            series = series[series.index <= pd.Period(end, freq='Q')]
        return series

    def summary(self, period=None):
        """Print a summary of key variables for a given period."""
        if self.data is None:
            raise RuntimeError("No data loaded.")

        if period is None:
            period = config.FORECAST_END

        t = pd.Period(period, freq='Q') if isinstance(period, str) else period

        key_vars = [
            ('GDPM', 'Real GDP'),
            ('GDPMPS', 'Nominal GDP'),
            ('CONS', 'Consumption'),
            ('IF', 'Investment'),
            ('X', 'Exports'),
            ('M', 'Imports'),
            ('CPI', 'CPI'),
            ('LFSUR', 'Unemployment rate (%)'),
            ('PSNBCY', 'Public sector net borrowing'),
            ('PSND', 'Public sector net debt'),
            ('CB', 'Current account balance'),
        ]

        print(f"\n=== Model Summary for {t} ===")
        for var, label in key_vars:
            if var in self.data.columns:
                val = self.data.at[t, var]
                print(f"  {label:35s}: {val:>15,.1f}")
        print()
