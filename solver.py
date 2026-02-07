"""
Gauss-Seidel solver for the OBR macroeconomic model.

Solves the simultaneous equation system one quarter at a time by iterating
through all equations until convergence. Includes a special pre-solver for
the SCOST/CCOST/UTCOST input-output cost block.
"""

import numpy as np
import pandas as pd
from . import config


class ConvergenceError(Exception):
    """Raised when the solver fails to converge within max iterations."""
    pass


class GaussSeidelSolver:
    """
    Iterative Gauss-Seidel solver for simultaneous equation systems.

    For each quarter t:
    1. Initialise endogenous variables from previous period
    2. Pre-solve the 3x3 linear cost block (SCOST/CCOST/UTCOST)
    3. Iterate through all equations, updating each variable in place
    4. Check convergence: max relative change < tolerance
    5. Apply @ADD(V) additive adjustments after convergence
    """

    def __init__(self, equations, max_iter=None, tolerance=None, damping=None):
        self.equations = equations
        self.max_iter = max_iter or config.SOLVER_MAX_ITER
        self.tol = tolerance or config.SOLVER_TOLERANCE
        self.damping = damping or config.SOLVER_DAMPING
        self.endogenous_vars = [name for name, _, _ in equations]

    def _initialise_period(self, data, t):
        """Set current-period endogenous vars from previous period as starting guess."""
        t_prev = t - 1
        for name in self.endogenous_vars:
            # Add column if it doesn't exist
            if name not in data.columns:
                data[name] = np.nan
            if pd.isna(data.at[t, name]):
                prev_val = data.at[t_prev, name]
                if not pd.isna(prev_val):
                    data.at[t, name] = prev_val
                else:
                    data.at[t, name] = 1.0  # fallback to avoid NaN

    def _pre_solve_cost_block(self, data, t):
        """
        Pre-solve the SCOST/CCOST/UTCOST simultaneous linear system.

        The equations are:
            SCOST = a1 + 9.78*(PPIY/PPIYBASE) + 1.64*(CCOST/100) + 1.09*(UTCOST/100)
            CCOST = a2 + 27.06*(PPIY/PPIYBASE) + 28.13*(SCOST/100) + 0.34*(UTCOST/100)
            UTCOST = a3 + 8.24*(PPIY/PPIYBASE) + 16.00*(SCOST/100) + 2.95*(CCOST/100)

        Written as: [I - C] * [SCOST, CCOST, UTCOST]' = [a1, a2, a3]'
        """
        if 'SCOST' not in data.columns:
            return

        try:
            # Read base values
            ULCMS = data.at[t, 'ULCMS']
            PMNOG = data.at[t, 'PMNOG']
            PMS = data.at[t, 'PMS']
            PBRENT = data.at[t, 'PBRENT']
            RXD = data.at[t, 'RXD']
            BPAPS = data.at[t, 'BPAPS']
            GVA = data.at[t, 'GVA']
            PPIY = data.at[t, 'PPIY']

            ULCMSBASE = data.at[t, 'ULCMSBASE']
            PMNOGBASE = data.at[t, 'PMNOGBASE']
            PMSBASE = data.at[t, 'PMSBASE']
            OILBASE = data.at[t, 'OILBASE']
            TXRATEBASE = data.at[t, 'TXRATEBASE']
            PPIYBASE = data.at[t, 'PPIYBASE']

            ulc_r = ULCMS / ULCMSBASE
            pmn_r = PMNOG / PMNOGBASE
            pms_r = PMS / PMSBASE
            oil_r = (PBRENT / RXD) / OILBASE
            tx_r = (BPAPS / GVA) / TXRATEBASE
            ppiy_r = PPIY / PPIYBASE

            # Exogenous parts of each cost equation (everything except SCOST, CCOST, UTCOST terms)
            a1 = 70.54 * ulc_r + 6.93 * pmn_r + 6.41 * pms_r + 0.09 * oil_r + 3.52 * tx_r + 9.78 * ppiy_r
            a2 = 40.25 * ulc_r + 2.80 * pmn_r + 0.90 * pms_r + 0.03 * oil_r + 0.51 * tx_r + 27.06 * ppiy_r
            a3 = 14.85 * ulc_r + 3.04 * pmn_r + 0.51 * pms_r + 51.52 * oil_r + 2.90 * tx_r + 8.24 * ppiy_r

            # Coefficient matrix: [I - cross-terms/100]
            # SCOST = a1 + 1.64*(CCOST/100) + 1.09*(UTCOST/100)
            # CCOST = a2 + 28.13*(SCOST/100) + 0.34*(UTCOST/100)
            # UTCOST = a3 + 16.00*(SCOST/100) + 2.95*(CCOST/100)
            A = np.array([
                [1.0, -1.64 / 100, -1.09 / 100],
                [-28.13 / 100, 1.0, -0.34 / 100],
                [-16.00 / 100, -2.95 / 100, 1.0],
            ])
            b = np.array([a1, a2, a3])

            solution = np.linalg.solve(A, b)
            data.at[t, 'SCOST'] = solution[0]
            data.at[t, 'CCOST'] = solution[1]
            data.at[t, 'UTCOST'] = solution[2]
        except Exception:
            pass  # Fall back to Gauss-Seidel iteration

    def _apply_additive_adjustments(self, data, t):
        """Apply @ADD(V) adjustments after convergence."""
        for var, adj_var in config.ADDITIVE_ADJUSTMENTS.items():
            if var in data.columns and adj_var in data.columns:
                adj_val = data.at[t, adj_var]
                if not pd.isna(adj_val):
                    data.at[t, var] += adj_val

    def solve_period(self, data, t, verbose=False):
        """
        Solve all equations for a single quarter t.

        Returns the number of iterations to convergence.
        """
        # Step 1: Initialise
        self._initialise_period(data, t)

        # Step 2: Pre-solve cost block
        self._pre_solve_cost_block(data, t)

        # Step 3: Gauss-Seidel iteration
        for iteration in range(self.max_iter):
            max_change = 0.0
            worst_var = ''

            for name, func, eq_type in self.equations:
                old_val = data.at[t, name]
                if pd.isna(old_val):
                    old_val = 0.0

                try:
                    new_val = func(data, t)
                except Exception as e:
                    if verbose:
                        print(f"  Error in {name}: {e}")
                    continue

                if new_val is None or np.isnan(new_val) or np.isinf(new_val):
                    continue

                # Apply damping
                damped_val = self.damping * new_val + (1 - self.damping) * old_val
                data.at[t, name] = damped_val

                # Track convergence
                if abs(old_val) > 1e-10:
                    rel_change = abs((damped_val - old_val) / old_val)
                    if rel_change > max_change:
                        max_change = rel_change
                        worst_var = name

            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: max_change={max_change:.2e} ({worst_var})")

            if max_change < self.tol:
                # Step 4: Apply additive adjustments
                self._apply_additive_adjustments(data, t)
                return iteration + 1

        # If we get here, we didn't converge
        print(f"  WARNING: Failed to converge at {t} after {self.max_iter} iterations "
              f"(max_change={max_change:.2e} in {worst_var})")
        self._apply_additive_adjustments(data, t)
        return self.max_iter

    def solve_range(self, data, start_period, end_period, verbose=True):
        """Solve the model for a range of quarters sequentially."""
        t = start_period
        results = []
        while t <= end_period:
            if verbose:
                print(f"Solving {t}...", end=' ')
            iters = self.solve_period(data, t, verbose=verbose)
            if verbose:
                print(f"converged in {iters} iterations")
            results.append((t, iters))
            t = t + 1
        return results
