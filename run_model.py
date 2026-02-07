"""
Exchequer - OBR Macroeconomic Model in Python
Entry point for running the model with synthetic data.

Usage:
    python run_model.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from obr_model.model import OBRModel


def main():
    print("=" * 60)
    print("  Exchequer - OBR Macroeconomic Model")
    print("=" * 60)
    print()

    model = OBRModel()

    print("Step 1: Generating synthetic data...")
    model.generate_synthetic_data()
    print()

    print("Step 2: Solving model...")
    print()
    results = model.solve('2025Q1', '2028Q4', verbose=True)
    print()

    print("Step 3: Results summary")
    model.summary('2025Q4')
    model.summary('2028Q4')

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')
    print(f"Step 4: Exporting results to {output_path}")
    model.export_results(output_path)

    print()
    print("Done!")


if __name__ == '__main__':
    main()
