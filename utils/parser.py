# parser.py
# utils/parser.py

import pandas as pd
import numpy as np
import os
import json
from typing import Tuple, Dict

def parse_production_csv(filepath: str) -> pd.DataFrame:
    """
    Parse production data from CSV.

    Assumes columns like:
    - 'time', 'well_name', 'oil_rate', 'water_rate', 'gas_rate', 'pressure'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    df = df.dropna()
    return df


def parse_reservoir_properties(filepath: str) -> Dict:
    """
    Parse a JSON or CSV file into rock and fluid properties.
    Assumes either:
    - JSON file with keys: perm, poro, fluid.oil.mu, fluid.water.mu
    - CSV file with columns: perm, poro, oil_mu, water_mu
    """
    if filepath.endswith(".json"):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data

    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        perm = df['perm'].values.reshape(-1)
        poro = df['poro'].values.reshape(-1)
        nx = int(np.cbrt(len(perm)))
        rock = {
            'perm': perm.reshape((nx, nx, nx)),
            'poro': poro.reshape((nx, nx, nx))
        }
        fluid = {
            'oil': {'mu': df['oil_mu'].iloc[0]},
            'water': {'mu': df['water_mu'].iloc[0]}
        }
        return {"rock": rock, "fluid": fluid}

    else:
        raise ValueError("Unsupported file format. Must be .json or .csv")


def parse_eclipse_output(filepath: str) -> pd.DataFrame:
    """
    Parse Eclipse simulation summary output (simplified).
    This version expects a CSV or formatted TSV conversion from Eclipse.

    Useful columns:
    - TIME, BHP, SOIL, PRESSURE, etc.
    """
    if filepath.endswith(".csv") or filepath.endswith(".tsv"):
        return pd.read_csv(filepath)
    else:
        raise ValueError("Only CSV/TSV Eclipse outputs are currently supported.")

