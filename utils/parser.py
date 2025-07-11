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
    Supports columns: time, well_name, oil_rate, water_rate, gas_rate, BHP pressure (optional), THP Pressure (optional), units.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    # Drop rows missing required columns
    required = ['time', 'well_name', 'oil_rate', 'water_rate', 'gas_rate']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=required)
    # Optional columns
    for col in ['BHP pressure', 'THP Pressure', 'Unit_oil_rate', 'Unit_gas_rate', 'Unit_water_rate', 'Unit_BHP', 'Unit_THP']:
        if col not in df.columns:
            df[col] = None
    return df


def parse_reservoir_properties(filepath: str) -> Dict:
    """
    Parse a JSON or CSV file into rock, fluid, and gas properties.
    Supports new fields and units.
    """
    if filepath.endswith(".json"):
        with open(filepath, 'r') as f:
            data = json.load(f)
        rock = data.get('rock', {})
        fluid = data.get('fluid', {})
        gas = data.get('gas', {})
        return {"rock": rock, "fluid": fluid, "gas": gas}

    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        rock, fluid, gas = {}, {}, {}
        for _, row in df.iterrows():
            prop = str(row['Property']).lower()
            val = row['Value']
            if prop in ['porosity', 'permeability', 'avg net pay', 'swi']:
                rock[prop] = val
            elif prop in ['oil viscosity', 'oil sg', 'api', 'boi', 'reservoir temperature']:
                fluid[prop] = val
            elif prop in ['gas sg', 'rsi', 'gor', 'pb', 'pi']:
                gas[prop] = val
        return {"rock": rock, "fluid": fluid, "gas": gas}
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

