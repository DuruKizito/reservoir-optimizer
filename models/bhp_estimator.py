# bhp_estimator.py
# models/bhp_estimator.py

import pandas as pd
from typing import Dict

class BHP_Estimator:
    def __init__(self, gravity: float = 9.81):
        """Initialize BHP estimator."""
        self.gravity = gravity
        self.results = {}

    def estimate_bhp(self,
                     surface_pressure: float,
                     depth: float,
                     fluid_density: float,
                     well_name: str,
                     gas_density: float = None) -> Dict:
        """
        Estimate Bottom-Hole Pressure using hydrostatic equation.
        If gas_density is provided, use it for gas wells.

        Args:
            surface_pressure (float): Surface pressure in Pa
            depth (float): True vertical depth (m)
            fluid_density (float): kg/m³
            well_name (str): Identifier
            gas_density (float, optional): Gas density in kg/m³

        Returns:
            Dict: Detailed BHP calculation
        """
        hydrostatic = fluid_density * self.gravity * depth
        if gas_density is not None:
            hydrostatic += gas_density * self.gravity * depth * 0.1  # weight gas effect
        bhp = surface_pressure + hydrostatic

        self.results[well_name] = {
            'well_name': well_name,
            'surface_pressure': surface_pressure,
            'depth': depth,
            'fluid_density': fluid_density,
            'gas_density': gas_density,
            'hydrostatic': hydrostatic,
            'bhp': bhp
        }
        return self.results[well_name]

    def process_well_data(self,
                          well_data: pd.DataFrame,
                          depth_column: str,
                          density_column: str,
                          pressure_column: str,
                          gas_density_column: str = None) -> pd.DataFrame:
        """
        Estimate BHP for a DataFrame of wells. Uses gas density if available.

        Args:
            well_data (pd.DataFrame): Input data
            depth_column (str): Name of depth column
            density_column (str): Name of density column
            pressure_column (str): Name of surface pressure column
            gas_density_column (str, optional): Name of gas density column

        Returns:
            pd.DataFrame: Results with BHP per well
        """
        results = []
        for _, row in well_data.iterrows():
            surface_pressure = row.get(pressure_column, 0)
            depth = row.get(depth_column, 0)
            fluid_density = row.get(density_column, 0)
            gas_density = row.get(gas_density_column, None) if gas_density_column else None
            res = self.estimate_bhp(surface_pressure, depth, fluid_density, row.get('well_name', 'well'), gas_density)
            results.append(res)
        return pd.DataFrame(results)

