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
                     well_name: str) -> Dict:
        """
        Estimate Bottom-Hole Pressure using hydrostatic equation.

        Args:
            surface_pressure (float): Surface pressure in Pa
            depth (float): True vertical depth (m)
            fluid_density (float): kg/m³
            well_name (str): Identifier

        Returns:
            Dict: Detailed BHP calculation
        """
        hydrostatic = fluid_density * self.gravity * depth
        bhp = surface_pressure + hydrostatic

        self.results[well_name] = {
            'well_name': well_name,
            'surface_pressure': surface_pressure,
            'depth': depth,
            'fluid_density': fluid_density,
            'hydrostatic': hydrostatic,
            'bhp': bhp
        }
        return self.results[well_name]

    def process_well_data(self,
                          well_data: pd.DataFrame,
                          depth_column: str,
                          density_column: str,
                          pressure_column: str) -> pd.DataFrame:
        """
        Estimate BHP for a DataFrame of wells.

        Args:
            well_data (pd.DataFrame): Input data
            depth_column (str): Name of depth column
            density_column (str): Name of density column
            pressure_column (str): Name of surface pressure column

        Returns:
            pd.DataFrame: Results with BHP per well
        """
        results = []
        for _, row in well_data.iterrows():
            result = self.estimate_bhp(
                surface_pressure=row[pressure_column],
                depth=row[depth_column],
                fluid_density=row[density_column],
                well_name=row.get("well_name", f"Well-{_}")
            )
            results.append(result)
        return pd.DataFrame(results)

