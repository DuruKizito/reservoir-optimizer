# reservoir_model.py
# models/reservoir_model.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Tuple

class ReservoirMLModel:
    def __init__(self, grid_size: Tuple[int, int, int], upscale_factor: int = 2):
        self.grid_size = grid_size
        self.upscale_factor = upscale_factor
        self.pressure_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.saturation_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def _upscale_grid(self, grid: Dict) -> Dict:
        nx, ny, nz = self.grid_size
        return {
            'cartDims': [nx // self.upscale_factor,
                         ny // self.upscale_factor,
                         nz // self.upscale_factor]
        }

    def _upscale_rock_properties(self, rock: Dict) -> Dict:
        nx, ny, nz = self.grid_size
        perm = np.zeros((nx // self.upscale_factor, ny // self.upscale_factor, nz // self.upscale_factor))
        poro = np.zeros_like(perm)

        for i in range(0, nx, self.upscale_factor):
            for j in range(0, ny, self.upscale_factor):
                for k in range(0, nz, self.upscale_factor):
                    i_r = slice(i, min(i + self.upscale_factor, nx))
                    j_r = slice(j, min(j + self.upscale_factor, ny))
                    k_r = slice(k, min(k + self.upscale_factor, nz))

                    perm[i // self.upscale_factor, j // self.upscale_factor, k // self.upscale_factor] = \
                        1 / np.mean(1 / rock['perm'][i_r, j_r, k_r])
                    poro[i // self.upscale_factor, j // self.upscale_factor, k // self.upscale_factor] = \
                        np.mean(rock['poro'][i_r, j_r, k_r])

        return {'perm': perm, 'poro': poro}

    def _prepare_features(self, grid: Dict, rock: Dict, fluid: Dict) -> np.ndarray:
        nx, ny, nz = grid['cartDims']
        features = np.zeros((nx * ny * nz, 4))
        idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    features[idx] = [
                        rock['perm'][i, j, k],
                        rock['poro'][i, j, k],
                        fluid['oil']['mu'],
                        fluid['water']['mu']
                    ]
                    idx += 1
        return features

    def train(self, training_data: Dict) -> Dict:
        X = training_data['features']
        y_pressure = training_data['pressure']
        y_saturation = training_data['saturation']
        X_scaled = self.scaler.fit_transform(X)

        self.pressure_model.fit(X_scaled, y_pressure)
        self.saturation_model.fit(X_scaled, y_saturation)

        return {
            'pressure_r2': self.pressure_model.score(X_scaled, y_pressure),
            'saturation_r2': self.saturation_model.score(X_scaled, y_saturation)
        }

    def predict(self, grid: Dict, rock: Dict, fluid: Dict, schedule: Dict) -> Tuple[np.ndarray, np.ndarray]:
        grid_up = self._upscale_grid(grid)
        rock_up = self._upscale_rock_properties(rock)
        features = self._prepare_features(grid_up, rock_up, fluid)
        features_scaled = self.scaler.transform(features)

        p = self.pressure_model.predict(features_scaled)
        s = self.saturation_model.predict(features_scaled)

        return (p.reshape(grid_up['cartDims']),
                s.reshape(grid_up['cartDims']))

    def save_model(self, directory="saved_models"):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.pressure_model, f"{directory}/pressure_model.pkl")
        joblib.dump(self.saturation_model, f"{directory}/saturation_model.pkl")
        joblib.dump(self.scaler, f"{directory}/scaler.pkl")

    def load_model(self, directory="saved_models"):
        self.pressure_model = joblib.load(f"{directory}/pressure_model.pkl")
        self.saturation_model = joblib.load(f"{directory}/saturation_model.pkl")
        self.scaler = joblib.load(f"{directory}/scaler.pkl")

