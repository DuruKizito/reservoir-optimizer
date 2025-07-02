# test_model.py
# tests/test_model.py

import numpy as np
from models.reservoir_model import ReservoirMLModel

def test_prediction_shapes():
    model = ReservoirMLModel((10, 10, 10))
    rock = {
        'perm': np.random.rand(10, 10, 10),
        'poro': np.random.rand(10, 10, 10)
    }
    fluid = {'oil': {'mu': 1.0}, 'water': {'mu': 1.0}}
    grid = {'cartDims': [10, 10, 10]}
    pressure, saturation = model.predict(grid, rock, fluid, {})
    assert pressure.shape == (5, 5, 5)
    assert saturation.shape == (5, 5, 5)

