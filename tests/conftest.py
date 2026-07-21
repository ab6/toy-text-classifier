# tests/conftest.py
import pytest
import numpy as np
import pandas as pd

@pytest.fixture(scope="session")
def sample_raw_data():
    """Generates a reproducible dummy dataframe mimicking raw data."""
    np.random.seed(42)
    data = {
        "feature_1": np.random.randn(100),
        "feature_2": np.random.choice(["A", "B", "C"], size=100),
        "label": np.random.choice([0, 1], size=100)
    }
    return pd.DataFrame(data)
