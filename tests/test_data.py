# tests/test_data.py
# import pytest
# import numpy as np
# from src.data import preprocess_features

# def test_preprocess_features_output_shape(sample_raw_data):
#     """Ensure categorical encoding doesn't break shape expectations."""
#     X, y = preprocess_features(sample_raw_data)
    
#     # Assertions for dimensions
#     assert X.shape[0] == 100
#     assert X.shape[1] > 1  # Verify one-hot encoding expanded columns
#     assert len(y) == 100

# def test_preprocess_features_no_nan(sample_raw_data):
#     """Verify that preprocessing eliminates or flags null fields."""
#     X, _ = preprocess_features(sample_raw_data)
    
#     assert not np.isnan(X).any()

