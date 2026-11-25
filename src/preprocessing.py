"""Leakage-safe feature preprocessing."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from .config import FEATURES

def make_preprocessor():
    # Amount is non-negative in this dataset; log1p reduces its long right tail.
    return ColumnTransformer([("amount", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())]), ["Amount"]),
                              ("other", StandardScaler(), [c for c in FEATURES if c != "Amount"])], remainder="drop")
