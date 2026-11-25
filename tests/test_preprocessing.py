import numpy as np
import pandas as pd
from src.config import FEATURES
from src.preprocessing import make_preprocessor

def test_preprocessor_shape_and_finite():
    X=pd.DataFrame([{c:float(i+1) for i,c in enumerate(FEATURES)},{c:float(i+2) for i,c in enumerate(FEATURES)}])
    transformed=make_preprocessor().fit_transform(X)
    assert transformed.shape==(2,len(FEATURES))
    assert np.isfinite(transformed).all()
