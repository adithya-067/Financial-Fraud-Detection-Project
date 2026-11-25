import numpy as np
import pytest
from src.config import FEATURES
from src.predict import validate_transaction, predict_one

class FakePipeline:
    def predict_proba(self,X): return np.array([[.8,.2]])

def test_validation_and_output():
    row={k:0 for k in FEATURES}; X=validate_transaction(row)
    assert list(X.columns)==FEATURES
    result=predict_one(row,{"pipeline":FakePipeline(),"threshold":.3})
    assert result["predicted_class"]==0 and result["fraud_probability"]==.2
    assert "not a definitive" in result["explanation"]

@pytest.mark.parametrize("changes",[{"V1":None},{"V1":float("nan")},{"Amount":-1},{"unexpected":3}])
def test_bad_input_rejected(changes):
    row={k:0 for k in FEATURES}; row.update(changes)
    with pytest.raises(ValueError): validate_transaction(row)

def test_missing_or_extra_feature_rejected():
    with pytest.raises(ValueError): validate_transaction({k:0 for k in FEATURES[:-1]})
