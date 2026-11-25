import pandas as pd
import pytest
from src.config import FEATURES
from src.data_validation import validate_frame, load_data

def frame(): return pd.DataFrame([{**{c:0.0 for c in FEATURES},"Class":0},{**{c:1.0 for c in FEATURES},"Class":1}])

def test_validation_summary():
    summary=validate_frame(frame())
    assert summary["rows"]==2 and summary["class_counts"]=={"0":1,"1":1}

def test_missing_column_rejected():
    with pytest.raises(ValueError,match="Missing required columns"):
        validate_frame(frame().drop(columns=["V1"]))

def test_invalid_feature_rejected():
    data=frame(); data.loc[0,"Amount"]=float("inf")
    with pytest.raises(ValueError,match="finite"):
        validate_frame(data)

def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError,match="creditcard.csv"):
        load_data(tmp_path/"missing.csv",save_report=False)
