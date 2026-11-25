"""Validated inference helpers."""
import joblib
import numpy as np
import pandas as pd
from .config import FEATURES, MODEL_DIR

def load_model(path=MODEL_DIR/"fraud_pipeline.joblib"):
    if not path.is_file(): raise FileNotFoundError("Trained model is unavailable. Place data/creditcard.csv in the project and run python -m src.train.")
    return joblib.load(path)

def validate_transaction(values):
    if not isinstance(values,dict): raise ValueError("Transaction must be an object of feature names and numeric values")
    missing=set(FEATURES)-set(values); extra=set(values)-set(FEATURES)
    if missing or extra: raise ValueError(f"Feature mismatch. Missing: {sorted(missing)}; extra: {sorted(extra)}")
    try: row={k:float(values[k]) for k in FEATURES}
    except (TypeError,ValueError): raise ValueError("All feature values must be numeric")
    if not np.isfinite(list(row.values())).all(): raise ValueError("Feature values must be finite")
    if row["Amount"] < 0: raise ValueError("Amount must be non-negative")
    return pd.DataFrame([row],columns=FEATURES)

def predict_one(values,artifact=None):
    artifact=artifact or load_model(); X=validate_transaction(values)
    probability=float(artifact["pipeline"].predict_proba(X)[0,1]); threshold=float(artifact["threshold"]); fraud=probability>=threshold
    return {"predicted_class":int(fraud),"fraud_probability":probability,"threshold":threshold,
            "label":"Potential fraud" if fraud else "Likely legitimate","explanation":"This is a model prediction for an anonymised historical dataset, not a definitive fraud determination."}
