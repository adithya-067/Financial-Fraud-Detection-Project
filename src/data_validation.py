"""Dataset loading and quality checks."""
from pathlib import Path
import json
import pandas as pd
from .config import DATA_PATH, FEATURES, TARGET, REPORT_DIR

def validate_frame(df: pd.DataFrame) -> dict:
    missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only non-null values 0 and 1")
    numeric = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    import numpy as np
    invalid_mask = numeric.isna() | ~np.isfinite(numeric)
    invalid = int(invalid_mask.to_numpy().sum())
    if invalid:
        raise ValueError("Feature columns contain missing, non-numeric, or infinite values")
    counts = df[TARGET].value_counts().to_dict()
    return {"rows": int(len(df)), "columns": int(len(df.columns)), "feature_count": len(FEATURES),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "fraud_percentage": float(100 * counts.get(1, 0) / len(df)),
            "duplicate_rows": int(df.duplicated().sum()), "missing_values": int(df.isna().sum().sum()),
            "dtypes": {c: str(df[c].dtype) for c in df.columns}, "invalid_feature_values": invalid}

def load_data(path: Path = DATA_PATH, save_report: bool = True) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Dataset not found at {path}. Download creditcard.csv from Kaggle and place it in the project's data/ folder.")
    df = pd.read_csv(path)
    summary = validate_frame(df)
    if save_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "data_quality.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return df
