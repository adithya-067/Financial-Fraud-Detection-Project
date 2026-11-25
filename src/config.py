"""Central project paths and reproducibility settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
PLOT_DIR = ROOT / "visualizations"
RANDOM_STATE = 42
TARGET = "Class"
FEATURES = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
for _path in (MODEL_DIR, REPORT_DIR, PLOT_DIR):
    _path.mkdir(parents=True, exist_ok=True)
