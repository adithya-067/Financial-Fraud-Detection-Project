"""Threshold selection from validation predictions only."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def threshold_table(y_true, probabilities):
    rows = []
    for threshold in np.unique(np.r_[np.linspace(0.01, 0.99, 99), 0.5]):
        labels = probabilities >= threshold
        rows.append({"threshold": float(threshold), "precision": precision_score(y_true, labels, zero_division=0),
                     "recall": recall_score(y_true, labels, zero_division=0), "f1": f1_score(y_true, labels, zero_division=0)})
    return pd.DataFrame(rows)

def select_threshold(y_true, probabilities, min_recall=0.80):
    table = threshold_table(y_true, probabilities)
    eligible = table[table.recall >= min_recall]
    # Among thresholds meeting the recall target, maximize precision; tie-break by F1.
    choice = (eligible.sort_values(["precision", "f1", "threshold"], ascending=[False, False, False]).iloc[0]
              if not eligible.empty else table.sort_values(["f1", "recall"], ascending=False).iloc[0])
    return float(choice.threshold), table
