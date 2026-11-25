"""Evaluation helpers and report generation."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report, roc_curve, precision_recall_curve)
from .config import REPORT_DIR, PLOT_DIR

def metrics(y, probabilities, threshold):
    pred = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "roc_auc": float(roc_auc_score(y, probabilities)),
            "pr_auc": float(average_precision_score(y, probabilities)), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0, "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0}

def save_test_outputs(y, probabilities, threshold, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    REPORT_DIR.mkdir(exist_ok=True); PLOT_DIR.mkdir(exist_ok=True)
    result = metrics(y, probabilities, threshold)
    result["classification_report"] = classification_report(y, probabilities >= threshold, output_dict=True, zero_division=0)
    (REPORT_DIR / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(y, probabilities >= threshold, display_labels=["Legitimate", "Fraud"], cmap="Blues", colorbar=False)
    plt.title("Held-out test confusion matrix"); plt.tight_layout(); plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=160); plt.close()
    p, r, _ = precision_recall_curve(y, probabilities)
    plt.figure(); plt.plot(r, p); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Held-out test precision-recall curve"); plt.tight_layout(); plt.savefig(PLOT_DIR / "test_precision_recall.png", dpi=160); plt.close()
    fpr, tpr, _ = roc_curve(y, probabilities)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--",color="gray"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("Held-out test ROC curve"); plt.tight_layout(); plt.savefig(PLOT_DIR / "test_roc.png", dpi=160); plt.close()
    return result
