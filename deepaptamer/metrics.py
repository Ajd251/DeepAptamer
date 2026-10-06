"""Held-out performance, shared by the command line and the app."""
from sklearn import metrics

COLUMNS = ["model", "AUROC", "AUPRC", "precision", "recall", "F1", "MCC"]


def compute(y_true, scores, threshold=0.5):
    """scores maps a model label to its probabilities. Returns one row of COLUMNS per model."""
    rows = []
    for label, p in scores.items():
        called = p > threshold
        rows.append({
            "model": label,
            "AUROC": metrics.roc_auc_score(y_true, p),
            "AUPRC": metrics.average_precision_score(y_true, p),
            "precision": metrics.precision_score(y_true, called, zero_division=0),
            "recall": metrics.recall_score(y_true, called, zero_division=0),
            "F1": metrics.f1_score(y_true, called, zero_division=0),
            "MCC": metrics.matthews_corrcoef(y_true, called),
        })
    return rows


def curves(y_true, scores):
    """Points for the ROC and precision-recall curves, as plain lists per model."""
    out = {}
    for label, p in scores.items():
        fpr, tpr, _ = metrics.roc_curve(y_true, p)
        precision, recall, _ = metrics.precision_recall_curve(y_true, p)
        out[label] = {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}
    return out
