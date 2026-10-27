import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(y_true, scores, threshold: float = 0.5) -> dict:
    """Metrics at one threshold. Label 1 = churned."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {scores.shape}")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both classes in y_true to compute ROC/PR")

    pred = scores >= threshold
    churned, stayed = y_true == 1, y_true == 0
    tp = int((pred & churned).sum())
    fn = int((~pred & churned).sum())
    fp = int((pred & stayed).sum())
    tn = int((~pred & stayed).sum())
    return {
        "threshold": float(threshold),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "accuracy": (tp + tn) / len(y_true),
        "missed_churners": fn,
        "wasted_offers": fp,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
    }
