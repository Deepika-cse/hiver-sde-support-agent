from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def escalation_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    false_auto = sum(bool(t) and not bool(p_) for t, p_ in zip(y_true, y_pred))
    positive = sum(bool(t) for t in y_true)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "false_auto_handle_count": int(false_auto),
        "false_auto_handle_rate": float(false_auto / positive) if positive else 0.0,
    }


def expected_calibration_error(y_true, confidence, bins=10):
    y_true = np.asarray(y_true, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            acc = y_true[mask].mean()
            conf = confidence[mask].mean()
            ece += mask.mean() * abs(acc - conf)
    return float(ece)
