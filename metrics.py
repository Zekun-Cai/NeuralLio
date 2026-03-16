import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error


def compute_classification_metrics(y_true, log_probs, average="macro"):
    y_true = np.concatenate(y_true)
    log_probs = np.concatenate(log_probs)
    # probs = np.exp(log_probs)
    y_pred = np.argmax(log_probs, axis=1)

    metrics = {}
    metrics["acc"] = accuracy_score(y_true, y_pred)
    metrics["err"] = 1.0 - metrics["acc"]
    try:
        metrics["f1"] = f1_score(y_true, y_pred, average=average)
    except Exception:
        metrics["f1"] = None

    metrics["auc"] = None
    return metrics


def compute_regression_metrics(y_true, y_pred):
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    metrics = {}
    metrics["mae"] = np.mean(np.abs(y_true - y_pred))
    metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    return metrics
