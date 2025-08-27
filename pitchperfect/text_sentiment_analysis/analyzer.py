"""
analyzer.py — Output utilities for MELD emotion classifier

Responsibilities:
- Evaluate a trained model on arrays (loss/accuracy)
- Predict class probabilities and labels
- Manual classification report (precision/recall/F1/support) without sklearn
- Confusion matrix computation
- Pretty-printers and optional JSON/CSV savers
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import json
import csv
import numpy as np


# -----------------------------
# Core evaluation & predictions
# -----------------------------
def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32
) -> Dict[str, float]:
    """Return dict with loss and accuracy from model.evaluate()."""
    loss, acc = model.evaluate(X, y, batch_size=batch_size, verbose=0)
    return {"loss": float(loss), "accuracy": float(acc)}


def predict_labels_and_probs(
    model,
    X: np.ndarray,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_pred_labels, y_pred_probs).
    - y_pred_labels: shape (N,), int labels via argmax
    - y_pred_probs:  shape (N, C), softmax probabilities
    """
    probs = model.predict(X, batch_size=batch_size, verbose=0)
    labels = np.argmax(probs, axis=1)
    return labels, probs


# -----------------------------
# Metrics (manual, no sklearn)
# -----------------------------
def classification_report_manual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Per-class metrics and macro/weighted averages:
      precision, recall, f1-score, support
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num_classes = len(class_names)
    report: Dict[str, Dict[str, float]] = {}

    supports = []
    precisions = []
    recalls = []
    f1s = []

    for i, name in enumerate(class_names):
        tp = int(np.sum((y_true == i) & (y_pred == i)))
        fp = int(np.sum((y_true != i) & (y_pred == i)))
        fn = int(np.sum((y_true == i) & (y_pred != i)))
        support = int(np.sum(y_true == i))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        report[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1),
            "support": support,
        }

        supports.append(support)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    total = int(np.sum(supports)) if supports else 0
    weights = np.asarray(supports, dtype=float) / total if total > 0 else np.zeros(num_classes, dtype=float)

    macro_avg = {
        "precision": float(np.mean(precisions) if precisions else 0.0),
        "recall": float(np.mean(recalls) if recalls else 0.0),
        "f1-score": float(np.mean(f1s) if f1s else 0.0),
        "support": total,
    }
    weighted_avg = {
        "precision": float(np.sum(weights * precisions) if precisions else 0.0),
        "recall": float(np.sum(weights * recalls) if recalls else 0.0),
        "f1-score": float(np.sum(weights * f1s) if f1s else 0.0),
        "support": total,
    }

    report["macro avg"] = macro_avg
    report["weighted avg"] = weighted_avg
    return report


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Compute confusion matrix M where M[i, j] = count of true=i predicted=j.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1 if y_true.size else 0
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# -----------------------------
# Pretty printers
# -----------------------------
def print_classification_report(
    report: Dict[str, Dict[str, float]],
    header: str = "CLASSIFICATION REPORT"
) -> None:
    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)
    for name, m in report.items():
        print(f"{name:<14} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {int(m['support']):>10d}")


def print_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str]
) -> None:
    """
    Prints a simple text table for the confusion matrix.
    Rows = true labels, Cols = predicted labels.
    """
    n = len(class_names)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = " " * 16 + " ".join([f"{name[:12]:>12}" for name in class_names])
    print(header)
    for i in range(n):
        row_counts = " ".join([f"{cm[i, j]:>12d}" for j in range(n)])
        print(f"{class_names[i][:14]:>14}  {row_counts}")


def print_overall_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else 0.0
    print(f"\nOverall Accuracy: {acc:.4f}")
    return acc


# -----------------------------
# Save helpers (optional)
# -----------------------------
def save_report_json(report: Dict[str, Dict[str, float]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report JSON to: {path}")


def save_confusion_matrix_csv(cm: np.ndarray, class_names: List[str], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for i, name in enumerate(class_names):
            writer.writerow([name] + list(map(int, cm[i].tolist())))
    print(f"Saved confusion matrix CSV to: {path}")


# -----------------------------
# One-shot analysis wrapper
# -----------------------------
def analyze_and_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    batch_size: int = 32,
    save_json_path: Optional[str] = None,
    save_cm_csv_path: Optional[str] = None
) -> Dict[str, object]:
    """
    Convenience wrapper:
      - evaluate
      - predict
      - classification report
      - confusion matrix
      - print everything
      - optionally save to disk
    Returns a dict with all artifacts.
    """
    eval_dict = evaluate_model(model, X_test, y_test, batch_size=batch_size)
    y_pred, y_probs = predict_labels_and_probs(model, X_test, batch_size=batch_size)

    report = classification_report_manual(y_test, y_pred, class_names)
    cm = confusion_matrix(y_test, y_pred, num_classes=len(class_names))

    # Pretty prints
    print(f"\nTest Loss: {eval_dict['loss']:.4f}  |  Test Accuracy: {eval_dict['accuracy']:.4f}")
    print_overall_accuracy(y_test, y_pred)
    print_classification_report(report)
    print_confusion_matrix(cm, class_names)

    # Optional saves
    if save_json_path:
        save_report_json(report, save_json_path)
    if save_cm_csv_path:
        save_confusion_matrix_csv(cm, class_names, save_cm_csv_path)

    return {
        "eval": eval_dict,
        "y_pred": y_pred,
        "y_prob": y_probs,
        "report": report,
        "confusion_matrix": cm,
    }
