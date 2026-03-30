"""evaluate.py — Model evaluation for Telco churn classifier
First written: 30.3.2026
"""
 
import json
import os
import pickle
import json
 
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
 
 
def load_model(path="models/model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
 
 
def get_test_data(dir="models"):
    X_test = pd.read_csv(os.path.join(dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(dir, "y_test.csv")).squeeze()
    return X_test, y_test
 
 
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
 
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }
    return metrics
 
 
def print_report(metrics):
    print("=" * 50)
    print("Model Evaluation Report")
    print("=" * 50)
    print(f"Accuracy : {metrics['accuracy']}")
    print(f"ROC-AUC  : {metrics['roc_auc']}")
    print()
    print("Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print()
    print("Classification Report:")
    cr = metrics["classification_report"]
    for label in ["0", "1"]:
        r = cr[label]
        print(
            f"  Class {label} — "
            f"precision: {r['precision']:.4f}  "
            f"recall: {r['recall']:.4f}  "
            f"f1: {r['f1-score']:.4f}  "
            f"support: {int(r['support'])}"
        )
    print("=" * 50)
 
 
def save_metrics(metrics, path="models/metrics.json"):
    os.makedirs("models", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")
 
 
if __name__ == "__main__":
    model = load_model()
    X_test, y_test = get_test_data()
    metrics = evaluate(model, X_test, y_test)
    print_report(metrics)
    save_metrics(metrics)
 