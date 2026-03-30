import os
import sys
import tempfile
import json

import pytest
from sklearn.ensemble import GradientBoostingClassifier
 
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src/")))
 
from evaluate import evaluate, load_model, get_test_data, save_metrics
from train import load_data, preprocess, train, save_model, save_test_data
 
# Train once at module level for all tests
_df = preprocess(load_data("data/churn.csv"))
_model, _X_test, _y_test = train(_df)
_tmpdir = tempfile.mkdtemp()
save_model(_model, os.path.join(_tmpdir, "model.pkl"))
save_test_data(_X_test, _y_test, dir=_tmpdir)
_metrics = evaluate(_model, _X_test, _y_test)
 
 
def test_load_model_returns_classifier():
    assert isinstance(load_model(os.path.join(_tmpdir, "model.pkl")), GradientBoostingClassifier)
 
def test_load_model_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_model("models/nonexistent.pkl")
 
def test_get_test_data_shapes():
    X, y = get_test_data(dir=_tmpdir)
    assert len(X) == len(y) > 0
 
def test_get_test_data_binary():
    _, y = get_test_data(dir=_tmpdir)
    assert set(y.unique()).issubset({0, 1})
 
def test_accuracy_in_range():
    assert 0.0 <= _metrics["accuracy"] <= 1.0
 
def test_roc_auc_better_than_random():
    assert _metrics["roc_auc"] > 0.5
 
def test_confusion_matrix_sums_to_test_size():
    cm = _metrics["confusion_matrix"]
    assert cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] == len(_y_test)
 
def test_classification_report_has_both_classes():
    assert "0" in _metrics["classification_report"] and "1" in _metrics["classification_report"]
 
def test_save_metrics_creates_valid_json():
    path = os.path.join(_tmpdir, "metrics.json")
    save_metrics(_metrics, path)
    with open(path) as f:
        loaded = json.load(f)
    assert "accuracy" in loaded and "roc_auc" in loaded