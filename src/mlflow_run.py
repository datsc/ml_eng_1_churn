"""mlflow_run.py — MLflow orchestrator for Telco churn classifier
Calls train.py and evaluate.py functions, logs everything to MLflow.
First written: 6.4.2026
"""

import mlflow
import mlflow.sklearn

from train import load_data, preprocess, train, save_model, save_test_data
from evaluate import evaluate, load_model, get_test_data, print_report, save_metrics

# --- Hyperparameters ---
N_ESTIMATORS = 100
RANDOM_STATE_MODEL = 15
TEST_SIZE = 0.2
RANDOM_STATE_SPLIT = 60

DATA_PATH = "data/churn.csv"
MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/metrics.json"
MODELS_DIR = "models"


def run():
    mlflow.set_experiment("telco-churn")

    with mlflow.start_run():

        # --- Train ---
        df = load_data(DATA_PATH)
        df = preprocess(df)
        model, X_test, y_test = train(df)
        save_model(model, MODEL_PATH)
        save_test_data(X_test, y_test, MODELS_DIR)

        # --- Evaluate ---
        model = load_model(MODEL_PATH)
        X_test, y_test = get_test_data(MODELS_DIR)
        metrics = evaluate(model, X_test, y_test)
        print_report(metrics)
        save_metrics(metrics, METRICS_PATH)

        # --- Log params ---
        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "random_state_model": RANDOM_STATE_MODEL,
            "test_size": TEST_SIZE,
            "random_state_split": RANDOM_STATE_SPLIT,
        })

        # --- Log metrics ---
        cr = metrics["classification_report"]
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "precision_churn": round(cr["1"]["precision"], 4),
            "recall_churn": round(cr["1"]["recall"], 4),
            "f1_churn": round(cr["1"]["f1-score"], 4),
        })

        # --- Log artifacts ---
        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(METRICS_PATH)

        print("MLflow run complete.")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    run()