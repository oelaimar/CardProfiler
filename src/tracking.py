import mlflow
import pandas as pd
from mlflow import set_experiment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.extraction import PROJECT_ROOT

EXPERIMENT_NAME = "classification_clients"
TRACKING_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"

def setup_mlflow(experiment_name: str = EXPERIMENT_NAME) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACTS_DIR.as_uri())
    mlflow,set_experiment(experiment_name)

def _clean_param_name(name: str) -> str:
    return name.replace("classifier__estimator__", "").replace("classifier__", "")

def compute_metrics(y_true, y_pred) -> dict:
    return{
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

def log_runs(pipelines: dict, X_train: pd.DataFrame, y_test:pd.Series, best_params: dict | None = None, extra_params: dict | None = None) -> pd.DataFrame:
    rows = []
    for name, pipeline in pipelines.items():
        with mlflow.start_run(run_name=name) as run:
            # Params
            mlflow.log_param("model", name)
            for key, value in (extra_params or {}).items():
                mlflow.log_param(key, value)
            if best_params and name in best_params:
                for key, value in best_params[name].items():
                    mlflow.log_param(_clean_param_name(key), value)

            # Metrics
            metrics = compute_metrics(y_test, pipeline.predict(X_train))
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            rows.append({"model" : name, "run_id" : run.info.run_id, **metrics})
    return pd.DataFrame(rows).set_index("model").round(4)
