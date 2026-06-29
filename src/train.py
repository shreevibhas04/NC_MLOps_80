"""
train.py — Stage 3: train baseline + advanced models, track with MLflow, register best.

Trains:
  * Logistic Regression (baseline, class_weight='balanced')
  * XGBoost (advanced, scale_pos_weight for imbalance)
Both are full sklearn Pipelines (preprocessor + classifier) so the saved model is
self-contained. Experiments are logged to MLflow; the best model by validation ROC-AUC
is registered and promoted to the 'production' alias, and saved to artifacts/.

Run:  python -m src.train
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import config as cfg
from src import data_prep as dp
from src.evaluate import compute_metrics, confusion

warnings.filterwarnings("ignore")


def _pipe(preprocessor, clf):
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def train_and_log():
    X_train, X_val, X_test, y_train, y_val, y_test = dp.get_splits()
    numeric, categorical = dp.get_feature_lists(
        pd.concat([X_train, y_train.rename(cfg.TARGET)], axis=1))
    pos = int(y_train.sum()); neg = int(len(y_train) - pos)

    models = {
        "logreg_baseline": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=cfg.RANDOM_STATE),
        "xgboost_advanced": XGBClassifier(**cfg.XGB_PARAMS),
    }

    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT)

    results = []
    for name, clf in models.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = _pipe(dp.build_preprocessor(numeric, categorical), clf)
            pipe.fit(X_train, y_train)
            val_prob = pipe.predict_proba(X_val)[:, 1]
            val_metrics = compute_metrics(y_val, val_prob)

           
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.sklearn.log_model(pipe, artifact_path="model",
                                     input_example=X_val.iloc[:3])
            results.append({"name": name, "run_id": run.info.run_id,
                            "val": val_metrics, "pipe": pipe})
            print(f"[{name}] val ROC-AUC={val_metrics['roc_auc']} "
                  f"PR-AUC={val_metrics['pr_auc']} recall={val_metrics['recall']}")

    # pick best by validation ROC-AUC, evaluate on test, register + promote
    best = max(results, key=lambda r: r["val"]["roc_auc"])
    test_prob = best["pipe"].predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_prob)
    test_cm = confusion(y_test, test_prob)
    print(f"\nBEST: {best['name']} | test {test_metrics} | cm {test_cm}")

    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", cfg.REGISTERED_MODEL)

    # persist artifacts for serving / monitoring
    joblib.dump(best["pipe"], cfg.ARTIFACT_DIR / "best_model.pkl")
    (cfg.ARTIFACT_DIR / "metrics.json").write_text(json.dumps({
        "best_model": best["name"], "registered_version": mv.version,
        "validation": {r["name"]: r["val"] for r in results},
        "test_best": test_metrics, "test_confusion": test_cm,
    }, indent=2))
    # reference sample (raw features + label) for drift monitoring baseline
    ref = X_train.copy(); ref[cfg.TARGET] = y_train.values
    ref.sample(min(5000, len(ref)), random_state=cfg.RANDOM_STATE).to_csv(
        cfg.ARTIFACT_DIR / "reference_sample.csv", index=False)
    # input schema for the API
    (cfg.ARTIFACT_DIR / "input_columns.json").write_text(
        json.dumps(list(X_train.columns), indent=2))

    print(
    f"saved best_model.pkl, metrics.json, "
    f"reference_sample.csv "
    f"(registered {cfg.REGISTERED_MODEL} "
    f"v{mv.version})"
)
    return best["name"], test_metrics


if __name__ == "__main__":
    train_and_log()
