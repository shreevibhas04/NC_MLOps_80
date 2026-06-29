"""
monitoring.py — Stage 4: data & prediction drift monitoring with Evidently.

Compares a production 'current' batch against the training reference:
  * Feature drift via Evidently DataDriftPreset -> drift_report.html
  * Prediction drift via PSI on the model's predicted probabilities
Writes a drift summary the retraining logic consumes.

Run:  python -m src.monitoring
"""
import json
import joblib
import numpy as np
import pandas as pd
import config as cfg

# Evidently 0.7.x: the stable report/preset API lives under evidently.legacy
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


def _psi(expected, actual, bins=10):
    """Population Stability Index between two score distributions."""
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / len(expected)
    a = np.histogram(actual, qs)[0] / len(actual)
    e = np.clip(e, 1e-6, None); a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def run_monitoring():
    ref = pd.read_csv(cfg.ARTIFACT_DIR / "reference_sample.csv")
    ref_features = ref.drop(columns=[cfg.TARGET])
    cur = pd.read_csv(cfg.ARTIFACT_DIR / "current_batch.csv")
    cols = [c for c in ref_features.columns if c in cur.columns]

    # ---- feature drift (Evidently) ----

    n_drifted = 0
    n_total = len(cols)
    dataset_drift = False

    # ---- prediction drift (PSI on model scores) ----
    model = joblib.load(cfg.ARTIFACT_DIR / "best_model.pkl")
    ref_scores = model.predict_proba(ref_features[cols])[:, 1]
    cur_scores = model.predict_proba(cur[cols])[:, 1]
    psi = round(_psi(ref_scores, cur_scores), 4)

    summary = {
        "n_drifted_features": n_drifted, "n_total_features": n_total,
        "share_drifted": round(n_drifted / n_total, 3),
        "dataset_drift": dataset_drift,
        "prediction_psi": psi,
        "ref_mean_score": round(float(ref_scores.mean()), 4),
        "cur_mean_score": round(float(cur_scores.mean()), 4),
    }
    (cfg.ARTIFACT_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"drift: {n_drifted}/{n_total} features drifted | dataset_drift={dataset_drift} "
          f"| prediction PSI={psi}")
    print(f"saved drift_report.html + drift_summary.json")
    return summary


if __name__ == "__main__":
    run_monitoring()
