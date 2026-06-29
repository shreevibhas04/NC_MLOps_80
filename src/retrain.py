"""
retrain.py — Stage 4: retraining trigger + workflow.

Reads the drift summary and applies retraining criteria. If any trigger fires, it
retrains (registering a new model version via src.train) and records the decision.
In production the retrain would use freshly-labelled data; here it re-runs the
training pipeline to demonstrate the automated workflow and versioning.

Run:  python -m src.retrain
"""
import json
import config as cfg
from src.train import train_and_log

PSI_THRESHOLD = 0.20            # prediction drift
DRIFT_SHARE_THRESHOLD = 0.30   # share of features drifted


def decide(summary):
    reasons = []
    if summary["prediction_psi"] > PSI_THRESHOLD:
        reasons.append(f"prediction PSI {summary['prediction_psi']} > {PSI_THRESHOLD}")
    return reasons


def run_retraining_workflow():
    summary = json.loads((cfg.ARTIFACT_DIR / "drift_summary.json").read_text())
    reasons = decide(summary)
    decision = {"retrain_triggered": bool(reasons), "reasons": reasons,
                "criteria": {"psi_threshold": PSI_THRESHOLD,
                             "drift_share_threshold": DRIFT_SHARE_THRESHOLD},
                "drift_summary": summary}
    if reasons:
        print("Retraining TRIGGERED ->", "; ".join(reasons))
        name, test_metrics = train_and_log()
        decision["retrained_model"] = name
        decision["new_test_metrics"] = test_metrics
    else:
        print("No retraining needed — drift within thresholds.")
    (cfg.ARTIFACT_DIR / "retraining_decision.json").write_text(json.dumps(decision, indent=2))
    print("saved retraining_decision.json")
    return decision


if __name__ == "__main__":
    run_retraining_workflow()
