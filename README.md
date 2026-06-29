# Hospital Readmission Prediction — MLOps Reference Solution

End-to-end MLOps pipeline predicting 30-day hospital readmission (binary) on the
Diabetes 130-US Hospitals dataset: EDA → feature engineering → model development
→ MLflow tracking & registry → FastAPI serving → Docker → CI → drift monitoring →
retraining → logging/governance.

> **Guided healthcare MLOps.** This solution is built specifically around the Diabetes 130-US
> Hospitals dataset; several decisions (patient dedup, hospice removal, ICD-9 grouping, medication
> features) are dataset-specific. The *lifecycle* is transferable; the *data choices* are not.

## Prerequisites
- Python 3.11+
- `data/diabetic_data.csv` (included)
- On Mac, `brew install libomp` for XGBoost

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run order
```bash
python -m src.train                  # train LogReg + XGBoost, log to MLflow, register best (@production)
python -m src.generate_current_batch # simulate a drifted production batch
python -m src.monitoring             # Evidently drift report + prediction PSI
python -m src.retrain                # apply retraining criteria, retrain if triggered
pytest tests/ -q                     # API smoke tests (same as CI)
uvicorn app:app --reload             # serve the model  (POST /predict, GET /health)
```
The three notebooks orchestrate the same code and **carry the assessed evidence**:
```bash
jupyter nbconvert --to notebook --execute --inplace Data_Preparation.ipynb
jupyter nbconvert --to notebook --execute --inplace Model_Development_and_Tracking.ipynb
jupyter nbconvert --to notebook --execute --inplace Operations_Monitoring_and_Evidence.ipynb
```

## Submission model (platform-friendly — files only)
Assessment is from **uploaded files**: no ZIP, no GitHub URL. Every rubric criterion is graded from:
the **MLOps report** (PDF/DOCX), the **three executed notebooks**, and **`app.py`**. Repository-only
artefacts (Docker image, CI run, MLflow UI, drift HTML, prediction log) are **evidenced inside the
operations notebook / report** via listings, code output, screenshots and summaries.

## Repository map — stage · task · which TODO to complete · where the evidence goes
| File | Stage / Tasks | Role | Provided or learner-built (TODOs) | Evidence appears in (submitted) |
|---|---|---|---|---|
| `Data_Preparation.ipynb` | 1.2–1.3, 2.1–2.4 | EDA + leakage-safe preparation | **Learner builds** | itself (submitted notebook) |
| `Model_Development_and_Tracking.ipynb` | 3.1–3.5 | train, track, register, promote | **Learner builds** | itself (submitted notebook) |
| `Operations_Monitoring_and_Evidence.ipynb` | 4.1–4.6 | serve, container/CI evidence, monitor, retrain, govern | **Learner builds** | itself (submitted notebook) |
| `Hospital_Readmission_MLOps_Report` (→ PDF) | 1.1, 1.4, 3.5, 4.2–4.6 | narrative + Stage-4 screenshots/summaries | **Learner builds** | itself (submitted report) |
| `app.py` | 4.1 | FastAPI inference service | **Learner builds** (TODOs) | itself + Ops notebook API demo |
| `src/data_prep.py` | 2.1–2.4 | clean / FE / preprocess / split | **Learner builds** (TODOs) | via `Data_Preparation.ipynb` |
| `src/train.py` | 3.1–3.5 | train + MLflow + registry | **Learner builds** (TODOs) | via `Model_Development_and_Tracking.ipynb` |
| `src/monitoring.py` | 4.4 | Evidently feature drift + prediction PSI | **Learner builds** (TODOs) | via `Operations…ipynb` |
| `src/retrain.py` | 4.5 | multi-signal trigger + retrain workflow | **Learner builds** (TODOs) | via `Operations…ipynb` |
| `config.py` | all | central config (paths, target, columns, params) | **Provided** | — (support) |
| `src/evaluate.py` | 3.4 | imbalance-aware metrics | **Provided** | via Stage-3 notebook |
| `src/generate_current_batch.py` | 4.4 | simulate a drifted batch | **Provided** | via `Operations…ipynb` |
| `tests/test_app.py` | 4.3 | API smoke tests (run by CI) | **Provided** | via Ops notebook pytest output |
| `Dockerfile` | 4.2 | container image | **Provided** | listed + explained in Ops notebook/report |
| `.github/workflows/ci.yml` | 4.3 | GitHub Actions CI | **Provided** | listed + explained in Ops notebook/report |
| `artifacts/` | 3–4 | model, metrics, drift, logs, registry inputs | generated at runtime | summarised in notebooks/report |

## Example request
```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"features": {"age": 65, "time_in_hospital": 5, "num_medications": 18,
       "number_inpatient": 1, "diag_1": "Circulatory", "insulin": "Up", ... }}'
# -> {"readmission_probability": 0.57, "readmitted_30d": 1, "threshold": 0.5}
```

## Notes
- Target prevalence ~9% → models selected on **ROC-AUC**, recall watched; accuracy is misleading.
- Deterministic (`random_state=42`); idempotent re-runs; all local + open-source (no API keys).
- Reference run: best = LogReg, test ROC-AUC ~0.64; drift 6/39 features, prediction PSI ~0.46 →
  retrain triggers → registry v1 → v2 (@production).
