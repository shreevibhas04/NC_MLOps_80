# REVIEW.md — MLOps Capstone 1 (Hospital Readmission)

QA self-audit of the reference solution. Verified end-to-end on the project venv.

## 0. Client-review changes applied (revision 2)
This revision addresses the client review (`client-feedback/MLOps Capstone 1 Review.docx`):
- **Positioning:** "dataset-agnostic" removed → **guided healthcare MLOps** (overview, report, guide, README).
- **4-week stage plan** added to the overview (objectives / tasks / deliverables / assessment / dependencies).
- **Submission model = files only** (report PDF/DOCX + 3 notebooks + `app.py`); **no ZIP, no GitHub URL**.
- **Stage-4 evidence moved into a new `Operations_Monitoring_and_Evidence.ipynb`** + the report
  (API demo, Docker/CI listings + pytest output, drift summary + plot, retraining decision, prediction
  log, registry history) — nothing is graded from a repository.
- **Rubric rebuilt** in the standard PredMaint format (Level 1/L1…/Level 3/Set Marks/Grading Guidelines)
  with explicit 100% / 50% / 0% per L3; every row names a **submittable file**; numbers are representative.
- **README repository map** added (file → stage/task → provided-or-TODO → where evidence appears).
- **Leakage wording** aligned to *split first, then fit preprocessing on train only*.
- **Monitoring/retraining** docs expanded (reference dataset, cadence, report generation, post-drift
  action, multi-signal trigger, representative thresholds).
- **Starter** expanded to 3 notebooks + `generate_current_batch.py`; descriptive TODOs.
- **Defects fixed:** EDA missing-value cell now uses `raw.isna()` (was empty); `predictions.log` is
  populated by the operations notebook; retraining shows a clean **v1 → v2** promotion.

## 1. Dataset & target
- Source: Diabetes 130-US Hospitals (`diabetic_data.csv`), 101,766 rows × 50 cols.
- Target: `readmitted == '<30'` → 1 (30-day readmission), else 0.
- After cleaning (first encounter per patient + drop expired/hospice): **69,973 rows, ~8.97% positive**.

## 2. Deliberate decisions (each is a teaching point)
- **First encounter per patient** kept → independent rows (prevents the same patient appearing in
  train & test = leakage). Drops ~30k duplicate-patient encounters.
- **Drop expired/hospice discharges** (disposition 11,13,14,19,20,21) → those patients cannot be
  readmitted; keeping them would create invalid negatives (label leakage).
- **Drop `weight`** (~97% missing), `payer_code` (administrative), and constant medication columns.
- **Feature engineering:** ICD9 `diag_1/2/3` → clinical groups; `age` → midpoint; `service_utilization`;
  `num_med_changes`; high-cardinality `medical_specialty` → top-10 + Other.
- **Imbalance handled** via `class_weight='balanced'` (LogReg) and `scale_pos_weight` (XGBoost), not SMOTE.
- **Leakage-safe preprocessing**: split first; ColumnTransformer fit on TRAIN only, inside the Pipeline.

## 3. Verified results (reference run)
- LogReg val ROC-AUC **0.654** / PR-AUC 0.171 / recall 0.543; XGBoost val ROC-AUC **0.653** / PR-AUC 0.178.
- Best (LogReg) **test**: ROC-AUC **0.644**, PR-AUC 0.165, recall 0.515, precision 0.136, accuracy 0.664;
  confusion (tn 8650, fp 4090, fn 609, tp 646).
- Registered `readmission_classifier`; **v1 → v2** promoted to the **`production`** alias after retraining.
- Drift: **6/39 features drifted**, dataset_drift=False, **prediction PSI 0.457** → retraining triggered.
- API: 3/3 pytest pass; all three notebooks executed with outputs.

> ROC-AUC ~0.65 is the **honest, expected** ceiling for this dataset (published baselines sit ~0.64-0.68).
> This is intentionally NOT inflated — readmission is hard, which is itself a learning outcome.

## 4. Known issues — status
| # | Severity | Location | Issue | Status |
|---|---|---|---|---|
| 1 | HIGH | XGBoost import | needs OpenMP (`libomp`) on Mac | RESOLVED — documented; CI uses Linux libgomp1 |
| 2 | MED | tests | NaN in JSON payload rejected by httpx (`allow_nan=False`) | RESOLVED — test sends None for missing; API imputes |
| 3 | MED | EDA notebook | missing-value table empty (`=='?'` after na_values) | RESOLVED — uses `raw.isna()` with percentages |
| 4 | LOW | predictions.log | shipped empty | RESOLVED — operations notebook logs predictions |
| 5 | LOW | mlflow | `artifact_path` deprecation warning (mlflow 3.x) | ACCEPTED — cosmetic |
| 6 | LOW | urllib3 | LibreSSL warning on macOS system Python | ACCEPTED — cosmetic |

## 5. Autograder / reproducibility checklist
| Requirement | Status |
|---|---|
| Deterministic (`random_state=42`) | ✓ |
| No internet/API key at runtime | ✓ (local CSV, local MLflow file store) |
| Outputs saved as files + evidenced in notebooks | ✓ |
| Idempotent / re-runnable | ✓ |
| Pinned dependencies | ✓ requirements.txt |
| Tests pass in CI | ✓ pytest 3/3 |
| All rubric criteria gradable from submitted files (no repo/ZIP/HTML/JSON) | ✓ |

## 6. What a reviewer should check
1. `python -m src.train` → reproduces ROC-AUC ~0.65, registers v1 @production.
2. `python -m src.generate_current_batch && python -m src.monitoring` → drift_report.html + PSI > 0.2.
3. `python -m src.retrain` → triggers, registers v2 @production, writes retraining_decision.json.
4. `pytest tests/ -q` → 3 passed.
5. All three notebooks run top-to-bottom and contain outputs; the operations notebook carries the
   Stage-4 evidence (API demo, Docker/CI listings + pytest, drift plot, retraining decision, log, registry).
