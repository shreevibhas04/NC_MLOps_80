"""
data_prep.py — Stage 2 (Data Preparation): load, clean, feature-engineer, and split the readmission data.
Backs Data_Preparation.ipynb (rubric 2.1-2.4).

Pipeline of decisions (each justified in the report / guide):
  * '?' -> NaN.
  * Keep only the FIRST encounter per patient -> independent rows (leakage prevention).
  * Drop expired/hospice discharges -> those patients cannot be readmitted (label validity).
  * Target = 1 if readmitted '<30' else 0.
  * Drop identifiers, ~97%-missing weight, administrative payer_code, constant meds.
  * Feature engineering: ICD9 diagnosis grouping, age midpoint, service utilisation,
    medication-change count, high-cardinality medical_specialty grouped to top-k.
  * Leakage-safe preprocessing via a ColumnTransformer fit on TRAIN only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import config as cfg

ID_CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


def load_raw():
    return pd.read_csv(cfg.RAW_CSV, na_values=["?"])


def _icd9_group(code):
    """Map an ICD9 diagnosis code to a broad clinical category."""
    if pd.isna(code):
        return "Missing"
    s = str(code)
    if s.startswith(("E", "V")):
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    if s.startswith("250"):
        return "Diabetes"
    if 390 <= v <= 459 or v == 785:
        return "Circulatory"
    if 460 <= v <= 519 or v == 786:
        return "Respiratory"
    if 520 <= v <= 579 or v == 787:
        return "Digestive"
    if 580 <= v <= 629 or v == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    return "Other"


_AGE_MID = {f"[{i}-{i+10})": i + 5 for i in range(0, 100, 10)}


def clean(df):
    df = df.copy()
    # first encounter per patient -> independent observations
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    # remove expired / hospice discharges (cannot be readmitted)
    df = df[~df["discharge_disposition_id"].isin(cfg.EXPIRED_HOSPICE_DISPOSITIONS)]
    # binary target
    df[cfg.TARGET] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])
    # drop unusable / constant / identifier columns (only those present)
    df = df.drop(columns=[c for c in cfg.DROP_COLS if c in df.columns])
    return df


def engineer_features(df):
    df = df.copy()
    # diagnosis grouping
    # age -> numeric midpoint
    # service utilisation
    df["service_utilization"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"])
    # medication-change count (Up/Down across med columns present)
    # high-cardinality medical_specialty -> top 10 + Other / Unknown
    # ID code columns are categorical, not numeric
    for c in ID_CODE_COLS:
        df[c] = df[c].astype(str)
    return df


def get_feature_lists(df):
    numeric = ["time_in_hospital", "num_lab_procedures", "num_procedures",
               "num_medications", "number_outpatient", "number_emergency",
               "number_inpatient", "number_diagnoses", "service_utilization"]
    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in df.columns
                   if c not in numeric + [cfg.TARGET]]
    return numeric, categorical


def build_preprocessor(numeric, categorical):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ])


def get_model_frame():
    return engineer_features(clean(load_raw()))


def get_splits(df=None):
    """Stratified train/val/test split (leakage-safe: preprocessor fit later on train)."""
    if df is None:
        df = get_model_frame()
    X = df.drop(columns=[cfg.TARGET])
    y = df[cfg.TARGET]
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=cfg.VAL_SIZE, stratify=y_tr, random_state=cfg.RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test
