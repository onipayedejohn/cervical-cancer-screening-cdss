"""Data loading, feature definitions and the leakage rules.

The feature list is deliberately short. Every input is something a nurse can ask
at a screening visit, before any test is done. That matters for two reasons:
the app stays quick to fill in, and nothing downstream of the screening visit
can sneak into the model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "risk_factors_cervical_cancer.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "Biopsy"

# Inputs the model is allowed to see (original column names in the UCI file)
FEATURES = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Num of pregnancies",
    "Smokes (years)",
    "Smokes (packs/year)",
    "Hormonal Contraceptives (years)",
    "IUD (years)",
    "STDs (number)",
    "STDs:condylomatosis",
    "STDs:syphilis",
    "STDs:HIV",
    "STDs: Number of diagnosis",
]

# Short labels for the app and the figures
LABELS = {
    "Age": "Age",
    "Number of sexual partners": "Lifetime sexual partners",
    "First sexual intercourse": "Age at first sex",
    "Num of pregnancies": "Pregnancies",
    "Smokes (years)": "Years smoking",
    "Smokes (packs/year)": "Pack-years",
    "Hormonal Contraceptives (years)": "Years on hormonal contraception",
    "IUD (years)": "Years with an IUD",
    "STDs (number)": "Number of past STIs",
    "STDs:condylomatosis": "History of genital warts",
    "STDs:syphilis": "History of syphilis",
    "STDs:HIV": "Living with HIV",
    "STDs: Number of diagnosis": "Number of STI diagnoses",
}

# Columns that must never reach the model, and why.
LEAKY_COLUMNS = {
    "Hinselmann": "colposcopy result from the same work-up as the biopsy",
    "Schiller": "Schiller's iodine test result from the same work-up",
    "Citology": "cytology result from the same work-up",
    "Dx:Cancer": "existing cancer diagnosis; handled by the rules layer instead",
    "Dx:CIN": "existing CIN diagnosis; handled by the rules layer instead",
    "Dx:HPV": "existing HPV diagnosis recorded alongside the outcome work-up",
    "Dx": "any of the diagnoses above",
}

DROPPED_FOR_QUALITY = {
    "STDs: Time since first diagnosis": "92% missing",
    "STDs: Time since last diagnosis": "92% missing",
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the UCI file. Missing answers are stored as '?' in the original."""
    return pd.read_csv(path, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows before any split.

    If the same record sits in both train and test, the test score is inflated.
    """
    out = df.drop_duplicates().reset_index(drop=True)
    out[TARGET] = out[TARGET].astype(int)
    return out


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].copy()


def assert_no_leakage(columns) -> None:
    bad = set(columns) & (set(LEAKY_COLUMNS) | {TARGET})
    if bad:
        raise AssertionError(f"Leaky columns in model input: {sorted(bad)}")
