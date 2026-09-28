"""Shared glue between the form/CSV inputs, the rules and the model.

The app and the batch mode both go through these functions, so a single
patient and a spreadsheet of patients are handled exactly the same way.
"""

from __future__ import annotations

import math

import pandas as pd

from .rules import HIVStatus, LastResult, ScreenTest, ScreeningInput, Status, assess

# Friendly CSV column -> original model feature
RISK_COLUMNS = {
    "sexual_partners": "Number of sexual partners",
    "age_first_sex": "First sexual intercourse",
    "pregnancies": "Num of pregnancies",
    "smoking_years": "Smokes (years)",
    "pack_years": "Smokes (packs/year)",
    "hormonal_contraception_years": "Hormonal Contraceptives (years)",
    "iud_years": "IUD (years)",
    "sti_count": "STDs (number)",
    "genital_warts": "STDs:condylomatosis",
    "syphilis": "STDs:syphilis",
}

RULE_COLUMNS = [
    "age", "hiv_status", "symptoms", "known_cervical_cancer", "ever_screened",
    "years_since_last_screen", "last_test_type", "last_result", "consecutive_negatives",
]

TEMPLATE_COLUMNS = ["patient_id"] + RULE_COLUMNS + list(RISK_COLUMNS)


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def model_row(age, hiv: HIVStatus, risk: dict) -> dict:
    """Build the model's input row. Blank answers stay missing on purpose."""
    row = {"Age": age}
    for friendly, feature in RISK_COLUMNS.items():
        v = risk.get(friendly)
        row[feature] = float("nan") if _blank(v) else float(v)
    row["STDs:HIV"] = {HIVStatus.POSITIVE: 1.0, HIVStatus.NEGATIVE: 0.0}.get(hiv, float("nan"))
    row["STDs: Number of diagnosis"] = row["STDs (number)"]
    return row


def screening_input_from_record(rec: dict) -> ScreeningInput:
    ever = str(rec.get("ever_screened", "0")).strip().lower() in {"1", "true", "yes", "y"}
    symptoms = [] if _blank(rec.get("symptoms")) else [
        s.strip() for s in str(rec["symptoms"]).split(";") if s.strip()
    ]
    hiv_raw = "unknown" if _blank(rec.get("hiv_status")) else str(rec["hiv_status"]).strip().lower()
    return ScreeningInput(
        age=int(float(rec["age"])),
        hiv_status=HIVStatus(hiv_raw),
        symptoms=symptoms,
        ever_screened=ever,
        years_since_last_screen=None if _blank(rec.get("years_since_last_screen")) else float(rec["years_since_last_screen"]),
        last_test_type=None if _blank(rec.get("last_test_type")) else ScreenTest(str(rec["last_test_type"]).strip().lower()),
        last_result=None if _blank(rec.get("last_result")) else LastResult(str(rec["last_result"]).strip().lower()),
        consecutive_negatives=0 if _blank(rec.get("consecutive_negatives")) else int(float(rec["consecutive_negatives"])),
        known_cervical_cancer=str(rec.get("known_cervical_cancer", "0")).strip().lower() in {"1", "true", "yes", "y"},
    )


def run_batch(df: pd.DataFrame, risk_model) -> pd.DataFrame:
    missing = [c for c in ["age"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    out = []
    for i, rec in enumerate(df.to_dict("records")):
        pid = rec.get("patient_id", i + 1)
        try:
            inp = screening_input_from_record(rec)
            decision = assess(inp)
            risk = risk_model.assess(model_row(inp.age, inp.hiv_status, rec))
            out.append({
                "patient_id": pid,
                "decision": decision.headline,
                "status": decision.status.value,
                "next_step": decision.next_step,
                "who_refs": ", ".join(decision.guideline_refs),
                "risk_estimate": round(risk.probability, 4),
                "risk_band": risk.band,
                "risk_applies": decision.status not in (Status.REFER, Status.FOLLOW_UP),
                "error": "",
            })
        except (ValueError, KeyError) as e:
            out.append({"patient_id": pid, "decision": "", "status": "error", "next_step": "",
                        "who_refs": "", "risk_estimate": None, "risk_band": "", "risk_applies": False,
                        "error": str(e)})
    return pd.DataFrame(out)
