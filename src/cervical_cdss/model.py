"""Loading the trained risk model and turning its output into something readable."""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .features import FEATURES, LABELS, MODEL_DIR, assert_no_leakage

BANDS = ["Lower", "Average", "Higher"]


@dataclass
class RiskResult:
    probability: float
    band: str
    relative_to_average: float
    observed_rate_in_band: float | None
    drivers: list[tuple[str, float]]   # (label, change in probability)


class RiskModel:
    def __init__(self, model_dir=MODEL_DIR):
        self.model = joblib.load(model_dir / "cervical_risk_model.joblib")
        self.meta = json.loads((model_dir / "model_metadata.json").read_text())
        self.cuts = self.meta["risk_band_cuts"]
        self.prevalence = self.meta["training_prevalence"]
        self.medians = self.meta["dev_medians"]
        self.band_rates = {b["band"]: b["observed_rate"] for b in self.meta["risk_bands_test"]}

    def _frame(self, rows: pd.DataFrame | dict) -> pd.DataFrame:
        df = pd.DataFrame([rows]) if isinstance(rows, dict) else rows.copy()
        missing = [c for c in FEATURES if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        df = df[FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
        assert_no_leakage(df.columns)
        return df

    def predict_proba(self, rows) -> np.ndarray:
        return self.model.predict_proba(self._frame(rows))[:, 1]

    def band_of(self, p: float) -> str:
        return BANDS[int(np.digitize(p, self.cuts))]

    def explain(self, row: dict, top: int = 4) -> list[tuple[str, float]]:
        """How much each answer moves her risk compared with a typical answer.

        For each feature we swap her value for the training median and see how
        far the prediction moves. Simple, model-agnostic, and easy to explain.
        """
        base = self._frame(row)
        p0 = float(self.model.predict_proba(base)[0, 1])
        swapped = []
        for f in FEATURES:
            if pd.isna(base.at[0, f]) or base.at[0, f] == self.medians.get(f):
                continue
            alt = base.copy()
            alt.at[0, f] = self.medians[f]
            swapped.append((f, alt))
        if not swapped:
            return []
        batch = pd.concat([a for _, a in swapped], ignore_index=True)
        ps = self.model.predict_proba(batch)[:, 1]
        effects = [(LABELS[f], p0 - float(p)) for (f, _), p in zip(swapped, ps)]
        effects = [e for e in effects if abs(e[1]) >= 0.001]
        return sorted(effects, key=lambda e: abs(e[1]), reverse=True)[:top]

    def assess(self, row: dict) -> RiskResult:
        p = float(self.predict_proba(row)[0])
        band = self.band_of(p)
        return RiskResult(
            probability=p,
            band=band,
            relative_to_average=p / self.prevalence,
            observed_rate_in_band=self.band_rates.get(band),
            drivers=self.explain(row),
        )
