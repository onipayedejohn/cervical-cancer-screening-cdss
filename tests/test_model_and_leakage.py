import json

import numpy as np
import pandas as pd
import pytest

from cervical_cdss.features import FEATURES, LEAKY_COLUMNS, MODEL_DIR, PROCESSED_DIR, TARGET, assert_no_leakage
from cervical_cdss.model import RiskModel

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "cervical_risk_model.joblib").exists(), reason="run scripts/train_evaluate.py first"
)


@pytest.fixture(scope="module")
def model():
    return RiskModel()


def test_feature_list_has_no_leaky_columns():
    assert not set(FEATURES) & set(LEAKY_COLUMNS)
    assert TARGET not in FEATURES


def test_leak_guard_raises():
    with pytest.raises(AssertionError):
        assert_no_leakage(FEATURES + ["Schiller"])


def test_dev_and_test_do_not_share_rows():
    dev = pd.read_csv(PROCESSED_DIR / "dev.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    key = lambda d: set(map(tuple, d.fillna(-1).values.tolist()))
    assert not key(dev) & key(test)


def test_model_only_knows_the_allowed_features(model):
    fitted = list(model.model.calibrated_classifiers_[0].estimator.feature_names_in_)
    assert fitted == FEATURES


def test_probabilities_are_valid(model):
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    p = model.predict_proba(test[FEATURES])
    assert np.all((p >= 0) & (p <= 1))


def test_missing_answers_are_handled(model):
    p = model.predict_proba({"Age": 35})
    assert 0 <= p[0] <= 1


def test_band_rates_increase(model):
    rates = [b["observed_rate"] for b in model.meta["risk_bands_test"]]
    assert rates == sorted(rates)


def test_explanations_are_labelled(model):
    row = {"Age": 45, "Number of sexual partners": 8, "First sexual intercourse": 14,
           "Hormonal Contraceptives (years)": 12}
    r = model.assess(row)
    assert r.band in {"Lower", "Average", "Higher"}
    for label, delta in r.drivers:
        assert isinstance(label, str) and isinstance(delta, float)


def test_metadata_records_versions():
    meta = json.loads((MODEL_DIR / "model_metadata.json").read_text())
    assert "sklearn_version" in meta and meta["threshold"] > 0
