import pytest

from cervical_cdss.rules import (
    HIVStatus,
    LastResult,
    ScreeningInput,
    Status,
    ScreenTest,
    assess,
)


def screened(age, hiv, years, test=ScreenTest.HPV_DNA, result=LastResult.NEGATIVE, negs=1, **kw):
    return ScreeningInput(
        age=age,
        hiv_status=hiv,
        ever_screened=True,
        years_since_last_screen=years,
        last_test_type=test,
        last_result=result,
        consecutive_negatives=negs,
        **kw,
    )


# Start ages
def test_general_population_under_30_not_eligible():
    r = assess(ScreeningInput(age=27, hiv_status=HIVStatus.NEGATIVE))
    assert r.status == Status.NOT_YET_ELIGIBLE
    assert r.next_due_in_years == 3


def test_hiv_positive_starts_at_25():
    r = assess(ScreeningInput(age=26, hiv_status=HIVStatus.POSITIVE))
    assert r.status == Status.DUE
    assert "Rec. 25" in r.guideline_refs


def test_unknown_hiv_uses_general_schedule_and_flags_it():
    r = assess(ScreeningInput(age=27, hiv_status=HIVStatus.UNKNOWN))
    assert r.status == Status.NOT_YET_ELIGIBLE
    assert any("HIV test" in n for n in r.notes)


def test_never_screened_in_range_is_due():
    r = assess(ScreeningInput(age=35, hiv_status=HIVStatus.NEGATIVE))
    assert r.status == Status.DUE
    assert "Rec. 7" in r.guideline_refs  # priority age group


# Symptoms override everything, including age
@pytest.mark.parametrize("age", [22, 40, 70])
def test_symptoms_always_refer(age):
    r = assess(ScreeningInput(age=age, symptoms=["postcoital_bleeding"]))
    assert r.status == Status.REFER


def test_symptoms_override_can_stop():
    r = assess(screened(60, HIVStatus.NEGATIVE, 2, negs=3, symptoms=["postmenopausal_bleeding"]))
    assert r.status == Status.REFER


# Intervals
def test_hpv_negative_general_within_5_years_not_due():
    r = assess(screened(40, HIVStatus.NEGATIVE, 4))
    assert r.status == Status.NOT_DUE
    assert r.next_due_in_years == 1


def test_hpv_negative_general_at_5_years_due():
    assert assess(screened(40, HIVStatus.NEGATIVE, 5)).status == Status.DUE


def test_hpv_negative_hiv_positive_due_at_3_years():
    r = assess(screened(40, HIVStatus.POSITIVE, 3))
    assert r.status == Status.DUE
    assert "Rec. 28" in r.guideline_refs


def test_via_interval_is_3_years():
    assert assess(screened(40, HIVStatus.NEGATIVE, 2, test=ScreenTest.VIA)).status == Status.NOT_DUE
    assert assess(screened(40, HIVStatus.NEGATIVE, 3, test=ScreenTest.VIA)).status == Status.DUE


def test_overdue_headline():
    r = assess(screened(45, HIVStatus.NEGATIVE, 12))
    assert r.status == Status.DUE and r.headline.startswith("Overdue")


# Stopping
def test_over_50_two_negatives_can_stop():
    r = assess(screened(55, HIVStatus.NEGATIVE, 3, negs=2))
    assert r.status == Status.CAN_STOP


def test_over_50_one_negative_cannot_stop():
    assert assess(screened(55, HIVStatus.NEGATIVE, 3, negs=1)).status == Status.NOT_DUE


def test_exactly_50_does_not_stop():
    assert assess(screened(50, HIVStatus.NEGATIVE, 3, negs=2)).status != Status.CAN_STOP


def test_two_negatives_but_lapsed_schedule_does_not_stop():
    r = assess(screened(60, HIVStatus.NEGATIVE, 14, negs=2))
    assert r.status == Status.DUE


# Management pathways
def test_unmanaged_positive_goes_to_follow_up():
    r = assess(screened(35, HIVStatus.NEGATIVE, 0.2, result=LastResult.POSITIVE_AWAITING, negs=0))
    assert r.status == Status.FOLLOW_UP


def test_triage_negative_retest_general_2_years():
    base = dict(result=LastResult.POSITIVE_TRIAGE_NEGATIVE, negs=0)
    assert assess(screened(35, HIVStatus.NEGATIVE, 1, **base)).status == Status.NOT_DUE
    assert assess(screened(35, HIVStatus.NEGATIVE, 2, **base)).status == Status.DUE


def test_triage_negative_retest_hiv_1_year():
    base = dict(result=LastResult.POSITIVE_TRIAGE_NEGATIVE, negs=0)
    assert assess(screened(35, HIVStatus.POSITIVE, 1, **base)).status == Status.DUE


def test_post_treatment_retest_at_one_year():
    base = dict(result=LastResult.TREATED, negs=0)
    assert assess(screened(35, HIVStatus.NEGATIVE, 0.5, **base)).status == Status.FOLLOW_UP
    assert assess(screened(35, HIVStatus.NEGATIVE, 1, **base)).status == Status.DUE


def test_known_cancer_is_not_screening():
    assert assess(ScreeningInput(age=45, known_cervical_cancer=True)).status == Status.FOLLOW_UP


# Input validation
def test_screened_without_details_is_rejected():
    with pytest.raises(ValueError):
        assess(ScreeningInput(age=40, ever_screened=True))


def test_unknown_symptom_code_is_rejected():
    with pytest.raises(ValueError):
        assess(ScreeningInput(age=40, symptoms=["headache"]))


def test_every_decision_explains_itself():
    cases = [
        ScreeningInput(age=20),
        ScreeningInput(age=35),
        screened(40, HIVStatus.NEGATIVE, 2),
        screened(60, HIVStatus.NEGATIVE, 2, negs=2),
        ScreeningInput(age=40, symptoms=["pelvic_pain"]),
    ]
    for c in cases:
        r = assess(c)
        assert r.headline and r.reason and r.next_step
