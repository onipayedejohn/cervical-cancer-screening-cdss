"""Guideline layer: WHO 2021 cervical screening recommendations as explicit rules.

Everything here is plain if/else logic on purpose. A clinician should be able to
read this file, check it against the guideline, and disagree with a specific line.
The ML model never overrides what this module decides.

Source: WHO guideline for screening and treatment of cervical pre-cancer lesions
for cervical cancer prevention, 2nd edition (2021).
https://www.ncbi.nlm.nih.gov/books/NBK572318/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    REFER = "refer"                  # symptoms: diagnostic work-up, not screening
    FOLLOW_UP = "follow_up"          # inside a management pathway already
    DUE = "due"                      # offer a screening test now
    NOT_DUE = "not_due"              # screened recently, come back later
    NOT_YET_ELIGIBLE = "not_yet_eligible"
    CAN_STOP = "can_stop"            # over 50 with two negatives in a row


class HIVStatus(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class ScreenTest(str, Enum):
    HPV_DNA = "hpv_dna"
    VIA = "via"
    CYTOLOGY = "cytology"


class LastResult(str, Enum):
    NEGATIVE = "negative"
    POSITIVE_AWAITING = "positive_awaiting"            # positive, not yet managed
    POSITIVE_TRIAGE_NEGATIVE = "positive_triage_negative"  # HPV+ but triage test negative
    TREATED = "treated"                                # treated for CIN2/3


RED_FLAG_SYMPTOMS = {
    "postcoital_bleeding": "bleeding after sex",
    "intermenstrual_bleeding": "bleeding between periods",
    "postmenopausal_bleeding": "bleeding after menopause",
    "abnormal_discharge": "persistent, unusual vaginal discharge",
    "pelvic_pain": "persistent pelvic or lower back pain",
}

# Start ages (Rec. 5 and Rec. 25)
START_AGE = {HIVStatus.NEGATIVE: 30, HIVStatus.POSITIVE: 25}

# Regular intervals in years as (earliest, latest) (Rec. 8, 9, 28, 29).
# We call someone "due" once the earliest point has passed. Being conservative
# here costs a test; being late costs a missed lesion.
INTERVALS = {
    (HIVStatus.NEGATIVE, ScreenTest.HPV_DNA): (5, 10),
    (HIVStatus.POSITIVE, ScreenTest.HPV_DNA): (3, 5),
    (HIVStatus.NEGATIVE, ScreenTest.VIA): (3, 3),
    (HIVStatus.POSITIVE, ScreenTest.VIA): (3, 3),
    (HIVStatus.NEGATIVE, ScreenTest.CYTOLOGY): (3, 3),
    (HIVStatus.POSITIVE, ScreenTest.CYTOLOGY): (3, 3),
}

# Retest after a positive HPV test with a negative triage test (Rec. 11 and 31)
TRIAGE_NEGATIVE_RETEST = {HIVStatus.NEGATIVE: 2, HIVStatus.POSITIVE: 1}

# Retest after treatment for CIN2/3 (Rec. 13 and 33)
POST_TREATMENT_RETEST = 1


@dataclass
class ScreeningInput:
    age: int
    hiv_status: HIVStatus = HIVStatus.UNKNOWN
    symptoms: list[str] = field(default_factory=list)
    ever_screened: bool = False
    years_since_last_screen: float | None = None
    last_test_type: ScreenTest | None = None
    last_result: LastResult | None = None
    consecutive_negatives: int = 0
    known_cervical_cancer: bool = False


@dataclass
class Recommendation:
    status: Status
    headline: str
    reason: str
    next_step: str
    guideline_refs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    next_due_in_years: float | None = None


def _schedule_group(hiv: HIVStatus) -> HIVStatus:
    # Unknown status follows the general schedule; we flag it separately.
    return HIVStatus.POSITIVE if hiv == HIVStatus.POSITIVE else HIVStatus.NEGATIVE


def _population_label(group: HIVStatus) -> str:
    return "women living with HIV" if group == HIVStatus.POSITIVE else "the general population"


def validate(inp: ScreeningInput) -> list[str]:
    """Return a list of problems with the input. Empty list means usable."""
    problems = []
    if not 10 <= inp.age <= 100:
        problems.append("Age should be between 10 and 100.")
    unknown = set(inp.symptoms) - set(RED_FLAG_SYMPTOMS)
    if unknown:
        problems.append(f"Unrecognised symptom codes: {sorted(unknown)}")
    if inp.ever_screened:
        if inp.years_since_last_screen is None or inp.years_since_last_screen < 0:
            problems.append("Enter how many years ago the last screen was.")
        if inp.last_test_type is None:
            problems.append("Select the type of the last screening test.")
        if inp.last_result is None:
            problems.append("Select the result of the last screening test.")
        if inp.consecutive_negatives < 0:
            problems.append("Consecutive negatives cannot be negative.")
    return problems


def assess(inp: ScreeningInput) -> Recommendation:
    """Apply the rules in priority order. The first rule that matches wins."""
    problems = validate(inp)
    if problems:
        raise ValueError(" ".join(problems))

    group = _schedule_group(inp.hiv_status)
    start_age = START_AGE[group]
    notes: list[str] = []

    if inp.hiv_status == HIVStatus.UNKNOWN:
        notes.append(
            "HIV status is unknown, so the general schedule is used. Offering an HIV "
            "test matters here: a positive result moves her to an earlier, more frequent schedule."
        )

    # 1. Known cancer or symptoms come first. Screening is for people without symptoms.
    if inp.known_cervical_cancer:
        return Recommendation(
            Status.FOLLOW_UP,
            "Under cancer care, not screening",
            "She already has a cervical cancer diagnosis.",
            "Continue care with the treating team. Routine screening does not apply.",
            notes=notes,
        )

    if inp.symptoms:
        described = ", ".join(RED_FLAG_SYMPTOMS[s] for s in inp.symptoms)
        return Recommendation(
            Status.REFER,
            "Refer for clinical assessment",
            f"She reports {described}. Symptoms call for a diagnostic work-up, "
            "and a screening test is not the right tool for that.",
            "Refer for speculum examination and diagnostic evaluation, whatever her screening history.",
            notes=notes,
        )

    # 2. Open management pathways
    if inp.ever_screened and inp.last_result == LastResult.POSITIVE_AWAITING:
        return Recommendation(
            Status.FOLLOW_UP,
            "Complete follow-up of a positive screen",
            "Her last screening result was positive and has not yet been managed.",
            "Arrange triage or treatment according to the local screen-and-treat protocol.",
            notes=notes,
        )

    if inp.ever_screened and inp.last_result == LastResult.TREATED:
        wait = POST_TREATMENT_RETEST
        ref = "Rec. 33" if group == HIVStatus.POSITIVE else "Rec. 13"
        years = inp.years_since_last_screen or 0
        if years >= wait:
            return Recommendation(
                Status.DUE,
                "Post-treatment retest is due",
                f"She was treated for a pre-cancer lesion {years:g} year(s) ago. WHO advises an "
                f"HPV DNA retest {wait} year after treatment.",
                "Offer an HPV DNA test now.",
                [ref],
                notes,
            )
        return Recommendation(
            Status.FOLLOW_UP,
            "Post-treatment retest not yet due",
            f"She was treated {years:g} year(s) ago. The retest is scheduled {wait} year after treatment.",
            "Book an HPV DNA retest at the 12-month mark.",
            [ref],
            notes,
            next_due_in_years=round(wait - years, 1),
        )

    # 3. Too young for routine screening
    if inp.age < start_age:
        ref = "Rec. 25" if group == HIVStatus.POSITIVE else "Rec. 5"
        return Recommendation(
            Status.NOT_YET_ELIGIBLE,
            "Not yet in the screening age range",
            f"Routine screening for {_population_label(group)} starts at {start_age}. She is {inp.age}.",
            f"No screening test needed now. Plan the first screen at age {start_age}.",
            [ref],
            notes,
            next_due_in_years=float(start_age - inp.age),
        )

    # 4. Never screened and in range: due
    priority_top = 49
    in_priority = start_age <= inp.age <= priority_top
    priority_ref = "Rec. 27" if group == HIVStatus.POSITIVE else "Rec. 7"
    if in_priority:
        notes.append(
            f"She is in the WHO priority age group ({start_age} to {priority_top}) for {_population_label(group)}."
        )

    if not inp.ever_screened:
        return Recommendation(
            Status.DUE,
            "Due for a first screen",
            f"She is {inp.age} and has never been screened.",
            "Offer HPV DNA testing, the preferred primary test. Use VIA or cytology only where HPV testing is unavailable.",
            ["Rec. 1", "Rec. 25" if group == HIVStatus.POSITIVE else "Rec. 5"]
            + ([priority_ref] if in_priority else []),
            notes,
        )

    years = float(inp.years_since_last_screen)

    # 5. HPV positive, triage negative: shortened retest
    if inp.last_result == LastResult.POSITIVE_TRIAGE_NEGATIVE:
        wait = TRIAGE_NEGATIVE_RETEST[group]
        ref = "Rec. 31" if group == HIVStatus.POSITIVE else "Rec. 11"
        if years >= wait:
            return Recommendation(
                Status.DUE,
                "Retest after negative triage is due",
                f"Her last HPV test was positive with a negative triage test {years:g} year(s) ago. "
                f"The retest interval for {_population_label(group)} is {wait} year(s).",
                "Offer an HPV DNA retest now.",
                [ref],
                notes,
            )
        return Recommendation(
            Status.NOT_DUE,
            "Retest scheduled",
            f"HPV positive with a negative triage test {years:g} year(s) ago. "
            f"Retest is due at {wait} year(s).",
            "Book the retest.",
            [ref],
            notes,
            next_due_in_years=round(wait - years, 1),
        )

    # 6. Over 50 with two consecutive negatives on schedule: can stop
    earliest, latest = INTERVALS[(group, inp.last_test_type)]
    on_schedule = years <= latest
    if inp.age > 50 and inp.consecutive_negatives >= 2 and on_schedule:
        return Recommendation(
            Status.CAN_STOP,
            "Screening can stop",
            f"She is over 50 with {inp.consecutive_negatives} consecutive negative results "
            "at regular intervals.",
            "No further routine screening needed unless symptoms develop.",
            ["Rec. 26" if group == HIVStatus.POSITIVE else "Rec. 6"],
            notes,
        )

    # 7. Regular interval check
    interval_ref = {
        (HIVStatus.NEGATIVE, ScreenTest.HPV_DNA): "Rec. 8",
        (HIVStatus.POSITIVE, ScreenTest.HPV_DNA): "Rec. 28",
    }.get((group, inp.last_test_type), "Rec. 29" if group == HIVStatus.POSITIVE else "Rec. 9")
    window = f"{earliest} to {latest}" if earliest != latest else f"{earliest}"
    test_name = {ScreenTest.HPV_DNA: "HPV DNA", ScreenTest.VIA: "VIA", ScreenTest.CYTOLOGY: "cytology"}[inp.last_test_type]

    if years >= earliest:
        overdue = years > latest
        return Recommendation(
            Status.DUE,
            "Overdue for screening" if overdue else "Due for screening",
            f"Her last negative {test_name} test was {years:g} year(s) ago. The interval for "
            f"{_population_label(group)} after {test_name} is {window} years.",
            "Offer an HPV DNA test now.",
            [interval_ref],
            notes,
        )

    return Recommendation(
        Status.NOT_DUE,
        "Up to date",
        f"Her last negative {test_name} test was {years:g} year(s) ago, inside the "
        f"{window}-year interval for {_population_label(group)}.",
        "No test needed today. Remind her of the next screening date.",
        [interval_ref],
        notes,
        next_due_in_years=round(earliest - years, 1),
    )
