"""Cervical Screening Check: Streamlit front end.

Run locally:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cervical_cdss.batch import TEMPLATE_COLUMNS, model_row, run_batch  # noqa: E402
from cervical_cdss.model import RiskModel  # noqa: E402
from cervical_cdss.rules import (  # noqa: E402
    RED_FLAG_SYMPTOMS,
    HIVStatus,
    LastResult,
    ScreenTest,
    ScreeningInput,
    Status,
    assess,
)
import ui  # noqa: E402

st.set_page_config(page_title="Cervical Screening Check", page_icon="🩺", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject_css()


@st.cache_resource
def load_model():
    return RiskModel()


@st.cache_data
def load_table(name):
    return pd.read_csv(ROOT / "reports" / "metrics" / name)


risk_model = load_model()
meta = risk_model.meta

ui.header()
tab_check, tab_batch, tab_model, tab_about = st.tabs(
    ["Screening check", "Batch review", "How the model performs", "Disclaimer and method"]
)

TEST_LABELS = {ScreenTest.HPV_DNA: "HPV DNA test", ScreenTest.VIA: "VIA", ScreenTest.CYTOLOGY: "Cytology (Pap)"}
RESULT_LABELS = {
    LastResult.NEGATIVE: "Negative",
    LastResult.POSITIVE_AWAITING: "Positive, not yet managed",
    LastResult.POSITIVE_TRIAGE_NEGATIVE: "HPV positive, triage negative",
    LastResult.TREATED: "Treated for a pre-cancer lesion",
}

def render_results(inp, risk_answers):
    try:
        rec = assess(inp)
    except ValueError as e:
        st.error(str(e))
        return

    st.markdown(ui.rule_card(rec), unsafe_allow_html=True)

    risk = risk_model.assess(model_row(inp.age, inp.hiv_status, risk_answers))
    applies = rec.status not in (Status.REFER, Status.FOLLOW_UP)
    if applies:
        st.markdown(ui.risk_card(risk, risk_model.cuts, applies), unsafe_allow_html=True)
    else:
        with st.expander("Model risk estimate (context only)"):
            st.markdown(ui.risk_card(risk, risk_model.cuts, applies), unsafe_allow_html=True)

    summary = (
        "Cervical Screening Check (prototype)\n"
        f"Age {inp.age}, HIV status {inp.hiv_status.value}\n"
        f"Decision: {rec.headline}\nReason: {rec.reason}\nNext step: {rec.next_step}\n"
        f"WHO 2021 references: {', '.join(rec.guideline_refs) or 'n/a'}\n"
        f"Model estimate: {risk.probability:.1%} ({risk.band} band)\n"
        "Not validated for clinical use."
    )
    st.download_button("Download summary", summary, file_name="screening_summary.txt",
                       use_container_width=True)


# ---------------------------------------------------------------- single patient
with tab_check:
    left, right = st.columns([5, 7], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<div class="section">Today\'s visit</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            age = c1.number_input("Age (years)", min_value=10, max_value=100, value=35, step=1)
            hiv = c2.selectbox("HIV status", list(HIVStatus), index=1,
                               format_func=lambda h: h.value.capitalize())
            symptoms = st.multiselect(
                "Symptoms reported today", list(RED_FLAG_SYMPTOMS),
                format_func=lambda s: RED_FLAG_SYMPTOMS[s].capitalize(),
                placeholder="None",
            )
            known_cancer = st.checkbox("Already diagnosed with cervical cancer")

            st.markdown('<div class="section">Screening history</div>', unsafe_allow_html=True)
            ever = st.radio("Screened before?", ["No", "Yes"], horizontal=True) == "Yes"
            c3, c4 = st.columns(2)
            years = c3.number_input("Years since last screen", min_value=0.0, max_value=60.0,
                                    value=3.0, step=0.5, disabled=not ever)
            test = c4.selectbox("Last test", list(ScreenTest), format_func=TEST_LABELS.get, disabled=not ever)
            c5, c6 = st.columns(2)
            result = c5.selectbox("Last result", list(LastResult), format_func=RESULT_LABELS.get,
                                  disabled=not ever)
            negs = c6.number_input("Negatives in a row", min_value=0, max_value=20, value=1, step=1,
                                   disabled=not ever)

            with st.expander("Risk profile for the model (optional)"):
                st.markdown('<div class="hint">Leave a field empty if she prefers not to answer. '
                            'The model was trained to handle missing answers.</div>', unsafe_allow_html=True)
                r1, r2 = st.columns(2)
                partners = r1.number_input("Lifetime sexual partners", min_value=0, max_value=50, value=None)
                first_sex = r2.number_input("Age at first sex", min_value=9, max_value=60, value=None)
                r3, r4 = st.columns(2)
                pregnancies = r3.number_input("Pregnancies", min_value=0, max_value=20, value=None)
                hc_years = r4.number_input("Years on hormonal contraception", min_value=0.0, max_value=40.0,
                                           value=None, step=0.5)
                r5, r6 = st.columns(2)
                smoke_years = r5.number_input("Years smoking", min_value=0.0, max_value=60.0, value=None, step=1.0)
                pack_years = r6.number_input("Pack-years", min_value=0.0, max_value=60.0, value=None, step=0.5)
                r7, r8 = st.columns(2)
                iud_years = r7.number_input("Years with an IUD", min_value=0.0, max_value=30.0, value=None, step=0.5)
                sti_count = r8.number_input("Number of past STIs", min_value=0, max_value=10, value=None)
                r9, r10 = st.columns(2)
                warts = r9.checkbox("History of genital warts")
                syphilis = r10.checkbox("History of syphilis")

    with right:
        risk_answers = {
            "sexual_partners": partners, "age_first_sex": first_sex, "pregnancies": pregnancies,
            "smoking_years": smoke_years, "pack_years": pack_years,
            "hormonal_contraception_years": hc_years, "iud_years": iud_years,
            "sti_count": sti_count,
            "genital_warts": 1 if warts else (None if sti_count is None else 0),
            "syphilis": 1 if syphilis else (None if sti_count is None else 0),
        }
        inp = ScreeningInput(
            age=int(age), hiv_status=hiv, symptoms=symptoms, ever_screened=ever,
            years_since_last_screen=years if ever else None,
            last_test_type=test if ever else None,
            last_result=result if ever else None,
            consecutive_negatives=int(negs) if ever else 0,
            known_cervical_cancer=known_cancer,
        )
        render_results(inp, risk_answers)

# ---------------------------------------------------------------- batch
with tab_batch:
    st.markdown(
        """
        <div class="card">
          <div class="card-label">Batch review</div>
          <h3>Check a clinic list in one go</h3>
          <p>Upload a CSV with one woman per row. Each row goes through the same rules and model as the
          single check. The example file contains ten made-up patients so you can see the format.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2)
    d1.download_button("Download blank template", ",".join(TEMPLATE_COLUMNS) + "\n",
                       file_name="batch_template.csv", use_container_width=True)
    d2.download_button("Download example (synthetic)", (ROOT / "data/sample/batch_example.csv").read_text(),
                       file_name="batch_example.csv", use_container_width=True)
    with st.expander("Accepted codes"):
        st.markdown(
            "- **hiv_status**: positive, negative, unknown\n"
            f"- **symptoms**: separate with a semicolon: {', '.join(RED_FLAG_SYMPTOMS)}\n"
            "- **last_test_type**: hpv_dna, via, cytology\n"
            "- **last_result**: negative, positive_awaiting, positive_triage_negative, treated\n"
            "- **ever_screened**, **known_cervical_cancer**, **genital_warts**, **syphilis**: 1 or 0\n"
            "- Risk profile columns can be left empty."
        )
    upload = st.file_uploader("Upload CSV", type=["csv"])
    use_example = st.toggle("Use the synthetic example instead")
    frame = None
    if upload is not None:
        frame = pd.read_csv(upload)
    elif use_example:
        frame = pd.read_csv(ROOT / "data/sample/batch_example.csv")

    if frame is not None:
        if len(frame) > 5000:
            st.error("Please upload 5,000 rows or fewer.")
        else:
            try:
                res = run_batch(frame, risk_model)
            except ValueError as e:
                st.error(str(e))
            else:
                counts = res["status"].value_counts()
                tiles = "".join(
                    ui.tile(int(counts.get(k, 0)), ui.STATUS_STYLE[k][0])
                    for k in ["refer", "follow_up", "due", "not_due"]
                )
                st.markdown(f'<div class="tiles">{tiles}</div>', unsafe_allow_html=True)
                if (res["status"] == "error").any():
                    st.warning(f"{(res['status'] == 'error').sum()} row(s) could not be read. See the error column.")
                shown = res.assign(status=res["status"].map(
                    lambda s: ui.STATUS_STYLE[s][0] if s in ui.STATUS_STYLE else "Error"))
                st.dataframe(
                    shown, hide_index=True, use_container_width=True,
                    column_config={
                        "risk_estimate": st.column_config.ProgressColumn(
                            "Risk estimate", format="%.3f", min_value=0.0, max_value=0.2),
                        "risk_applies": st.column_config.CheckboxColumn("Risk relevant"),
                    },
                )
                st.download_button("Download results", res.to_csv(index=False),
                                   file_name="screening_results.csv", use_container_width=True)

# ---------------------------------------------------------------- performance
with tab_model:
    tm, ci = meta["test_metrics"], meta["test_ci_95"]

    def rng(k):
        return f"95% CI {ci[k][0]:.2f} to {ci[k][1]:.2f}"

    st.markdown(
        f"""
        <div class="card">
          <div class="card-label">Honest summary</div>
          <h3>A modest model, measured without leakage</h3>
          <p>The model only sees answers a nurse can collect before any test is done. On a locked
          test set of {meta['n_test']} women ({meta['positives_test']} positive biopsies) it ranks risk
          better than chance, but not by a wide margin. That is why the guideline rules make the
          decisions and the model only adds context.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tiles = "".join([
        ui.tile(f"{tm['roc_auc']:.2f}", "ROC-AUC", rng("roc_auc")),
        ui.tile(f"{tm['pr_auc']:.2f}", "PR-AUC", f"no-skill level {tm['prevalence']:.2f}"),
        ui.tile(f"{tm['sensitivity']:.0%}", "Sensitivity at threshold", rng("sensitivity")),
        ui.tile(f"{tm['npv']:.0%}", "Negative predictive value", rng("npv")),
    ])
    st.markdown(f'<div class="tiles">{tiles}</div>', unsafe_allow_html=True)

    bands = pd.DataFrame(meta["risk_bands_test"])
    bands["observed_rate"] = (bands["observed_rate"] * 100).round(1).astype(str) + "%"
    bands.columns = ["Risk band", "Women", "Positive biopsies", "Observed rate"]
    st.markdown("##### Risk bands on the test set")
    st.dataframe(bands, hide_index=True, use_container_width=True)

    st.markdown("##### Five models compared, including their energy use")
    comp = load_table("model_comparison.csv")
    view = pd.DataFrame({
        "Model": comp["model"],
        "PR-AUC (mean)": comp["pr_auc_mean"].round(3),
        "ROC-AUC (mean)": comp["roc_auc_mean"].round(3),
        "Energy, nested CV (Wh)": comp["cv_energy_wh"].round(4),
        "CO₂e, nested CV (g)": comp["cv_co2_g"].round(4),
        "Scoring 1,000 women (ms)": comp["inference_ms_per_1000"].round(1),
        "Model size (KB)": comp["model_size_kb"].round(0).astype(int),
    })
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.caption("Energy measured with CodeCarbon using Ghana's grid carbon intensity. On a cloud VM without "
               "hardware power counters it estimates CPU power, so compare the models with each other rather "
               "than reading the absolute numbers literally.")

    fig = ROOT / "reports" / "figures"
    g1, g2 = st.columns(2, gap="large")
    g1.image(str(fig / "model_comparison.png"), use_container_width=True)
    g2.image(str(fig / "green_vs_performance.png"), use_container_width=True)
    g3, g4 = st.columns(2, gap="large")
    g3.image(str(fig / "test_roc_pr.png"), use_container_width=True)
    g4.image(str(fig / "feature_importance.png"), use_container_width=True)
    g5, g6 = st.columns(2, gap="large")
    g5.image(str(fig / "calibration.png"), use_container_width=True)
    g6.image(str(fig / "decision_curve.png"), use_container_width=True)

    leak = meta["leakage_demo"]
    st.markdown(
        f"""
        <div class="card">
          <div class="card-label">Why many published scores on this dataset look much higher</div>
          <p>If I let the same kind of model see the colposcopy, Schiller and cytology results, its test
          ROC-AUC jumps to <b>{leak['with_leaky_columns_roc_auc']:.2f}</b>. Those results come from the
          same work-up as the biopsy, so they are not available when a screening decision is made.
          A model that uses them is partly reading the answer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(
        f"""
        <div class="card">
          <div class="card-label">Disclaimer</div>
          <h3>Please read before using</h3>
          <p>This app is a research and teaching prototype. It is not a medical device, it has not been
          reviewed by a regulator, and it has not been tested in any clinic. It must not be used to make
          decisions about real patients.</p>
          <p>The guideline rules are my own reading of the WHO 2021 recommendations and may contain errors.
          National programmes, including Ghana's, may use different ages, tests or intervals. Always follow
          local protocols and clinical judgement.</p>
          <p>The risk model was trained on {meta['n_after_dedup']} women from one hospital in Caracas,
          Venezuela. Most were younger than 30 and very few were over 50, so estimates for older women are
          especially uncertain. It has not been validated on any Ghanaian or African population.</p>
          <p>Do not enter names or other identifying details. Nothing you enter is stored by the app.</p>
        </div>
        <div class="card">
          <div class="card-label">Method in brief</div>
          <p><b>Data.</b> UCI Cervical Cancer (Risk Factors) dataset, CC BY 4.0. {meta['n_raw'] - meta['n_after_dedup']}
          exact duplicates removed before splitting. Outcome: biopsy result.</p>
          <p><b>Leakage control.</b> Colposcopy, Schiller, cytology and prior diagnosis columns excluded.
          The test set was split off first and used once.</p>
          <p><b>Model selection.</b> Five candidates compared with nested cross-validation (5 folds × 5
          repeats outside, 3 folds inside) on the development set, ranked by PR-AUC because only about
          6% of cases are positive. Ties within 0.01 go to the model that uses less energy.</p>
          <p><b>Threshold.</b> Chosen from out-of-fold predictions to reach at least
          {meta['target_sensitivity']:.0%} sensitivity, because missing a lesion costs more than an
          extra follow-up conversation.</p>
          <p><b>Guideline.</b> WHO guideline for screening and treatment of cervical pre-cancer lesions
          for cervical cancer prevention, second edition, 2021.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
