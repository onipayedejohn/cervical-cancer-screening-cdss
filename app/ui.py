"""Styling and small HTML building blocks for the Streamlit app.

System fonts, one accent colour, and status colours that always come with a
text label, so nothing depends on colour alone.
"""

from html import escape

import streamlit as st

STATUS_STYLE = {
    # status: (label, text colour, background)
    "refer": ("Refer", "#9f1c1c", "#fdecec"),
    "follow_up": ("Follow-up", "#8a4a00", "#fff1dc"),
    "due": ("Screen now", "#7a5b00", "#fff7d1"),
    "not_due": ("Up to date", "#0f6b3a", "#e3f5ea"),
    "can_stop": ("Can stop", "#0f6b3a", "#e3f5ea"),
    "not_yet_eligible": ("Not yet eligible", "#3d4451", "#eef0f3"),
}

BAND_STYLE = {
    "Lower": ("#0f6b3a", "#e3f5ea"),
    "Average": ("#3d4451", "#eef0f3"),
    "Higher": ("#8a4a00", "#fff1dc"),
}

CSS = """
<style>
:root {
  --ink: #111827;
  --ink-2: #4b5563;
  --ink-3: #6b7280;
  --line: #e5e7eb;
  --card: #ffffff;
  --page: #f5f6f8;
  --accent: #0f766e;
  --accent-soft: #e6f4f2;
  --radius: 14px;
}
html, body, [class*="css"], .stApp, button, input, textarea, select {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
               Arial, "Noto Sans", sans-serif !important;
}
.stApp { background: var(--page); }
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
header[data-testid="stHeader"] { background: transparent; }

h1, h2, h3 { color: var(--ink); letter-spacing: -0.01em; }
.eyebrow { font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
           color: var(--accent); margin-bottom: .2rem; }
.app-title { font-size: 2rem; font-weight: 700; color: var(--ink); line-height: 1.15; margin: 0; }
.app-sub { color: var(--ink-2); font-size: 1rem; margin: .35rem 0 1rem; max-width: 60ch; }

.notice { border: 1px solid #f3d9a4; background: #fffaf0; color: #6b4a00; border-radius: 10px;
          padding: .65rem .9rem; font-size: .88rem; margin-bottom: 1.2rem; }

.card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
        padding: 1.15rem 1.25rem; margin-bottom: 1rem; }
.card-label { font-size: .75rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase;
              color: var(--ink-3); margin-bottom: .55rem; }
.card h3 { font-size: 1.3rem; margin: .5rem 0 .35rem; }
.card p { color: var(--ink-2); margin: 0 0 .6rem; line-height: 1.5; }
.next { border-top: 1px solid var(--line); padding-top: .7rem; margin-top: .7rem; color: var(--ink); }
.next b { color: var(--ink); }

.pill { display: inline-block; font-size: .8rem; font-weight: 600; padding: .22rem .65rem;
        border-radius: 999px; }
.chip { display: inline-block; font-size: .75rem; color: var(--ink-2); border: 1px solid var(--line);
        border-radius: 6px; padding: .1rem .45rem; margin: .2rem .3rem 0 0; background: #fafafa; }
.note { font-size: .88rem; color: var(--ink-2); background: #f8fafc; border-radius: 8px;
        padding: .5rem .7rem; margin-top: .5rem; }

.risk-row { display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap; margin: .4rem 0 .2rem; }
.risk-big { font-size: 2.1rem; font-weight: 700; color: var(--ink); line-height: 1; }
.risk-rel { color: var(--ink-2); font-size: .95rem; }
.meter { position: relative; height: 8px; border-radius: 99px; margin: .8rem 0 .3rem;
         background: linear-gradient(90deg, #cfe9da 0 40%, #e5e7eb 40% 80%, #f6d9b0 80% 100%); }
.meter span { position: absolute; top: -4px; width: 4px; height: 16px; border-radius: 2px;
              background: var(--ink); transform: translateX(-2px); }
.meter-labels { display: flex; justify-content: space-between; font-size: .72rem; color: var(--ink-3); }
.driver { display: flex; justify-content: space-between; gap: 1rem; padding: .42rem 0;
          border-bottom: 1px dashed var(--line); font-size: .92rem; color: var(--ink); }
.driver:last-child { border-bottom: 0; }
.up { color: #9a3412; font-weight: 600; white-space: nowrap; }
.down { color: #0f6b3a; font-weight: 600; white-space: nowrap; }

.tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem; margin-bottom: 1rem; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: .9rem 1rem; }
.tile .v { font-size: 1.6rem; font-weight: 700; color: var(--ink); }
.tile .k { font-size: .8rem; color: var(--ink-3); margin-top: .1rem; }
.tile .s { font-size: .78rem; color: var(--ink-2); margin-top: .35rem; }

div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--card); border-color: var(--line) !important;
                            border-radius: var(--radius) !important; }
.section { font-weight: 650; color: var(--ink); font-size: .95rem; margin: .4rem 0 .1rem; }
.hint { color: var(--ink-3); font-size: .82rem; margin-bottom: .4rem; }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; overflow-x: auto; }
.stTabs [data-baseweb="tab"] { padding: .5rem .9rem; }
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
.stFormSubmitButton > button { background: var(--accent); color: #fff; border: 0; width: 100%; padding: .65rem; }
.stFormSubmitButton > button:hover { background: #0b5f59; color: #fff; }

footer { visibility: hidden; }

@media (max-width: 900px) {
  .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  .block-container { padding: 1.2rem .9rem 3rem; }
  .app-title { font-size: 1.55rem; }
  .card { padding: 1rem; }
  .risk-big { font-size: 1.8rem; }
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def header():
    st.markdown(
        """
        <div class="eyebrow">Clinical decision support prototype</div>
        <div class="app-title">Cervical Screening Check</div>
        <p class="app-sub">WHO 2021 screening rules decide what to do today. A small risk
        model adds context on who may need closer follow-up.</p>
        <div class="notice"><b>Research and teaching prototype.</b> Not a medical device and not
        validated for clinical use. See the Disclaimer tab.</div>
        """,
        unsafe_allow_html=True,
    )


def rule_card(rec) -> str:
    label, fg, bg = STATUS_STYLE[rec.status.value]
    refs = "".join(f'<span class="chip">WHO {escape(r)}</span>' for r in rec.guideline_refs)
    notes = "".join(f'<div class="note">{escape(n)}</div>' for n in rec.notes)
    due = ""
    if rec.next_due_in_years:
        due = f'<p><b>Next due in:</b> about {rec.next_due_in_years:g} year(s)</p>'
    return f"""
    <div class="card">
      <div class="card-label">Guideline decision</div>
      <span class="pill" style="color:{fg};background:{bg}">{label}</span>
      <h3>{escape(rec.headline)}</h3>
      <p>{escape(rec.reason)}</p>
      {due}
      <div class="next"><b>Next step:</b> {escape(rec.next_step)}</div>
      <div style="margin-top:.5rem">{refs}</div>
      {notes}
    </div>
    """


def risk_card(risk, cuts, applies: bool) -> str:
    fg, bg = BAND_STYLE[risk.band]
    # Position the marker on a 0-40-80-100 scale matching the band widths
    lo, hi = cuts
    p = risk.probability
    if p < lo:
        pos = 40 * p / lo
    elif p < hi:
        pos = 40 + 40 * (p - lo) / (hi - lo)
    else:
        pos = min(99, 80 + 20 * (p - hi) / max(hi, 1e-6))
    drivers = "".join(
        f'<div class="driver"><span>{escape(lbl)}</span>'
        f'<span class="{"up" if d > 0 else "down"}">{"raises" if d > 0 else "lowers"} '
        f'{abs(d) * 100:.1f} pts</span></div>'
        for lbl, d in risk.drivers
    ) or '<p>Her answers sit close to typical values, so no single factor stands out.</p>'
    band_rate = (
        f"In the held-out test set, {risk.observed_rate_in_band:.0%} of women in the "
        f"{risk.band.lower()} band had a positive biopsy."
        if risk.observed_rate_in_band is not None else ""
    )
    caveat = (
        "This estimate does not change the guideline decision. Use it to prioritise counselling "
        "and to make sure she comes back on schedule."
        if applies else
        "The guideline decision above takes priority, so this estimate is shown for context only."
    )
    return f"""
    <div class="card">
      <div class="card-label">Model risk estimate</div>
      <span class="pill" style="color:{fg};background:{bg}">{risk.band} risk band</span>
      <div class="risk-row">
        <span class="risk-big">{p:.1%}</span>
        <span class="risk-rel">estimated chance of a positive biopsy,
        {risk.relative_to_average:.1f}× the average in the training data</span>
      </div>
      <div class="meter"><span style="left:{pos:.1f}%"></span></div>
      <div class="meter-labels"><span>Lower</span><span>Average</span><span>Higher</span></div>
      <p style="margin-top:.7rem">{band_rate}</p>
      <div class="card-label" style="margin-top:.9rem">What moved the estimate</div>
      {drivers}
      <div class="note">{caveat}</div>
    </div>
    """


def tile(value, key, sub="") -> str:
    return f'<div class="tile"><div class="v">{value}</div><div class="k">{key}</div><div class="s">{sub}</div></div>'
