import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanIQ · Approval Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load models (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = Path(__file__).parent
    clf = joblib.load(base / "models" / "rf_classifier_pipeline.pkl")
    reg = joblib.load(base / "models" / "rf_regressor_pipeline.pkl")
    return clf, reg

clf, reg = load_models()

CLF_COLS = ["no_of_dependents","income_annum","loan_amount","loan_term",
            "cibil_score","residential_assets_value","commercial_assets_value",
            "luxury_assets_value","bank_asset_value","education","self_employed"]
REG_COLS = ["no_of_dependents","income_annum","loan_term","cibil_score",
            "residential_assets_value","commercial_assets_value",
            "luxury_assets_value","bank_asset_value","education","self_employed","loan_status"]

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg:       #080B10;
  --surface:  #0D1117;
  --card:     #111720;
  --border:   #1E2A38;
  --gold:     #C9A84C;
  --gold-d:   #8A6D2F;
  --gold-l:   #F0C96A;
  --cyan:     #00D4FF;
  --green:    #00E676;
  --red:      #FF3B5C;
  --text:     #E8EDF5;
  --muted:    #5A6A80;
  --accent:   #1A2535;
}

html, body, [class*="css"], .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hero Header ── */
.hero {
  background: linear-gradient(135deg, #080B10 0%, #0D1520 50%, #080B10 100%);
  border-bottom: 1px solid var(--border);
  padding: 2.5rem 3rem 2rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 300px; height: 300px;
  border: 1px solid rgba(201,168,76,.08);
  border-radius: 50%;
}
.hero::after {
  content: '';
  position: absolute;
  top: -30px; right: -30px;
  width: 180px; height: 180px;
  border: 1px solid rgba(201,168,76,.12);
  border-radius: 50%;
}
.hero-badge {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--gold);
  background: rgba(201,168,76,.08);
  border: 1px solid rgba(201,168,76,.2);
  padding: .3rem .8rem;
  border-radius: 2px;
  margin-bottom: 1rem;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 2.8rem;
  font-weight: 800;
  letter-spacing: -.03em;
  color: var(--text);
  line-height: 1.1;
  margin: 0 0 .5rem;
}
.hero-title span { color: var(--gold); }
.hero-sub {
  font-family: 'DM Sans', sans-serif;
  font-size: .95rem;
  color: var(--muted);
  font-weight: 300;
}
.hero-stats {
  display: flex;
  gap: 2.5rem;
  margin-top: 1.8rem;
}
.stat-item { text-align: left; }
.stat-val {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--gold-l);
}
.stat-lbl {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .65rem;
  color: var(--muted);
  letter-spacing: .1em;
  text-transform: uppercase;
}

/* ── Main layout ── */
.main-grid {
  display: grid;
  grid-template-columns: 1fr 380px;
  gap: 0;
  min-height: calc(100vh - 180px);
}
.form-panel {
  padding: 2.5rem 3rem;
  border-right: 1px solid var(--border);
}
.result-panel {
  padding: 2.5rem 2rem;
  background: var(--surface);
  position: sticky;
  top: 0;
  height: fit-content;
}

/* ── Section labels ── */
.section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .65rem;
  font-weight: 500;
  letter-spacing: .15em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 1.2rem;
  padding-bottom: .5rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: .5rem;
}
.section-label::before {
  content: '';
  display: inline-block;
  width: 6px; height: 6px;
  background: var(--gold);
  border-radius: 50%;
}

/* ── Input styling ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div > div,
.stSlider > div {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Mono', monospace !important;
}
.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > div:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 2px rgba(201,168,76,.15) !important;
}
label {
  font-family: 'DM Sans', sans-serif !important;
  font-size: .8rem !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: .06em !important;
}

/* ── CIBIL gauge ── */
.gauge-wrap {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
  margin-bottom: 1.2rem;
}
.gauge-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .65rem;
  color: var(--muted);
  letter-spacing: .12em;
  text-transform: uppercase;
  margin-bottom: .5rem;
}
.gauge-track {
  height: 6px;
  background: var(--border);
  border-radius: 3px;
  overflow: hidden;
}
.gauge-fill {
  height: 100%;
  border-radius: 3px;
  transition: width .5s ease;
}
.gauge-markers {
  display: flex;
  justify-content: space-between;
  margin-top: .3rem;
}
.gauge-marker {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .6rem;
  color: var(--muted);
}

/* ── Result card ── */
.result-title {
  font-family: 'Syne', sans-serif;
  font-size: 1rem;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .12em;
  margin-bottom: 1.5rem;
}
.verdict-box {
  border-radius: 6px;
  padding: 1.5rem;
  margin-bottom: 1.2rem;
  border: 1px solid;
  text-align: center;
}
.verdict-box.approved {
  background: rgba(0,230,118,.05);
  border-color: rgba(0,230,118,.25);
}
.verdict-box.rejected {
  background: rgba(255,59,92,.05);
  border-color: rgba(255,59,92,.25);
}
.verdict-icon {
  font-size: 2rem;
  margin-bottom: .5rem;
}
.verdict-text {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  font-weight: 800;
  letter-spacing: -.02em;
}
.verdict-box.approved .verdict-text { color: var(--green); }
.verdict-box.rejected .verdict-text { color: var(--red); }

.metric-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: .75rem 0;
  border-bottom: 1px solid var(--border);
}
.metric-row:last-child { border-bottom: none; }
.metric-key {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .7rem;
  color: var(--muted);
  letter-spacing: .08em;
  text-transform: uppercase;
}
.metric-val {
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem;
  font-weight: 700;
}
.metric-val.gold { color: var(--gold-l); }
.metric-val.green { color: var(--green); }
.metric-val.red { color: var(--red); }
.metric-val.cyan { color: var(--cyan); }

/* Confidence bar */
.conf-bar-wrap { margin-top: 1.2rem; }
.conf-bar-label {
  display: flex;
  justify-content: space-between;
  font-family: 'IBM Plex Mono', monospace;
  font-size: .65rem;
  color: var(--muted);
  margin-bottom: .4rem;
}
.conf-bar-track {
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
}
.conf-bar-fill {
  height: 100%;
  border-radius: 2px;
}

/* ── Submit button ── */
.stButton > button {
  background: linear-gradient(135deg, var(--gold-d), var(--gold)) !important;
  color: #080B10 !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  letter-spacing: .04em !important;
  border: none !important;
  border-radius: 4px !important;
  padding: .85rem 2rem !important;
  width: 100% !important;
  cursor: pointer !important;
  transition: all .2s !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 24px rgba(201,168,76,.3) !important;
}

/* ── Idle state ── */
.idle-state {
  text-align: center;
  padding: 3rem 1rem;
  color: var(--muted);
}
.idle-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  opacity: .3;
}
.idle-text {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .75rem;
  letter-spacing: .1em;
  text-transform: uppercase;
}

/* Divider */
.hline {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.5rem 0;
}

/* Tag pill */
.tag {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: .6rem;
  padding: .2rem .5rem;
  border-radius: 2px;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.tag.good { background: rgba(0,230,118,.1); color: var(--green); border: 1px solid rgba(0,230,118,.2); }
.tag.warn { background: rgba(201,168,76,.1); color: var(--gold); border: 1px solid rgba(201,168,76,.2); }
.tag.bad  { background: rgba(255,59,92,.1);  color: var(--red);  border: 1px solid rgba(255,59,92,.2); }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">◈ AI-Powered · Two-Stage Pipeline</div>
  <div class="hero-title">Loan<span>IQ</span> Approval Engine</div>
  <div class="hero-sub">Random Forest · Stage I: Classification · Stage II: Amount Prediction</div>
  <div class="hero-stats">
    <div class="stat-item">
      <div class="stat-val">98.2%</div>
      <div class="stat-lbl">Model Accuracy</div>
    </div>
    <div class="stat-item">
      <div class="stat-val">4,269</div>
      <div class="stat-lbl">Training Records</div>
    </div>
    <div class="stat-item">
      <div class="stat-val">2-Stage</div>
      <div class="stat-lbl">Pipeline</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout: two columns ────────────────────────────────────────────────────
left, right = st.columns([3, 1.5], gap="large")

with left:
    st.markdown('<div style="padding: 2rem 1rem 0;">', unsafe_allow_html=True)

    # ── Personal Info ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Personal Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        dependents = st.number_input("Dependents", min_value=0, max_value=20, value=2)
    with c2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with c3:
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    income = st.number_input("Annual Income (₹)", min_value=100000, max_value=100000000,
                              value=9600000, step=100000,
                              help="Your gross annual income")

    st.markdown('<hr class="hline"/>', unsafe_allow_html=True)

    # ── Loan Details ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Loan Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        loan_amount = st.number_input("Requested Amount (₹)", min_value=10000,
                                      max_value=500000000, value=29900000, step=100000)
    with c2:
        loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=30, value=12)

    cibil = st.slider("CIBIL Score", min_value=300, max_value=950, value=750, step=5)

    # CIBIL gauge
    pct = (cibil - 300) / 650 * 100
    if cibil >= 750:
        color, tag_cls, tag_lbl = "#00E676", "good", "Excellent"
    elif cibil >= 650:
        color, tag_cls, tag_lbl = "#C9A84C", "warn", "Fair"
    else:
        color, tag_cls, tag_lbl = "#FF3B5C", "bad", "Poor"

    st.markdown(f"""
    <div class="gauge-wrap">
      <div class="gauge-label">Credit Score Indicator &nbsp;
        <span class="tag {tag_cls}">{tag_lbl}</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{pct}%; background:{color};"></div>
      </div>
      <div class="gauge-markers">
        <span class="gauge-marker">300 · Poor</span>
        <span class="gauge-marker">650 · Fair</span>
        <span class="gauge-marker">750 · Good</span>
        <span class="gauge-marker">950 · Excellent</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="hline"/>', unsafe_allow_html=True)

    # ── Assets ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Asset Portfolio</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        res_assets = st.number_input("Residential Assets (₹)", value=2400000, step=100000)
        lux_assets = st.number_input("Luxury Assets (₹)", value=22700000, step=100000)
    with c2:
        com_assets = st.number_input("Commercial Assets (₹)", value=17600000, step=100000)
        bank_assets = st.number_input("Bank Assets (₹)", value=8000000, step=100000)

    total_assets = res_assets + com_assets + lux_assets + bank_assets
    ltv = (loan_amount / total_assets * 100) if total_assets > 0 else 0

    st.markdown(f"""
    <div style="display:flex;gap:1.5rem;margin:.5rem 0 1.5rem;padding:1rem;
                background:var(--card);border:1px solid var(--border);border-radius:6px;">
      <div>
        <div class="stat-lbl">Total Assets</div>
        <div class="stat-val" style="font-size:1.1rem;">₹{total_assets:,.0f}</div>
      </div>
      <div>
        <div class="stat-lbl">Loan-to-Asset Ratio</div>
        <div class="stat-val" style="font-size:1.1rem;
          color:{'var(--green)' if ltv < 50 else 'var(--gold)' if ltv < 80 else 'var(--red)'}">
          {ltv:.1f}%
        </div>
      </div>
      <div>
        <div class="stat-lbl">Income Multiple</div>
        <div class="stat-val" style="font-size:1.1rem;">
          {loan_amount/income:.1f}x
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("◈  Run Prediction Engine", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Right panel: Result ────────────────────────────────────────────────────
with right:
    st.markdown('<div style="padding: 2rem 0 0;">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">◈ Prediction Output</div>', unsafe_allow_html=True)

    if predict_btn:
        # Build DataFrames
        clf_row = {
            "no_of_dependents": dependents, "income_annum": income,
            "loan_amount": loan_amount, "loan_term": loan_term,
            "cibil_score": cibil, "residential_assets_value": res_assets,
            "commercial_assets_value": com_assets, "luxury_assets_value": lux_assets,
            "bank_asset_value": bank_assets, "education": education,
            "self_employed": self_employed,
        }
        clf_df = pd.DataFrame([clf_row], columns=CLF_COLS)

        with st.spinner(""):
            proba    = clf.predict_proba(clf_df)[0]
            classes  = list(clf.classes_)
            appr_prob = float(proba[classes.index(1)])
            approved  = appr_prob >= 0.5

        if approved:
            reg_row = {**{k: v for k, v in clf_row.items() if k != "loan_amount"},
                       "loan_status": "approved"}
            reg_df = pd.DataFrame([reg_row], columns=REG_COLS)
            predicted_amount = float(reg.predict(reg_df)[0])
        else:
            predicted_amount = None

        # Verdict
        if approved:
            st.markdown(f"""
            <div class="verdict-box approved">
              <div class="verdict-icon">✦</div>
              <div class="verdict-text">APPROVED</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-box rejected">
              <div class="verdict-icon">✕</div>
              <div class="verdict-text">REJECTED</div>
            </div>""", unsafe_allow_html=True)

        # Metrics
        amt_html = f'<span class="metric-val gold">₹{predicted_amount:,.0f}</span>' \
                   if predicted_amount else '<span class="metric-val" style="color:var(--muted)">N/A</span>'

        st.markdown(f"""
        <div style="margin-top:.5rem;">
          <div class="metric-row">
            <span class="metric-key">Confidence</span>
            <span class="metric-val {'green' if approved else 'red'}">{appr_prob*100:.1f}%</span>
          </div>
          <div class="metric-row">
            <span class="metric-key">Requested</span>
            <span class="metric-val cyan">₹{loan_amount:,.0f}</span>
          </div>
          <div class="metric-row">
            <span class="metric-key">Sanctioned Est.</span>
            {amt_html}
          </div>
          <div class="metric-row">
            <span class="metric-key">CIBIL</span>
            <span class="metric-val {'green' if cibil>=750 else 'gold' if cibil>=650 else 'red'}">{cibil}</span>
          </div>
          <div class="metric-row">
            <span class="metric-key">LTV Ratio</span>
            <span class="metric-val {'green' if ltv<50 else 'gold' if ltv<80 else 'red'}">{ltv:.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        bar_color = "#00E676" if approved else "#FF3B5C"
        st.markdown(f"""
        <div class="conf-bar-wrap">
          <div class="conf-bar-label">
            <span>Approval Confidence</span>
            <span>{appr_prob*100:.1f}%</span>
          </div>
          <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:{appr_prob*100:.1f}%;background:{bar_color};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Tip
        if not approved:
            tips = []
            if cibil < 700: tips.append("· Improve CIBIL score above 700")
            if ltv > 80:    tips.append("· Reduce loan amount or add assets")
            if loan_amount > income * 5: tips.append("· Requested amount is high vs income")
            if tips:
                st.markdown(f"""
                <div style="margin-top:1.2rem;padding:1rem;background:rgba(255,59,92,.05);
                            border:1px solid rgba(255,59,92,.15);border-radius:4px;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
                              color:var(--red);letter-spacing:.1em;
                              text-transform:uppercase;margin-bottom:.5rem;">
                    Improvement Areas
                  </div>
                  <div style="font-size:.8rem;color:#FF8FA3;line-height:1.8;">
                    {'<br>'.join(tips)}
                  </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="idle-state">
          <div class="idle-icon">◈</div>
          <div class="idle-text">Awaiting input<br><br>Fill the form and<br>run the engine</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 3rem;border-top:1px solid #1E2A38;
            display:flex;justify-content:space-between;align-items:center;">
  <span style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
               color:#5A6A80;letter-spacing:.1em;">
    LOANIQ · RANDOM FOREST ENGINE · TWO-STAGE ML PIPELINE
  </span>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#5A6A80;">
    v1.0.0
  </span>
</div>
""", unsafe_allow_html=True)