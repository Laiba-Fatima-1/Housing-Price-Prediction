import streamlit as st
import textwrap
import joblib
import json
import pandas as pd
import numpy as np

def html_md(raw: str):
    """
    Render an HTML string safely, bypassing st.markdown entirely.
    st.markdown(..., unsafe_allow_html=True) still runs its markdown
    parser first -- stray characters like the '*' in CSS attribute
    selectors (e.g. [class*="css"]) get misread as markdown emphasis
    and can corrupt/leak raw HTML or CSS as visible text. st.html()
    renders the string as literal HTML with no markdown parsing at
    all, so this can't happen regardless of indentation or content.
    """
    lines = [line.strip() for line in textwrap.dedent(raw).split("\n")]
    st.html("\n".join(lines))


st.set_page_config(page_title="The Appraisal Office — Housing Price Estimator", page_icon="⌂", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_assets():
    model = joblib.load("housing_model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return model, feature_cols, metrics

model, feature_cols, metrics = load_assets()
OCEAN_OPTIONS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

html_md("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,500;0,600;0,700;1,500&family=Work+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root {
    --blue: #1E2A38;
    --blue-soft: #52667A;
    --paper: #F1F5F6;
    --paper2: #E7EEF0;
    --grid: #AFC9D3;
    --copper: #A15C33;
    --copper-dark: #7A4526;
    --sage: #2F6B63;
    --line: #B9CBD1;
}
html, body, [class*="css"] { font-family: 'Work Sans', sans-serif; }
.stApp {
    background:
      repeating-linear-gradient(0deg, transparent, transparent 39px, var(--grid) 40px),
      repeating-linear-gradient(90deg, transparent, transparent 39px, var(--grid) 40px),
      var(--paper);
    background-size: 40px 40px, 40px 40px, cover;
    background-blend-mode: multiply, multiply, normal;
    opacity: 1;
}
.stApp::before { content: ""; }
#MainMenu, footer, header {visibility: hidden;}

.eyebrow { font-family:'Space Mono', monospace; font-size:0.72rem; letter-spacing:0.22em; text-transform:uppercase; color: var(--copper); font-weight:700; margin-bottom:0.4rem; }
.masthead { font-family:'Spectral', serif; font-weight:700; font-size:2.9rem; line-height:1.05; color: var(--blue); margin:0 0 0.3rem 0; }
.masthead em { color: var(--copper); font-style: italic; font-weight: 500; }
.subhead { font-size:1.0rem; color: var(--blue-soft); max-width:680px; line-height:1.55; }
.divider { border:none; border-top:1.5px dashed var(--line); margin:1.5rem 0 1.7rem 0; }

.panel { background: rgba(255,255,255,0.72); border:1px solid var(--line); border-radius:6px; padding:1.3rem 1.4rem; }
.panel-label { font-family:'Space Mono', monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.12em; color: var(--blue-soft); margin-bottom:0.7rem; display:block; }

div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: white !important; border:1.3px solid var(--line) !important; border-radius:4px !important;
    font-family:'Space Mono', monospace !important; color: var(--blue) !important;
}
div.stButton > button {
    background: var(--blue); color: var(--paper); border:none; border-radius:3px;
    font-weight:600; font-size:0.9rem; letter-spacing:0.03em; padding:0.65rem 1.6rem; width:100%;
}
div.stButton > button:hover { background: var(--copper-dark); color: white; }

.seal-wrap { display:flex; justify-content:center; align-items:center; padding:1rem 0 0.6rem 0; }
.seal {
    width:190px; height:190px; border-radius:50%;
    border: 3px solid var(--copper);
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    position:relative; font-family:'Spectral', serif; color: var(--copper-dark);
    box-shadow: inset 0 0 0 4px var(--paper), inset 0 0 0 5.5px var(--copper);
    opacity:0; animation: sealIn 0.4s ease-out 0.05s forwards;
}
@keyframes sealIn { 0%{opacity:0; transform:scale(1.5) rotate(4deg);} 70%{opacity:1;} 100%{opacity:1; transform:scale(1) rotate(0deg);} }
.seal .tag { font-family:'Space Mono', monospace; font-size:0.6rem; letter-spacing:0.15em; text-transform:uppercase; color: var(--blue-soft); }
.seal .price { font-size:1.55rem; font-weight:700; margin-top:4px; }
.seal .range { font-family:'Space Mono', monospace; font-size:0.66rem; margin-top:5px; color: var(--blue-soft); }

.ledger-bar-row { display:flex; align-items:center; margin-bottom:0.55rem; }
.ledger-label { width:150px; font-family:'Space Mono', monospace; font-size:0.72rem; color: var(--blue); flex-shrink:0; }
.ledger-track { flex:1; background: var(--paper2); border:1px solid var(--line); border-radius:3px; height:14px; overflow:hidden; }
.ledger-fill { height:100%; background: var(--copper); }
.ledger-val { width:52px; text-align:right; font-family:'Space Mono', monospace; font-size:0.7rem; color: var(--blue-soft); }

.metric-card { background: rgba(255,255,255,0.72); border:1px solid var(--line); border-radius:6px; padding:1rem 1.2rem; }
.metric-label { font-family:'Space Mono', monospace; font-size:0.66rem; text-transform:uppercase; letter-spacing:0.1em; color: var(--blue-soft); }
.metric-value { font-family:'Spectral', serif; font-size:1.7rem; font-weight:700; color: var(--blue); margin-top:0.1rem; }

section[data-testid="stSidebar"] { background: var(--blue); }
section[data-testid="stSidebar"] * { color: var(--paper) !important; }
</style>
""")

with st.sidebar:
    st.html("<div class='eyebrow' style='color:#D89A6A;'>ARCH TECHNOLOGIES</div>")
    st.html("<div style='font-family:Spectral,serif; font-size:1.35rem; font-weight:700; margin-bottom:1rem;'>ML Internship<br>Month 2 · Task 3</div>")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("Random Forest Regressor (200 trees)")
    st.markdown("**Training data**")
    st.markdown(f"{metrics['train_size']:,} properties (California Housing dataset)")
    st.markdown("---")
    st.markdown("**Performance**")
    st.markdown(f"R\u00b2 — {metrics['r2']:.3f}")
    st.markdown(f"MAE — ${metrics['mae']:,.0f}")
    st.markdown(f"RMSE — ${metrics['rmse']:,.0f}")
    st.markdown("---")
    st.html("<span style='font-size:0.78rem; opacity:0.8;'>Dataset: California Housing Prices (public, GitHub-hosted mirror of the same data distributed on Kaggle).</span>")

st.html("<div class='eyebrow'>TASK 03 — MACHINE LEARNING</div>")
st.html("<div class='masthead'>The Appraisal Office<br><em>an estimate, drafted from the numbers</em></div>")
st.html("<div class='subhead'>Enter a property's statistics below. The office reads them against thousands of prior California sales and drafts an estimated market value, sealed with a confidence range.</div>")
st.html("<hr class='divider'>")

col_form, col_result = st.columns([1.35, 1], gap="large")

with col_form:
    st.html("<span class='panel-label'>\u25a2 Location</span>")
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        longitude = st.number_input("Longitude", value=-119.5, min_value=-125.0, max_value=-113.0, step=0.1, format="%.2f")
    with lc2:
        latitude = st.number_input("Latitude", value=36.5, min_value=32.0, max_value=42.5, step=0.1, format="%.2f")
    with lc3:
        ocean_proximity = st.selectbox("Ocean proximity", OCEAN_OPTIONS, index=0)

    st.html("<span class='panel-label' style='margin-top:1rem;'>\u25a2 Structure</span>")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        housing_median_age = st.number_input("Median age (yrs)", value=28, min_value=1, max_value=52, step=1)
    with sc2:
        total_rooms = st.number_input("Total rooms (block)", value=2600, min_value=2, max_value=40000, step=50)
    with sc3:
        total_bedrooms = st.number_input("Total bedrooms (block)", value=540, min_value=1, max_value=7000, step=10)

    st.html("<span class='panel-label' style='margin-top:1rem;'>\u25a2 Community</span>")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        population = st.number_input("Population (block)", value=1425, min_value=1, max_value=36000, step=50)
    with cc2:
        households = st.number_input("Households (block)", value=500, min_value=1, max_value=6100, step=10)
    with cc3:
        median_income = st.number_input("Median income ($10k)", value=3.9, min_value=0.4, max_value=15.0, step=0.1, format="%.1f")

    appraise = st.button("Appraise Property", use_container_width=True)

with col_result:
    st.html("<span class='panel-label'>\u25c8 Appraisal</span>")
    if appraise:
        row = {c: 0 for c in feature_cols}
        row["longitude"] = longitude
        row["latitude"] = latitude
        row["housing_median_age"] = housing_median_age
        row["total_rooms"] = total_rooms
        row["total_bedrooms"] = total_bedrooms
        row["population"] = population
        row["households"] = households
        row["median_income"] = median_income
        row["rooms_per_household"] = total_rooms / households
        row["bedrooms_per_room"] = total_bedrooms / total_rooms
        row["population_per_household"] = population / households
        ocean_col = f"ocean_{ocean_proximity}"
        if ocean_col in row:
            row[ocean_col] = 1

        X = pd.DataFrame([row])[feature_cols]
        pred = model.predict(X)[0]
        low, high = pred - metrics["mae"], pred + metrics["mae"]

        html_md(f"""
        <div class='seal-wrap'>
            <div class='seal'>
                <div class='tag'>appraised value</div>
                <div class='price'>${pred:,.0f}</div>
                <div class='range'>\u00b1 ${metrics['mae']:,.0f} range</div>
            </div>
        </div>
        """)
        st.html(f"<div style='text-align:center; font-family:Space Mono, monospace; font-size:0.78rem; color:var(--blue-soft);'>likely between ${low:,.0f} and ${high:,.0f}</div>")

        st.html("<div style='margin-top:1.3rem;'>")
        st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=5, size=200)
        st.html("</div>")
    else:
        html_md("""
        <div style='background:rgba(255,255,255,0.6); border:1.5px dashed var(--line); border-radius:6px;
                    padding:2.4rem 1.5rem; text-align:center; color:var(--blue-soft); font-size:0.9rem;'>
            Fill in the property details and press <strong>Appraise Property</strong><br>to draft an estimate.
        </div>
        """)

st.html("<hr class='divider'>")
st.html("<div class='eyebrow'>MODEL PERFORMANCE</div>")
st.html("<div style='font-family:Spectral,serif; font-size:1.5rem; font-weight:700; margin-bottom:1rem; color:var(--blue);'>Evaluated on a held-out test set</div>")

m1, m2, m3 = st.columns(3)
for col, label, value in [(m1, "R\u00b2 Score", f"{metrics['r2']:.3f}"), (m2, "MAE", f"${metrics['mae']:,.0f}"), (m3, "RMSE", f"${metrics['rmse']:,.0f}")]:
    with col:
        st.html(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>")

st.html("<div style='height:1.4rem;'></div>")
col_imp, col_notes = st.columns([1.1, 1], gap="large")
with col_imp:
    st.html("<div class='panel-label'>FEATURE IMPORTANCE</div>")
    top_features = metrics["feature_importances"][:7]
    max_imp = top_features[0][1]
    bars = ""
    for name, imp in top_features:
        pct = (imp / max_imp) * 100
        bars += f"""<div class='ledger-bar-row'>
            <div class='ledger-label'>{name}</div>
            <div class='ledger-track'><div class='ledger-fill' style='width:{pct:.0f}%;'></div></div>
            <div class='ledger-val'>{imp:.3f}</div>
        </div>"""
    st.html(bars)

with col_notes:
    st.html("<div class='panel-label'>READING THESE NUMBERS</div>")
    html_md(f"""
    <div style='font-size:0.87rem; line-height:1.7; color:var(--blue-soft);'>
    <strong>R\u00b2</strong> — the fraction of price variation the model explains. {metrics['r2']:.3f} means it accounts for about {metrics['r2']*100:.0f}% of what drives price differences between properties.<br>
    <strong>MAE</strong> — on average, predictions are off by about ${metrics['mae']:,.0f}.<br>
    <strong>RMSE</strong> — similar to MAE but penalizes large misses more heavily.<br>
    <strong>Feature importance</strong> — median income dominates: unsurprising, since it's the single strongest proxy for what a neighborhood's homes can sell for.
    </div>
    """)

st.html("<div style='height:2rem;'></div>")
st.html("<div style='text-align:center; font-family:Space Mono, monospace; font-size:0.7rem; letter-spacing:0.1em; color:var(--blue-soft); opacity:0.75;'>THE APPRAISAL OFFICE — BUILT FOR ARCH TECHNOLOGIES ML INTERNSHIP, MONTH 2</div>")