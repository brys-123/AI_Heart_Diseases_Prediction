import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="AI CardioScan — Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "show_patient_form" not in st.session_state:
    st.session_state["show_patient_form"] = True

default_inputs = {
    "age": 52,
    "sex": 1,
    "cp": 0,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 0,
    "ca": 0,
    "thal": 1,
}
for key, value in default_inputs.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap');

:root {
  --grad1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  --grad2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  --grad3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  --grad4: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
  --grad5: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
  --white: #ffffff;
  --soft:  #f8faff;
  --text:  #1a1f3a;
  --muted: #6b7280;
  --card:  rgba(255,255,255,0.85);
}

html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif;
  color: var(--text);
  box-sizing: border-box;
}
*, *::before, *::after {
  box-sizing: inherit;
}

.stApp {
  background: linear-gradient(160deg,
    #e0e7ff 0%,
    #f0f4ff 20%,
    #fdf4ff 45%,
    #fff0f6 65%,
    #f0fffe 85%,
    #f5f0ff 100%);
  min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── HERO ── */
.hero {
  text-align: center;
  padding: 2.5rem 1rem 1.2rem;
}
.hero-badge {
  display: inline-block;
  background: var(--grad1);
  color: white;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  padding: 0.4rem 1.4rem;
  border-radius: 999px;
  margin-bottom: 1.2rem;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 3rem;
  font-weight: 800;
  line-height: 1.1;
  margin-bottom: 0.6rem;
  background: linear-gradient(135deg, #667eea, #764ba2, #f5576c);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-heart {
  font-size: 4rem;
  margin: 0 auto 1rem;
  display: inline-block;
  animation: heartbeat 1.2s ease-in-out infinite;
}
@keyframes heartbeat {
  0%, 100% {
    transform: scale(1);
  }
  14%, 42% {
    transform: scale(1.25);
  }
  28%, 70% {
    transform: scale(1);
  }
  56% {
    transform: scale(1.12);
  }
}
.hero-sub {
  color: var(--muted);
  font-size: 1rem;
  font-weight: 400;
}
.hero-pills {
  display: flex;
  justify-content: center;
  gap: 0.8rem;
  flex-wrap: wrap;
  margin-top: 1.2rem;
}
.pill {
  background: white;
  border-radius: 999px;
  padding: 0.4rem 1rem;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text);
  box-shadow: 0 2px 12px rgba(100,100,200,0.12);
}
.pill span {
  font-weight: 800;
  background: var(--grad1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, #c4b5fd, #93c5fd, #f9a8d4, transparent);
  margin: 1.5rem 0;
  opacity: 0.6;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #f5f3ff 50%, #fdf2f8 100%) !important;
  border-right: 1px solid rgba(167,139,250,0.2) !important;
}
section[data-testid="stSidebar"] label {
  color: #4b5563 !important;
  font-size: 0.83rem !important;
  font-weight: 500 !important;
}
.sb-brand {
  text-align: center;
  padding: 1.5rem 0 1rem;
}
.sb-brand-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.3rem;
  font-weight: 800;
  background: var(--grad1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.sb-brand-sub {
  font-size: 0.72rem;
  color: var(--muted);
  margin-top: 0.2rem;
}
.sb-section {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: #a78bfa;
  border-bottom: 1px solid rgba(167,139,250,0.25);
  padding-bottom: 0.4rem;
  margin: 1.2rem 0 0.7rem;
}

/* ── CARDS ── */
.card {
  background: rgba(255,255,255,0.9);
  border: 1px solid rgba(167,139,250,0.2);
  border-radius: 20px;
  padding: 1.6rem;
  box-shadow: 0 4px 24px rgba(100,100,200,0.08);
  margin-bottom: 1rem;
}
.card-title {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  background: var(--grad1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 1.2rem;
}

/* ── METRIC GRID ── */
.mgrid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.7rem;
}
.mcell {
  background: linear-gradient(135deg, #f8f7ff, #fff);
  border: 1px solid rgba(167,139,250,0.2);
  border-radius: 12px;
  padding: 0.75rem 0.5rem;
  text-align: center;
}
.mcell-val {
  font-weight: 700;
  font-size: 0.95rem;
  color: var(--text);
}
.mcell-lbl {
  font-size: 0.62rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-top: 0.2rem;
}

/* ── RESULT ── */
.result-positive {
  background: linear-gradient(135deg, #fff0f6, #fff5f5);
  border: 1.5px solid rgba(245,87,108,0.35);
  border-radius: 20px;
  padding: 2rem 1.5rem;
  text-align: center;
}
.result-negative {
  background: linear-gradient(135deg, #f0fff8, #f0faff);
  border: 1.5px solid rgba(67,233,123,0.4);
  border-radius: 20px;
  padding: 2rem 1.5rem;
  text-align: center;
}
.r-icon { font-size: 3.2rem; margin-bottom: 0.5rem; }
.r-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  font-weight: 800;
  margin-bottom: 0.3rem;
}
.r-title-pos {
  background: var(--grad2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.r-title-neg {
  background: var(--grad4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.r-desc { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.3rem; }

/* ── PROB BARS ── */
.bar-wrap { margin: 0.6rem 0; }
.bar-row {
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: #6b7280;
  margin-bottom: 0.35rem;
  font-weight: 500;
}
.bar-bg {
  background: rgba(0,0,0,0.06);
  border-radius: 999px;
  height: 9px;
  overflow: hidden;
}
.bar-red {
  height: 100%;
  border-radius: 999px;
  background: var(--grad2);
}
.bar-green {
  height: 100%;
  border-radius: 999px;
  background: var(--grad4);
}
.bar-blue {
  height: 100%;
  border-radius: 999px;
  background: var(--grad3);
}

/* ── SCORE ROW ── */
.srow {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.8rem;
  margin-top: 1.2rem;
}
.sbox {
  background: white;
  border-radius: 14px;
  padding: 1rem 0.6rem;
  text-align: center;
  box-shadow: 0 2px 12px rgba(100,100,200,0.08);
  border: 1px solid rgba(167,139,250,0.15);
}
.sbox-val {
  font-family: 'Syne', sans-serif;
  font-size: 1.3rem;
  font-weight: 800;
}
.sbox-lbl {
  font-size: 0.63rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-top: 0.2rem;
}

/* ── IDLE ── */
.idle {
  text-align: center;
  padding: 4rem 1rem;
}
.idle-icon { font-size: 4rem; margin-bottom: 1rem; }
.idle-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.1rem;
  font-weight: 800;
  color: #c4b5fd;
}
.idle-sub { font-size: 0.83rem; color: #d1d5db; margin-top: 0.5rem; }

/* ── BUTTON ── */
.stButton > button {
  width: 100% !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 0.88rem !important;
  font-weight: 700 !important;
  letter-spacing: 1.5px !important;
  color: white !important;
  background: linear-gradient(135deg, #667eea, #764ba2) !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 0.85rem 1rem !important;
  margin-top: 1.2rem !important;
  box-shadow: 0 4px 20px rgba(102,126,234,0.4) !important;
  transition: all 0.25s ease !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #764ba2, #f5576c) !important;
  box-shadow: 0 6px 28px rgba(102,126,234,0.55) !important;
  transform: translateY(-2px) !important;
}

/* ── DISCLAIMER ── */
.disc {
  background: linear-gradient(135deg, #fffbeb, #fef9f0);
  border-left: 3px solid #fbbf24;
  border-radius: 0 12px 12px 0;
  padding: 0.85rem 1rem;
  margin-top: 1rem;
  font-size: 0.79rem;
  color: #78716c;
  line-height: 1.6;
}

/* ── FOOTER ── */
.footer {
  text-align: center;
  padding: 1.5rem 0;
  font-size: 0.75rem;
  color: #9ca3af;
}

@media (max-width: 900px) {
  .hero {
    padding: 1.5rem 0.8rem 1rem;
  }
  .hero-title {
    font-size: 2.4rem;
  }
  .hero-heart {
    font-size: 3.4rem;
  }
  .hero-pills {
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
  }
  .pill {
    width: 100%;
    max-width: 320px;
    text-align: center;
  }
  .card {
    padding: 1.2rem;
  }
  .mgrid {
    grid-template-columns: 1fr;
  }
  .mcell {
    padding: 0.9rem 0.6rem;
    min-width: 0;
  }
  .result-positive,
  .result-negative {
    padding: 1.4rem 1rem;
    width: 100%;
    overflow-wrap: break-word;
  }
  .bar-row {
    flex-wrap: wrap;
    gap: 0.4rem;
  }
  .srow {
    grid-template-columns: 1fr;
  }
  .sbox {
    padding: 0.95rem 0.75rem;
  }
  .pill {
    width: 100%;
    max-width: 100%;
  }
  .hero-pills {
    flex-direction: column;
    align-items: center;
  }
  section[data-testid="stSidebar"] {
    width: 100% !important;
    min-width: 0 !important;
  }
  .stButton > button {
    font-size: 0.82rem !important;
    padding: 0.8rem 1rem !important;
  }
}

@media (max-width: 640px) {
  .mgrid {
    grid-template-columns: 1fr;
  }
  .hero-title {
    font-size: 2rem;
  }
  .hero-sub {
    font-size: 0.94rem;
  }
  .hero-badge {
    padding: 0.35rem 1rem;
    font-size: 0.62rem;
  }
  .card-title {
    font-size: 0.7rem;
  }
  .bar-row {
    flex-direction: column;
    align-items: stretch;
  }
  .bar-row span {
    width: 100%;
  }
  .stApp, .hero, .card, .result-positive, .result-negative, .sbox, .pill {
    min-width: 0;
  }
}

hr { border-color: rgba(167,139,250,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ──
@st.cache_resource
def load_model():
    base_path = Path(__file__).resolve().parent
    model_path = base_path / 'heart_disease_model.pkl'
    scaler_path = base_path / 'scaler.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()


# ── HERO ──
st.markdown("""
<div class="hero">
  <div class="hero-heart">🫀</div>
  <div class="hero-badge">🤖 Artificial Intelligence · Medical Diagnostics</div>
  <div class="hero-title">AI CardioScan</div>
  <div class="hero-sub">Next-Generation Heart Disease Prediction Powered by Artificial Intelligence</div>
  <div class="hero-pills">
    <div class="pill">🩺 Care Focus <span>Heart health screening</span></div>
    <div class="pill">📌 Input <span>13 clinical metrics</span></div>
    <div class="pill">💡 Outcome <span>Risk-based guidance</span></div>
  </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

toggle_label = "Hide Patient Input Form" if st.session_state["show_patient_form"] else "Show Patient Input Form"
if st.button(toggle_label, key="toggle_patient_form_main"):
    st.session_state["show_patient_form"] = not st.session_state["show_patient_form"]

predict_btn = False
if not st.session_state["show_patient_form"]:
    st.info("Patient input form is hidden. Click the button again to show it.")
else:
    with st.expander("Patient Input Form", expanded=True):
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="sb-section">Demographics</div>', unsafe_allow_html=True)
            age = st.slider("Age (years)", 20, 80, 52, key="age")
            sex = st.selectbox("Biological Sex", [1, 0], index=0,
                               format_func=lambda x: "♂ Male" if x == 1 else "♀ Female", key="sex")

            st.markdown('<div class="sb-section">Cardiac Symptoms</div>', unsafe_allow_html=True)
            cp = st.selectbox("Chest Pain Type", [0,1,2,3], index=0,
                              format_func=lambda x: {
                                  0:"Typical Angina",
                                  1:"Atypical Angina",
                                  2:"Non-anginal Pain",
                                  3:"Asymptomatic"}[x], key="cp")
            thalach = st.slider("Max Heart Rate (bpm)", 60, 220, 168, key="thalach")
            exang = st.selectbox("Exercise Induced Angina", [0,1], index=0,
                                 format_func=lambda x: "Yes" if x == 1 else "No", key="exang")
            oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1, key="oldpeak")
            slope = st.selectbox("ST Segment Slope", [0,1,2], index=1,
                                 format_func=lambda x: {
                                     0:"Upsloping",
                                     1:"Flat",
                                     2:"Downsloping"}[x], key="slope")

        with right:
            st.markdown('<div class="sb-section">Lab Results</div>', unsafe_allow_html=True)
            trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 125, key="trestbps")
            chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 212, key="chol")
            fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl", [0,1], index=0,
                               format_func=lambda x: "Yes" if x == 1 else "No", key="fbs")
            restecg = st.selectbox("Resting ECG Result", [0,1,2], index=0,
                                   format_func=lambda x: {
                                       0:"Normal",
                                       1:"ST-T Abnormality",
                                       2:"LV Hypertrophy"}[x], key="restecg")

            st.markdown('<div class="sb-section">Advanced Tests</div>', unsafe_allow_html=True)
            ca = st.selectbox("Major Vessels Colored (0–4)", [0,1,2,3,4], index=0, key="ca")
            thal = st.selectbox("Thalassemia Type", [1,2,3], index=0,
                                format_func=lambda x: {
                                    1:"Normal",
                                    2:"Fixed Defect",
                                    3:"Reversible Defect"}[x], key="thal")

            predict_btn = st.button("🔍 Run AI Prediction", key="predict_btn")

try:
    age = st.session_state.age
    sex = st.session_state.sex
    cp = st.session_state.cp
    thalach = st.session_state.thalach
    exang = st.session_state.exang
    oldpeak = st.session_state.oldpeak
    slope = st.session_state.slope
    trestbps = st.session_state.trestbps
    chol = st.session_state.chol
    fbs = st.session_state.fbs
    restecg = st.session_state.restecg
    ca = st.session_state.ca
    thal = st.session_state.thal
except Exception:
    age = 52
    sex = 1
    cp = 0
    thalach = 168
    exang = 0
    oldpeak = 1.0
    slope = 1
    trestbps = 125
    chol = 212
    fbs = 0
    restecg = 0
    ca = 0
    thal = 1


# ── MAIN ──
col1, col2 = st.columns([1.3, 1], gap="large")

cp_map   = {0:"Typical", 1:"Atypical", 2:"Non-anginal", 3:"Asymp."}
thal_map = {1:"Normal", 2:"Fixed Def.", 3:"Reversible"}
ecg_map  = {0:"Normal", 1:"ST Abnorm.", 2:"LV Hyper."}

with col1:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Patient Data Summary</div>
      <div class="mgrid">
        <div class="mcell"><div class="mcell-val">{age} yrs</div><div class="mcell-lbl">Age</div></div>
        <div class="mcell"><div class="mcell-val">{"Male" if sex==1 else "Female"}</div><div class="mcell-lbl">Sex</div></div>
        <div class="mcell"><div class="mcell-val">{trestbps}</div><div class="mcell-lbl">BP mmHg</div></div>
        <div class="mcell"><div class="mcell-val">{chol}</div><div class="mcell-lbl">Cholesterol</div></div>
        <div class="mcell"><div class="mcell-val">{thalach} bpm</div><div class="mcell-lbl">Max HR</div></div>
        <div class="mcell"><div class="mcell-val">{oldpeak}</div><div class="mcell-lbl">ST Depress.</div></div>
        <div class="mcell"><div class="mcell-val">{cp_map[cp]}</div><div class="mcell-lbl">Chest Pain</div></div>
        <div class="mcell"><div class="mcell-val">{ca} vessels</div><div class="mcell-lbl">CA</div></div>
        <div class="mcell"><div class="mcell-val">{thal_map[thal]}</div><div class="mcell-lbl">Thalassemia</div></div>
        <div class="mcell"><div class="mcell-val">{ecg_map[restecg]}</div><div class="mcell-lbl">ECG</div></div>
        <div class="mcell"><div class="mcell-val">{"High" if fbs==1 else "Normal"}</div><div class="mcell-lbl">Fasting Sugar</div></div>
        <div class="mcell"><div class="mcell-val">{"Yes" if exang==1 else "No"}</div><div class="mcell-lbl">Ex. Angina</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disc">
      ⚠️ <strong style="color:#b45309">Medical Disclaimer:</strong>
      This AI tool is strictly for educational and research purposes.
      It does not replace professional medical advice or clinical diagnosis.
      Please consult a licensed cardiologist for any health concerns.
    </div>
    """, unsafe_allow_html=True)


with col2:
    st.markdown('<div style="font-size:0.65rem; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; color:#a78bfa; margin-bottom:1rem;">AI Diagnosis Result</div>', unsafe_allow_html=True)

    if predict_btn:
        input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                   thalach, exang, oldpeak, slope, ca, thal]],
                                 columns=['age','sex','cp','trestbps','chol','fbs',
                                          'restecg','thalach','exang','oldpeak',
                                          'slope','ca','thal'])
        scaled     = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        proba      = model.predict_proba(scaled)[0]
        d_pct      = proba[1] * 100
        h_pct      = proba[0] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
              <div class="r-icon">⚠️</div>
              <div class="r-title r-title-pos">Heart Disease Detected</div>
              <div class="r-desc">Analysis indicates elevated cardiovascular risk</div>
              <div class="bar-wrap">
                <div class="bar-row"><span>Disease Risk</span><span style="color:#f5576c;font-weight:700">{d_pct:.1f}%</span></div>
                <div class="bar-bg"><div class="bar-red" style="width:{d_pct}%"></div></div>
              </div>
              <div class="bar-wrap">
                <div class="bar-row"><span>Healthy Score</span><span style="color:#43e97b;font-weight:700">{h_pct:.1f}%</span></div>
                <div class="bar-bg"><div class="bar-green" style="width:{h_pct}%"></div></div>
              </div>
              <div class="disc">
                <strong>Recommendation:</strong> Please consult a healthcare provider for follow-up testing, manage blood pressure and cholesterol, and adopt a heart-healthy diet and exercise routine.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
              <div class="r-icon">✅</div>
              <div class="r-title r-title-neg">No Heart Disease</div>
              <div class="r-desc">Analysis indicates low cardiovascular risk</div>
              <div class="bar-wrap">
                <div class="bar-row"><span>Healthy Score</span><span style="color:#43e97b;font-weight:700">{h_pct:.1f}%</span></div>
                <div class="bar-bg"><div class="bar-green" style="width:{h_pct}%"></div></div>
              </div>
              <div class="bar-wrap">
                <div class="bar-row"><span>Disease Risk</span><span style="color:#f5576c;font-weight:700">{d_pct:.1f}%</span></div>
                <div class="bar-bg"><div class="bar-red" style="width:{d_pct}%"></div></div>
              </div>
              <div class="disc">
                <strong>Recommendation:</strong> Continue maintaining a heart-healthy lifestyle, monitor vital signs regularly, and schedule routine check-ups with your doctor.
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="srow">
          <div class="sbox">
            <div class="sbox-val" style="background:linear-gradient(135deg,#f093fb,#f5576c);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{d_pct:.1f}%</div>
            <div class="sbox-lbl">Risk Score</div>
          </div>
          <div class="sbox">
            <div class="sbox-val" style="background:linear-gradient(135deg,#43e97b,#38f9d7);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{h_pct:.1f}%</div>
            <div class="sbox-lbl">Health Score</div>
          </div>
          <div class="sbox">
            <div class="sbox-val" style="background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent">13</div>
            <div class="sbox-lbl">Clinical Inputs</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3.5rem 1rem;">
          <div style="font-size:3.5rem; margin-bottom:1rem;">🫀</div>
          <div style="font-family:Syne,sans-serif; font-size:1rem; font-weight:800;
                      background:linear-gradient(135deg,#667eea,#764ba2);
                      -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Awaiting AI Analysis
          </div>
          <div style="font-size:0.82rem; color:#9ca3af; margin-top:0.5rem;">
            Complete the patient form and click<br>
            <strong style="color:#a78bfa">Run AI Prediction</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
  🫀 AI CardioScan · Heart disease risk evaluation tool
</div>
""", unsafe_allow_html=True)