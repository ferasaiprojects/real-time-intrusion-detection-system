# dashboard.py — Redesigned AI-IDS Streamlit dashboard (enhanced background + UI tweaks)
import os
import json
import joblib
from typing import Optional

import pandas as pd
import streamlit as st

from pcap_feature_extractor import extract_features_from_pcap

# ---------- Page config ----------
st.set_page_config(page_title="AI Intrusion Detection System", page_icon="🛡️", layout="wide")

# ---------- Paths & constants ----------
MODEL_PATH = os.path.join("models", "ids_model.pkl")
DEFAULT_PCAP = os.path.join("sample_pcaps", "2026-02-28-traffic-analysis-exercise.pcap")
EXPECTED_FEATURES = 42

# ---------- Sidebar theme + animation toggle ----------
with st.sidebar:
    ui_theme = st.selectbox("UI Theme", ["Soft-Dark (recommended)", "Light (high-contrast)"], index=0)
    animated_bg = st.checkbox("Animated background", value=True)
    reduce_motion = st.checkbox("Prefer reduced motion (static bg)", value=False)

# ---------- Enhanced Unified CSS (cards, grid, badges, animated background) ----------
# --- tweak these variables to change the background look quickly ---
_COMMON_CSS = r"""
<style>
:root{
  /* primary accents - tweak here */
  --accent1: #4f6ef6;
  --accent2: #0fb6b0;
  --bg-1: #061224;
  --bg-2: #03202b;
  --glass-alpha: 0.12;
  --glass-border: rgba(255,255,255,0.06);
  --noise-alpha: 0.03;
  --card-radius: 14px;
  --card-padding: 14px;
  --muted-dark: #9fb4d8;
  --muted-light: #475569;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}

/* page / layout */
.stApp {
  position: relative;
  z-index: 1;
  padding-top: 24px;
  padding-bottom: 48px;
  overflow-x: hidden;
}

/* Background layers (full-bleed, sits behind the app) */
.background {
  position: fixed;
  inset: 0;
  z-index: -2;
  pointer-events: none;
  background: linear-gradient(120deg, var(--bg-1) 0%, var(--bg-2) 60%);
  transition: background 400ms linear;
}

/* animated gradient layer */
.bg-animated {
  position: absolute; inset: 0; opacity: 0.7; z-index: -2;
  background: radial-gradient(800px 400px at 10% 30%, rgba(79,110,246,0.12), transparent 10%),
              radial-gradient(600px 300px at 90% 70%, rgba(15,182,176,0.10), transparent 12%),
              transparent;
  filter: blur(40px) saturate(1.05);
  transform: translate3d(0,0,0);
  animation: gradientShift 18s ease-in-out infinite;
  will-change: transform, opacity;
}

/* subtle slow movement */
@keyframes gradientShift {
  0% { transform: translateY(0px) translateX(0px) scale(1); opacity:0.78; }
  33% { transform: translateY(-18px) translateX(12px) scale(1.02); opacity:0.86; }
  66% { transform: translateY(14px) translateX(-14px) scale(0.98); opacity:0.74; }
  100% { transform: translateY(0px) translateX(0px) scale(1); opacity:0.78; }
}

/* tonal noise overlay (very light) */
.bg-noise {
  position:absolute; inset:0; z-index:-1; pointer-events:none;
  background-image:
    linear-gradient(transparent, rgba(0,0,0,var(--noise-alpha)));
  mix-blend-mode: screen;
  opacity: 0.25;
}

/* glass panel / cards */
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 18px; }
.summary-card {
  border-radius: var(--card-radius);
  padding: var(--card-padding);
  position: relative;
  overflow: hidden;
  min-height: 96px;
  backdrop-filter: blur(6px) saturate(1.05);
  -webkit-backdrop-filter: blur(6px) saturate(1.05);
  border: 1px solid var(--glass-border);
  background: linear-gradient(180deg, rgba(255,255,255,var(--glass-alpha)), rgba(255,255,255,0.02));
  box-shadow: 0 6px 24px rgba(2,6,23,0.45);
}
.summary-card .title { font-size: 0.95rem; font-weight: 600; margin-bottom: 6px; }
.summary-card .value { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.3px; }
.summary-card .sub { font-size: 0.85rem; color: rgba(0,0,0,0.45); }

/* glass highlight circle */
.summary-card::after{
  content: "";
  position:absolute; right:-60px; top:-40px; width:160px; height:160px; opacity:0.07; transform:rotate(25deg);
  border-radius: 50%;
}

/* badged label */
.badge { display:inline-block; padding:6px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }

/* Accent stripe */
.stripe { height:6px; border-radius:6px; margin-bottom:8px; }
.stripe.accent { background: linear-gradient(90deg,var(--accent1),var(--accent2)); }

/* Table header minor tweak (conservative) */
div[data-testid="stDataFrameContainer"] table thead th { font-weight:600; }

/* small responsive tweaks */
@media (max-width:880px){ .card-grid { grid-template-columns: 1fr; } }

/* THEME OVERRIDES */
/* Soft-Dark theme */
.soft-dark .stApp { color: #d4e8ff !important; }
.soft-dark .summary-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  box-shadow: 0 10px 40px rgba(2,6,23,0.6);
  color: #d4e8ff;
}
.soft-dark .summary-card::after { background: radial-gradient(circle at 30% 20%, var(--accent1), transparent 40%); }
.soft-dark .badge { color: #041022; }

/* Light theme */
.light .stApp { color: #0f1724 !important; }
.light .summary-card {
  background: linear-gradient(180deg,#ffffff,#f7f9fb);
  box-shadow: 0 6px 18px rgba(16,24,40,0.06);
  color: #0f1724;
}
.light .summary-card::after { background: radial-gradient(circle at 30% 20%, var(--accent2), transparent 40%); }

/* Accessibility: respect reduced motion */
@media (prefers-reduced-motion: reduce) {
  .bg-animated { animation: none !important; transform: none !important; }
}

/* explicit user-controlled reduced-motion/static fallbacks */
.static-bg .bg-animated { animation: none !important; transform: none !important; opacity:0.7; }

/* small header hero */
.hero {
  padding: 18px 18px;
  border-radius: 12px;
  margin-bottom: 18px;
  display:flex;
  gap: 10px;
  align-items:center;
  justify-content: space-between;
}
.hero .title { font-size:1.25rem; font-weight:700; letter-spacing:-0.2px; }
.hero .subtitle { font-size:0.95rem; color: rgba(255,255,255,0.8); font-weight:500; }

/* fallback for older browsers */
@supports not ((-webkit-backdrop-filter: blur(1px)) or (backdrop-filter: blur(1px))) {
  .summary-card { background: linear-gradient(180deg, rgba(0,0,0,0.045), rgba(255,255,255,0.02)); }
}
</style>
"""

# ---------- Inject background & CSS ----------
# Build the background HTML. We place it before the app markup so it's visually behind the app.
bg_html = """
<div class="background">
  <div class="bg-animated" aria-hidden="true"></div>
  <div class="bg-noise" aria-hidden="true"></div>
</div>
"""
st.markdown(_COMMON_CSS, unsafe_allow_html=True)

# wrap app in theme container (keeps scoping consistent)
if ui_theme.startswith("Soft"):
    wrapper_open = '<div class="soft-dark %s">' % ("" if animated_bg and not reduce_motion else "static-bg")
else:
    wrapper_open = '<div class="light %s">' % ("" if animated_bg and not reduce_motion else "static-bg")

# only insert the background markup if animations are enabled; otherwise insert a static background container
if animated_bg and not reduce_motion:
    st.markdown(bg_html, unsafe_allow_html=True)
else:
    # static fallback: single subtle radial highlights (no animation)
    static_bg_html = """
    <div class="background" style="background: linear-gradient(120deg, #08172a 0%, #01202a 70%);">
      <div class="bg-noise" aria-hidden="true"></div>
    </div>
    """
    st.markdown(static_bg_html, unsafe_allow_html=True)

st.markdown(wrapper_open, unsafe_allow_html=True)

# ---------- Utilities (unchanged) ----------
@st.cache_resource
def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)


def safe_predict(model, X: pd.DataFrame):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X_proc = X.copy().fillna(0)
    try:
        proba = model.predict_proba(X_proc)
        if getattr(proba, "ndim", 1) == 1:
            probs = [float(p) for p in proba]
        else:
            probs = [float(p) for p in proba[:, 1]]
    except Exception:
        try:
            preds = model.predict(X_proc)
            probs = [1.0 if int(p) == 1 else 0.0 for p in preds]
        except Exception:
            probs = [0.0] * len(X_proc)
    preds_int = [1 if (p is not None and p >= 0.5) else 0 for p in probs]
    return probs, preds_int


def pad_features(df: pd.DataFrame, expected_cols: int = EXPECTED_FEATURES) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    cur = df.shape[1]
    if cur < expected_cols:
        for i in range(expected_cols - cur):
            df[f"dummy_{i}"] = 0
    return df


def map_severity_by_prob(prob: float) -> str:
    if prob >= 0.9:
        return "HIGH"
    if prob >= 0.7:
        return "MEDIUM"
    if prob >= 0.4:
        return "LOW"
    return "NORMAL"

# ---------- Sidebar controls ----------
with st.sidebar:
    st.title("⚙️ Controls")
    label_preset = st.selectbox("Label Preset",
                                ["Binary (BENIGN / MALICIOUS)", "Severity (NORMAL/LOW/MEDIUM/HIGH)", "Custom labels"], index=0)
    custom_label_0 = None
    custom_label_1 = None
    if label_preset == "Custom labels":
        custom_label_0 = st.text_input("Label for prediction 0", value="BENIGN")
        custom_label_1 = st.text_input("Label for prediction 1", value="MALICIOUS")

    threshold = st.slider("Probability threshold (suspicious)", 0.0, 1.0, 0.60, 0.01)
    prob_filter = st.slider("Show flows with prob >=", 0.0, 1.0, 0.0, 0.01)
    top_n = st.number_input("Top N suspicious flows", min_value=5, max_value=500, value=20)
    show_raw = st.checkbox("Show raw features (large)", value=False)
    highlight_suspicious = st.checkbox("Highlight suspicious rows", value=True)
    run_on_upload = st.checkbox("Auto-run detection on upload", value=True)

    st.markdown("---")
    if st.button("Clear cached model"):
        try:
            st.cache_resource.clear()
        except Exception:
            st.session_state.pop("loaded_model", None)
        st.experimental_rerun()

# ---------- Load model ----------
try:
    model = load_model()
except FileNotFoundError:
    st.sidebar.error(f"Model missing at {MODEL_PATH}. Upload or place the model file in 'models/'.")
    model = None
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model = None

# ---------- Header / Upload area ----------
st.markdown("""
# 🛡️ AI-Powered Intrusion Detection System
Upload a PCAP, choose a label scheme, and run detection. The dashboard is responsive and organized into clear sections.
""")

# small hero with glass background
st.markdown(
    """
    <div class="summary-card hero" style="display:flex; align-items:center; justify-content:space-between;">
      <div>
        <div class="title">AI Intrusion Detection</div>
        <div class="sub">Upload PCAP → Extract features → Run model</div>
      </div>
      <div style="text-align:right;">
        <div style="font-weight:600">Theme: %s</div>
        <div style="font-size:0.88rem; color: rgba(255,255,255,0.65)">Animated background: %s</div>
      </div>
    </div>
    """ % (ui_theme, "On" if animated_bg and not reduce_motion else "Off"),
    unsafe_allow_html=True,
)

left, right = st.columns([2, 1])
with left:
    st.subheader("1) Upload PCAP")
    uploaded = st.file_uploader("Upload PCAP (.pcap)", type=["pcap"])
    sample_choice = st.selectbox("Or choose sample", ["Default sample (recommended)", "No sample / upload only"])
    if uploaded is not None:
        pcap_path = "temp_uploaded.pcap"
        with open(pcap_path, "wb") as f:
            f.write(uploaded.read())
    else:
        pcap_path = DEFAULT_PCAP if sample_choice.startswith("Default") else None
    st.markdown("**PCAP:**")
    st.write(pcap_path if pcap_path else "No PCAP selected")

with right:
    st.subheader("Model & Quick Info")
    st.markdown(""" 
    <div class="summary-card" style="padding:12px">
      <div class="stripe accent"></div>
      <div class="title">Model Path</div>
      <div class="value">%s</div>
      <div class="sub">Threshold: %.2f</div>
    </div>
    """ % (MODEL_PATH, threshold), unsafe_allow_html=True)

# ---------- Run logic ----------
do_run = (run_on_upload and pcap_path is not None) or st.button("Extract & Predict ▶️")

if not pcap_path:
    st.info("No PCAP selected — upload or choose sample to run detection.")

# ---------- When running: extract, predict, show visuals ----------
if do_run and pcap_path and model is not None:
    with st.spinner("Extracting features from PCAP... ⛏️"):
        try:
            df = extract_features_from_pcap(pcap_path)
            if df is None or df.shape[0] == 0:
                st.warning("No flows parsed from PCAP. Check the file content.")
                st.stop()
        except Exception as e:
            st.exception(f"Feature extraction failed: {e}")
            st.stop()

    df = pad_features(df, EXPECTED_FEATURES)

    with st.spinner("Running model inference... 🤖"):
        probs, preds_int = safe_predict(model, df)

    results = df.copy().reset_index(drop=True)
    results["malicious_probability"] = [float(p) for p in probs]
    results["prediction_int"] = [int(x) for x in preds_int]
    results["is_suspicious"] = results["malicious_probability"] >= threshold

    # labels
    if label_preset == "Binary (BENIGN / MALICIOUS)":
        lbl0, lbl1 = "BENIGN", "MALICIOUS"
        results["label"] = results["prediction_int"].map({0: lbl0, 1: lbl1})
    elif label_preset == "Severity (NORMAL/LOW/MEDIUM/HIGH)":
        results["label"] = results["malicious_probability"].apply(map_severity_by_prob)
    else:
        lbl0 = custom_label_0 if custom_label_0 else "BENIGN"
        lbl1 = custom_label_1 if custom_label_1 else "MALICIOUS"
        results["label"] = results["prediction_int"].map({0: lbl0, 1: lbl1})

    total = len(results)
    attacks = int(results["prediction_int"].sum())
    normals = total - attacks
    attack_ratio = attacks / total if total else 0.0

    if attack_ratio < 0.05:
        threat = "VERY LOW"
        color_emoji = "🟢"
    elif attack_ratio < 0.15:
        threat = "LOW"
        color_emoji = "🟢"
    elif attack_ratio < 0.3:
        threat = "MEDIUM"
        color_emoji = "🟠"
    else:
        threat = "HIGH"
        color_emoji = "🔴"

    # ---- Summary cards in a neat grid ----
    st.markdown("<div class='card-grid'>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='summary-card'>
          <div class='stripe accent'></div>
          <div class='title'>Total Flows</div>
          <div class='value'>{total}</div>
          <div class='sub'>Total parsed flows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class='summary-card'>
          <div class='stripe accent'></div>
          <div class='title'>Benign / Normal</div>
          <div class='value'>{normals}</div>
          <div class='sub'>Non-alert flows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class='summary-card'>
          <div class='stripe accent'></div>
          <div class='title'>Suspicious / Alerts</div>
          <div class='value'>{attacks}</div>
          <div class='sub'>Detected suspicious flows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class='summary-card'>
          <div class='stripe accent'></div>
          <div class='title'>Threat Level</div>
          <div class='value'>{threat} {color_emoji}</div>
          <div class='sub'>{attack_ratio*100:.2f}% of flows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Charts + quick visualizations
    st.markdown("## Traffic Overview")
    dist_df = results.groupby("label").size().reset_index(name="count").sort_values("count", ascending=False)
    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            import altair as alt
            chart = alt.Chart(results).mark_bar().encode(x=alt.X("label:N", sort="-y"), y="count()")
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.bar_chart(dist_df.set_index("label"))

    with col2:
        st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
        st.markdown(f"**Top suspicious (prob >= {prob_filter:.2f})**")
        top_count = results[results["malicious_probability"] >= prob_filter].shape[0]
        st.markdown(f"\n\n### {top_count} matching flows")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## Probability Distribution")
    try:
        import altair as alt
        hist = alt.Chart(results).mark_bar().encode(x=alt.X("malicious_probability:Q", bin=alt.Bin(maxbins=40)), y='count()').properties(height=220)
        st.altair_chart(hist, use_container_width=True)
    except Exception:
        st.bar_chart(results['malicious_probability'].value_counts().sort_index())

    st.markdown("## Attack Probability Over Sample (first 200 rows)")
    st.line_chart(results["malicious_probability"].head(200))

    # Interactive table
    st.markdown("## Detected Flows — Interactive View")
    filtered = results[results["malicious_probability"] >= prob_filter].sort_values("malicious_probability", ascending=False)
    if filtered.shape[0] == 0:
        st.info("No flows match the current probability filter. Lower the filter or choose a different PCAP.")
    else:
        top = filtered.head(top_n).copy()
        if highlight_suspicious:
            try:
                styled = top.style.apply(lambda r: ["background-color: rgba(255,0,0,0.12)" if r["is_suspicious"] else "" for _ in r], axis=1)
                st.dataframe(styled, height=400)
            except Exception:
                st.dataframe(top, height=400)
        else:
            display_cols = ["label", "malicious_probability", "prediction_int"] + ([c for c in top.columns if c.startswith("dummy_")][:3])
            st.dataframe(top[display_cols], height=400)

    # Inspect single flow / JSON view
    st.markdown("## Inspect a single flow / JSON view")
    idx = st.number_input("Row index (0-based)", 0, max(0, total - 1), 0)
    if total > 0:
        row = results.iloc[int(idx)].to_dict()
        st.json(row)

    # Export
    st.markdown("## Export results")
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    json_str = results.to_json(orient="records", indent=2)
    st.download_button("📥 Download CSV", csv_bytes, file_name="ai_ids_results.csv", mime="text/csv")
    st.download_button("📥 Download JSON", json_str.encode("utf-8"), file_name="ai_ids_results.json", mime="application/json")

    if st.button("💾 Save results to predictions.csv (server)"):
        out_path = "predictions.csv"
        results.to_csv(out_path, index=False)
        st.success(f"Saved to {out_path}")

    if show_raw:
        st.markdown("---")
        st.subheader("Raw extracted features")
        st.dataframe(df, height=300)

    st.success("Detection completed ✅")

else:
    st.markdown("<div class='summary-card' style='padding:14px'>", unsafe_allow_html=True)
    st.markdown("## Welcome 👋")
    st.markdown(
        "This dashboard extracts features from PCAPs and runs a trained ML model to detect suspicious activity.\n\nUse the controls on the left to configure labels, thresholds and run detection.")
    st.markdown("</div>", unsafe_allow_html=True)

# close theme wrapper
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("Made with ❤️ by the AI-IDS team")