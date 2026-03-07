# dashboard.py — AI-IDS Streamlit dashboard (multi-sample, thumbnails, keyboard shortcut, optional AgGrid)
# Drop this file into your app folder and run: streamlit run dashboard.py

import os
import glob
import tempfile
import joblib
from typing import Optional, List, Dict, Any
import time

import pandas as pd
import streamlit as st

from pcap_feature_extractor import extract_features_from_pcap

# ---------- Page config ----------
st.set_page_config(page_title="AI Intrusion Detection System", page_icon="🛡️", layout="wide")

# ---------- Paths & constants ----------
MODEL_PATH = os.path.join("models", "ids_model.pkl")
DEFAULT_PCAP_FILENAME = "2026-02-28-traffic-analysis-exercise.pcap"
SAMPLE_DIR_CANDIDATES = ["sample_pcap", "sample_pcaps", "sample_pcap/", "sample_pcaps/", "sample_pcap_files"]
EXPECTED_FEATURES = 42
MAX_TOP_ROWS = 500

# ---------- Utilities ----------
def clear_model_cache():
    for key in ["loaded_model", "model_cached", "model"]:
        if key in st.session_state:
            try:
                del st.session_state[key]
            except Exception:
                pass
    try:
        st.cache_resource.clear()
    except Exception:
        try:
            st.experimental_memo.clear()
        except Exception:
            try:
                st.cache_data.clear()
            except Exception:
                pass


def find_sample_pcaps(dirs: List[str] = SAMPLE_DIR_CANDIDATES, pattern: str = "*.pcap") -> List[str]:
    found = []
    for d in dirs:
        if os.path.isdir(d):
            entries = glob.glob(os.path.join(d, pattern))
            for e in entries:
                if os.path.isfile(e):
                    found.append(os.path.abspath(e))
    for d in dirs:
        pattern_recursive = os.path.join(d, "**", pattern)
        matches = glob.glob(pattern_recursive, recursive=True)
        for m in matches:
            if os.path.isfile(m):
                ab = os.path.abspath(m)
                if ab not in found:
                    found.append(ab)
    top_level = glob.glob(pattern)
    for t in top_level:
        if os.path.isfile(t):
            ab = os.path.abspath(t)
            if ab not in found:
                found.append(ab)
    found = sorted(found)
    return found


# cache sample summary to avoid repeated heavy parsing
@st.cache_data(ttl=300)
def get_sample_summary(path: str) -> Dict[str, Any]:
    """Return a small summary for a sample pcap used as a thumbnail.
    Attempts to call extract_features_from_pcap for a quick flow preview; falls back to file metadata.
    """
    summary: Dict[str, Any] = {}
    try:
        summary["path"] = os.path.abspath(path)
        summary["name"] = os.path.basename(path)
        summary["size_bytes"] = os.path.getsize(path)
        summary["modified_time"] = time.ctime(os.path.getmtime(path))
        # attempt to extract features (may be heavy for many files, but cached)
        try:
            df = extract_features_from_pcap(path)
            if isinstance(df, pd.DataFrame) and df.shape[0] > 0:
                summary["num_flows"] = int(df.shape[0])
                # grab a compact preview of first row (top 6 columns)
                first_row = df.reset_index(drop=True).iloc[0]
                preview = {}
                for i, col in enumerate(first_row.index):
                    if i >= 6:
                        break
                    val = first_row[col]
                    # convert numpy types
                    try:
                        preview[col] = float(val) if pd.api.types.is_numeric_dtype(type(val)) else str(val)
                    except Exception:
                        preview[col] = str(val)
                summary["preview"] = preview
            else:
                summary["num_flows"] = 0
                summary["preview"] = {}
        except Exception as e:
            summary["num_flows"] = None
            summary["preview_error"] = str(e)
    except Exception as e:
        summary = {"path": path, "name": os.path.basename(path), "error": str(e)}
    return summary


# ---------- CSS for theme & headers ----------
_COMMON_CSS = r"""
<style>
:root{ --accent1: #0b76ff; --accent2: #00b894; --muted-dark: #9fb4d8; --muted-light: #475569; }
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 18px; }
.summary-card { border-radius: 12px; padding: 12px; position: relative; overflow: hidden; min-height: 88px; }
.summary-card .title { font-size: 0.92rem; font-weight: 700; margin-bottom: 6px; }
.summary-card .value { font-size: 1.55rem; font-weight: 800; }
.summary-card .sub { font-size: 0.82rem; color: var(--muted, rgba(0,0,0,0.45)); }
.section-header { display:flex; align-items:center; gap:12px; padding:10px 12px; border-radius:8px; border-left:6px solid var(--accent1); background: rgba(255,255,255,0.02); margin-bottom:8px; }
.section-header h3 { margin:0; font-size:1.05rem; }
.sample-card { border-radius:8px; padding:8px; background: rgba(255,255,255,0.01); border:1px solid rgba(255,255,255,0.03); margin-bottom:8px; }
.sample-card .meta { font-size:0.82rem; color:var(--muted-light); }
@media (max-width:880px){ .card-grid { grid-template-columns: 1fr; } }
</style>
"""

_DARK_CSS = """
<style>
.stApp { background: linear-gradient(180deg,#041022,#07182a) !important; color: #e6f0ff !important; }
.summary-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow: 0 8px 30px rgba(2,6,23,0.6); color: #e6f0ff; }
.section-header { background: rgba(255,255,255,0.02); }
.sample-card { background: rgba(255,255,255,0.02); }
</style>
"""

_LIGHT_CSS = """
<style>
.stApp { background: linear-gradient(180deg,#f8fafc,#eef2ff) !important; color: #071023 !important; }
.summary-card { background: linear-gradient(180deg,#ffffff,#f7f9fb); box-shadow: 0 6px 18px rgba(16,24,40,0.06); color: #071023; }
.section-header { background: rgba(0,0,0,0.02); }
.sample-card { background: rgba(255,255,255,0.98); }
</style>
"""

# ---------- Model loading ----------
@st.cache_resource
def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

# ---------- Prediction helpers ----------
def safe_predict(model, X: pd.DataFrame):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X_proc = X.copy().fillna(0)
    try:
        proba = model.predict_proba(X_proc)
        if hasattr(proba, "ndim") and proba.ndim == 1:
            probs = [float(p) for p in proba]
        else:
            try:
                probs = [float(p) for p in proba[:, 1]]
            except Exception:
                probs = [float(p) for p in proba[:, 0]]
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
            col_name = f"dummy_{i}"
            if col_name not in df.columns:
                df[col_name] = 0
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def map_severity_by_prob(prob: float) -> str:
    if prob >= 0.9:
        return "HIGH"
    if prob >= 0.7:
        return "MEDIUM"
    if prob >= 0.4:
        return "LOW"
    return "NORMAL"

# ---------- Sidebar: theme + controls ----------
with st.sidebar:
    st.title("⚙️ Controls")
    ui_theme = st.selectbox("UI Theme", ["Soft-Dark (recommended)", "Light (high-contrast)"], index=0)
    background_style = st.selectbox("Background", ["Professional gradient (recommended)", "Plain"], index=0)

    st.markdown("---")
    label_preset = st.selectbox("Label Preset",
                                ["Binary (BENIGN / MALICIOUS)", "Severity (NORMAL/LOW/MEDIUM/HIGH)", "Custom labels"],
                                index=0)
    custom_label_0 = None
    custom_label_1 = None
    if label_preset == "Custom labels":
        custom_label_0 = st.text_input("Label for prediction 0", value="BENIGN")
        custom_label_1 = st.text_input("Label for prediction 1", value="MALICIOUS")

    threshold = st.slider("Probability threshold (suspicious)", 0.0, 1.0, 0.60, 0.01)
    prob_filter = st.slider("Show flows with prob >=", 0.0, 1.0, 0.0, 0.01)
    top_n = int(st.number_input("Top N suspicious flows", min_value=5, max_value=MAX_TOP_ROWS, value=20))
    show_raw = st.checkbox("Show raw features (large)", value=False)
    highlight_suspicious = st.checkbox("Highlight suspicious rows", value=True)
    run_on_upload = st.checkbox("Auto-run detection on upload", value=True)

    st.markdown("---")
    if st.button("Clear cached model"):
        clear_model_cache()
        st.experimental_rerun()

# ---------- Inject CSS & theme wrapper ----------
st.markdown(_COMMON_CSS, unsafe_allow_html=True)
if ui_theme.startswith("Soft"):
    st.markdown('<div class="soft-dark">', unsafe_allow_html=True)
    st.markdown(_DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown('<div class="light">', unsafe_allow_html=True)
    st.markdown(_LIGHT_CSS, unsafe_allow_html=True)

if background_style.startswith("Professional"):
    st.markdown("""
    <style>
    body > div[role="application"] { background-image: radial-gradient(circle at 10% 10%, rgba(11,118,255,0.03), transparent 10%), linear-gradient(180deg, rgba(3,10,20,0.6), rgba(3,10,20,0.85)); background-attachment: fixed; }
    </style>
    """, unsafe_allow_html=True)

# ---------- Load model ----------
try:
    model = load_model()
except FileNotFoundError:
    st.sidebar.error(f"Model missing at {MODEL_PATH}. Upload or place the model file in 'models'.")
    model = None
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model = None

# ---------- Try to import AgGrid (optional) ----------
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    aggrid_available = True
except Exception:
    aggrid_available = False

# ---------- Main UI ----------
st.markdown("# 🛡️ AI-Powered Intrusion Detection System")
st.markdown("Upload a PCAP, choose label scheme and run detection. The dashboard focuses on readable, accessible statistics.")

left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="section-header"><h3>1) PCAP Source</h3></div>', unsafe_allow_html=True)

    sample_list = find_sample_pcaps()

    source_mode = st.radio("Choose PCAP source:", ["Use default sample (recommended)", "Upload custom PCAP"], index=0)

    pcap_path = None
    default_selected = None

    if source_mode.startswith("Use default"):
        if sample_list:
            filenames = [os.path.basename(p) for p in sample_list]
            selected_idx = st.session_state.get("selected_default_idx", 0)
            if selected_idx >= len(filenames):
                selected_idx = 0
            sel = st.selectbox("Select default sample:", options=filenames, index=selected_idx)
            sel_idx = filenames.index(sel)
            st.session_state["selected_default_idx"] = sel_idx
            default_selected = sample_list[sel_idx]
            pcap_path = default_selected
            st.success(f"Using default sample: {os.path.basename(pcap_path)}")
            st.write(pcap_path)
        else:
            st.warning("No sample PCAP files found in sample directories. Upload a PCAP or place sample files in one of the sample directories.")
            uploaded = st.file_uploader("Upload PCAP (.pcap)", type=["pcap"])
            if uploaded is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap")
                tmp.write(uploaded.read())
                tmp.flush()
                tmp.close()
                pcap_path = tmp.name
    else:
        uploaded = st.file_uploader("Upload PCAP (.pcap)", type=["pcap"])  # shown only in upload mode
        if uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap")
            tmp.write(uploaded.read())
            tmp.flush()
            tmp.close()
            pcap_path = tmp.name
        else:
            st.info("No custom PCAP uploaded yet. Choose 'Use default sample' to run immediately using a sample file.")

    st.markdown("**PCAP path:**")
    st.write(pcap_path if pcap_path else "No PCAP selected")

    # quick hint for keyboard shortcut
    st.markdown("Press **R** to re-run inference (keyboard shortcut)")

with right:
    st.markdown('<div class="section-header"><h3>Samples</h3></div>', unsafe_allow_html=True)

    if sample_list:
        # Display a compact scrollable list of sample cards
        for sp in sample_list:
            s = get_sample_summary(sp)
            # make the currently selected sample visually distinct
            highlight = (pcap_path == s.get("path"))
            card_html = f"""
            <div class='sample-card' style='border: 1px solid {'#0b76ff' if highlight else 'rgba(255,255,255,0.03)'};'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='font-weight:700'>{s.get('name')}</div>
                <div class='meta'>{s.get('num_flows', '?')} flows</div>
              </div>
              <div style='font-size:0.85rem; margin-top:6px;'>Size: {s.get('size_bytes', '?')} bytes<br>Modified: {s.get('modified_time', '-')}
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            # show small preview table for first flow if available
            if s.get("preview"):
                try:
                    preview_df = pd.DataFrame([s.get("preview")])
                    st.dataframe(preview_df, height=90)
                except Exception:
                    st.write(s.get("preview"))
            elif s.get("preview_error"):
                st.caption("Preview error: " + str(s.get("preview_error")))
    else:
        st.info("No sample files found. Place .pcap files in one of the sample directories or upload a custom PCAP.")

    st.markdown('<div class="section-header"><h3>Model & Quick Info</h3></div>', unsafe_allow_html=True)
    model_info_html = f"""
    <div class="summary-card" style="padding:12px">
      <div class="title">Model Path</div>
      <div class="value" title="{MODEL_PATH}">{os.path.basename(MODEL_PATH)}</div>
      <div class="sub">Threshold: {threshold:.2f}</div>
    </div>
    """
    st.markdown(model_info_html, unsafe_allow_html=True)

# ---------- Run logic ----------
do_run = (run_on_upload and pcap_path is not None) or st.button("Extract & Predict ▶️")

if not pcap_path:
    st.info("No PCAP selected — use the default sample or upload a custom file to run detection.")

# ---------- Run: extract, predict, visuals ----------
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

    # ---- KPI cards ----
    st.markdown('<div class="section-header"><h3>Summary</h3></div>', unsafe_allow_html=True)
    st.markdown("<div class='card-grid'>", unsafe_allow_html=True)
    kpis = [
        ("Total Flows", total, "Total parsed flows"),
        ("Benign / Normal", normals, "Non-alert flows"),
        ("Suspicious / Alerts", attacks, "Detected suspicious flows"),
        ("Threat Level", f"{threat} {color_emoji}", f"{attack_ratio*100:.2f}% of flows"),
    ]
    for title, value, sub in kpis:
        st.markdown(
            f"""
            <div class='summary-card'>
              <div class='title'>{title}</div>
              <div class='value'>{value}</div>
              <div class='sub'>{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Charts & Table
    st.markdown('<div class="section-header"><h3>Traffic Overview</h3></div>', unsafe_allow_html=True)
    try:
        import altair as alt
        label_counts = results.groupby("label").size().reset_index(name="count").sort_values("count", ascending=False)
        bar = (
            alt.Chart(label_counts)
            .mark_bar()
            .encode(x=alt.X("label:N", title="Label", sort="-y"), y=alt.Y("count:Q", title="Count"), tooltip=["label", "count"])
            .properties(height=220)
        )
        st.altair_chart(bar, use_container_width=True)
    except Exception:
        st.bar_chart(results["label"].value_counts())

    st.markdown('<div class="section-header"><h3>Probability Distribution</h3></div>', unsafe_allow_html=True)
    try:
        import altair as alt
        hist = (
            alt.Chart(results)
            .mark_bar()
            .encode(
                x=alt.X("malicious_probability:Q", bin=alt.Bin(maxbins=40), title="Malicious probability"),
                y=alt.Y("count():Q", title="Flows"),
                tooltip=[alt.Tooltip("count()", title="Flows")]
            )
            .properties(height=220)
        )
        st.altair_chart(hist, use_container_width=True)
    except Exception:
        st.bar_chart(results["malicious_probability"].value_counts().sort_index())

    st.markdown('<div class="section-header"><h3>Attack Probability Over Sample</h3></div>', unsafe_allow_html=True)
    st.line_chart(results["malicious_probability"].head(200))

    st.markdown('<div class="section-header"><h3>Detected Flows — Interactive View</h3></div>', unsafe_allow_html=True)
    filtered = results[results["malicious_probability"] >= prob_filter].sort_values("malicious_probability", ascending=False)
    if filtered.shape[0] == 0:
        st.info("No flows match the current probability filter. Lower the filter or choose a different PCAP.")
    else:
        top = filtered.head(top_n).copy()
        display_cols = ["label", "malicious_probability", "prediction_int"] + [c for c in top.columns if c.startswith("dummy_")][:3]
        display_cols = [c for c in display_cols if c in top.columns]
        display_df = top[display_cols]

        # Use AgGrid if available for better interactivity
        if aggrid_available:
            try:
                gb = GridOptionsBuilder.from_dataframe(display_df)
                gb.configure_selection(selection_mode='single', use_checkbox=True)
                grid_options = gb.build()
                grid_response = AgGrid(display_df, gridOptions=grid_options, height=350)
            except Exception:
                st.dataframe(display_df, height=400)
        else:
            st.dataframe(display_df, height=400)

    st.markdown('<div class="section-header"><h3>Inspect a single flow / JSON view</h3></div>', unsafe_allow_html=True)
    idx = st.number_input("Row index (0-based)", 0, max(0, total - 1), 0)
    if total > 0:
        row = results.iloc[int(idx)].to_dict()
        st.json(row)

    st.markdown('<div class="section-header"><h3>Export results</h3></div>', unsafe_allow_html=True)
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
        st.subheader("Raw extracted features (first 300 rows)")
        st.dataframe(df.head(300))

    st.success("Detection completed ✅")

else:
    st.markdown('<div class="section-header"><h3>Welcome</h3></div>', unsafe_allow_html=True)
    st.markdown("<div class='summary-card' style='padding:14px'>This dashboard extracts features from PCAPs and runs a trained ML model to detect suspicious activity. Use the controls on the left to configure labels, thresholds and run detection.</div>", unsafe_allow_html=True)

# close theme wrapper
if ui_theme.startswith("Soft"):
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Keyboard shortcut (R to rerun) ----------
# This injects a small script that listens for 'r' or 'R' keypress and clicks the Extract & Predict button
st.markdown("""
<script>
window.addEventListener('keydown', function(e) {
  if (e.key === 'r' || e.key === 'R') {
    // find button containing the text 'Extract & Predict'
    const buttons = document.querySelectorAll('button');
    for (let i=0;i<buttons.length;i++){
      const b = buttons[i];
      if (b.innerText && b.innerText.includes('Extract & Predict')){
        b.click();
        break;
      }
    }
  }
});
</script>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("Made with ❤️ by the AI-IDS team")
