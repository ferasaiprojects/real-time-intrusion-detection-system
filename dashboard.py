import os
import glob
import tempfile
import joblib
from typing import Optional, List, Dict, Any
import time

import pandas as pd
import numpy as np
import streamlit as st

# try imports from project
try:
    # expects src/predict.py with predict_traffic(...)
    from src.predict import predict_traffic, align_pcap_df_to_model  # align may or may not exist
    HAVE_ALIGN = True
except Exception:
    # best-effort: still try to import predict_traffic only
    try:
        from src.predict import predict_traffic
    except Exception:
        predict_traffic = None
    align_pcap_df_to_model = None
    HAVE_ALIGN = False

# local extractor module (must exist)
try:
    from pcap_feature_extractor import extract_features_from_pcap
except Exception:
    extract_features_from_pcap = None

# ---------- Constants ----------
MODEL_PATH = os.path.join("models", "ids_model.pkl")
PIPELINE_PATH = os.path.join("models", "pipeline_ids.pkl")  # optional
SAMPLE_DIR_CANDIDATES = ["sample_pcap", "sample_pcaps", "sample_pcap/", "sample_pcaps/", "sample_pcap_files"]
EXPECTED_FEATURES = 42
MAX_TOP_ROWS = 500

FEATURE_COLUMNS = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
    'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss',
    'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
    'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
    'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
]

# ---------- Utilities ----------
def clear_model_cache():
    for key in ["loaded_model", "model_cached", "model"]:
        if key in st.session_state:
            try:
                del st.session_state[key]
            except Exception:
                pass
    try:
        if hasattr(st, "cache_resource") and hasattr(st.cache_resource, "clear"):
            st.cache_resource.clear()
    except Exception:
        pass
    try:
        if hasattr(st, "experimental_memo") and hasattr(st.experimental_memo, "clear"):
            st.experimental_memo.clear()
    except Exception:
        pass
    try:
        if hasattr(st, "cache_data") and hasattr(st.cache_data, "clear"):
            st.cache_data.clear()
    except Exception:
        pass

def find_sample_pcaps(dirs: List[str] = SAMPLE_DIR_CANDIDATES, pattern: str = "*.pcap") -> List[str]:
    found_set = set()
    found = []
    def _add_path(p: str):
        ab = os.path.realpath(os.path.abspath(p))
        if ab not in found_set and os.path.isfile(ab):
            found_set.add(ab)
            found.append(ab)
    for d in dirs:
        try:
            if not d:
                continue
            if os.path.isdir(d):
                for e in glob.glob(os.path.join(d, pattern)):
                    _add_path(e)
        except Exception:
            continue
    for d in dirs:
        try:
            if os.path.isdir(d):
                for m in glob.glob(os.path.join(d, "**", pattern), recursive=True):
                    _add_path(m)
        except Exception:
            continue
    try:
        for t in glob.glob(pattern):
            _add_path(t)
    except Exception:
        pass
    found = sorted(found)
    return found

@st.cache_data(ttl=300)
def get_sample_summary(path: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    try:
        summary["path"] = os.path.realpath(os.path.abspath(path))
        summary["name"] = os.path.basename(path)
        summary["size_bytes"] = os.path.getsize(path)
        summary["modified_time"] = time.ctime(os.path.getmtime(path))
        try:
            if extract_features_from_pcap:
                df = extract_features_from_pcap(path)
            else:
                df = pd.DataFrame()
            if isinstance(df, pd.DataFrame) and df.shape[0] > 0:
                summary["num_flows"] = int(df.shape[0])
                first_row = df.reset_index(drop=True).iloc[0]
                preview = {}
                for i, col in enumerate(first_row.index):
                    if i >= 6:
                        break
                    val = first_row[col]
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

def fallback_align_pcap_df_to_model(df: pd.DataFrame, model=None, train_csv: Optional[str]=None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    aligned = pd.DataFrame(0, index=range(len(df)), columns=FEATURE_COLUMNS)
    for c in df.columns:
        if c in aligned.columns:
            aligned[c] = df[c].values
    return aligned

def map_severity_by_prob(prob: float) -> str:
    if prob >= 0.9:
        return "HIGH"
    if prob >= 0.7:
        return "MEDIUM"
    if prob >= 0.4:
        return "LOW"
    return "NORMAL"

# ---------- Page config ----------
st.set_page_config(page_title="AI Intrusion Detection System", page_icon="🛡️", layout="wide")

# ---------- CSS (softer shadows + larger title) ----------
_COMMON_CSS = r"""
<style>
:root{
  --accent1: #0b76ff;
  --accent2: #00b894;
  --muted-dark: #9fb4d8;
  --muted-light: #475569;
  --card-bg: rgba(255,255,255,0.02);
  --card-border: rgba(255,255,255,0.04);
  --shadow-soft: 0 4px 10px rgba(2,6,23,0.12);
  --shadow-subtle: 0 2px 6px rgba(2,6,23,0.08);
}

/* Page title card */
.page-title {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  padding:18px 18px;
  margin-bottom:14px;
  border-radius:12px;
  border-left:6px solid var(--accent1);
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  box-shadow: var(--shadow-subtle);
  word-break: break-word;
  overflow-wrap: anywhere;
}
.page-title .headline { font-size: 1.65rem; font-weight:900; line-height:1.05; }
.page-title .sub { opacity:0.85; color:var(--muted-dark); font-size:0.98rem; }

/* Generic section card */
.section-card {
  border-radius:10px;
  padding:12px;
  margin-bottom:12px;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  box-shadow: var(--shadow-soft);
}

/* Section header within a card */
.section-header-card {
  display:flex; align-items:center; gap:12px;
  padding:8px 10px; border-radius:8px; border-left:6px solid var(--accent1);
  background: rgba(255,255,255,0.01); margin-bottom:8px;
}
.section-header-card h3 { margin:0; font-size:1.05rem; }

/* Grid */
.card-grid { display: grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 14px; margin-bottom: 14px; align-items: stretch; }

/* KPI card */
.summary-card {
  border-radius: 10px;
  padding: 12px;
  min-height: 84px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 6px;
  background: var(--card-bg);
  border: 1px solid rgba(255,255,255,0.03);
  box-shadow: var(--shadow-subtle);
}
.summary-card .title { font-size: 0.90rem; font-weight: 700; color: var(--muted-light); }
.summary-card .value { font-size: 1.6rem; font-weight: 800; color: var(--accent1); }
.summary-card.kpi { border-left: 6px solid var(--accent2); padding-left: 12px; }

.sample-card { border-radius:8px; padding:8px; background: rgba(255,255,255,0.01); border:1px solid rgba(255,255,255,0.02); margin-bottom:8px; box-shadow: var(--shadow-subtle); }
.sample-card .meta { font-size:0.82rem; color:var(--muted-light); }

@media (max-width:880px){
  .card-grid { grid-template-columns: 1fr; }
  .summary-card { min-height: 76px; padding: 10px; }
  .summary-card .value { font-size: 1.3rem; }
  .page-title { padding:12px; }
  .page-title .headline { font-size: 1.35rem; }
}
</style>
"""

_DARK_CSS = """
<style>
.stApp { background: linear-gradient(180deg,#041022,#07182a) !important; color: #e6f0ff !important; }
.section-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); color: #e6f0ff; }
.sample-card { background: rgba(255,255,255,0.02); }
.summary-card { color: #e6f0ff; }
</style>
"""

_LIGHT_CSS = """
<style>
.stApp { background: linear-gradient(180deg,#f8fafc,#eef2ff) !important; color: #071023 !important; }
.section-card { background: linear-gradient(180deg,#ffffff,#f7f9fb); color: #071023; }
.sample-card { background: rgba(255,255,255,0.98); }
.summary-card { color: #071023; }
</style>
"""

# ---------- Sidebar controls ----------
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
    top_n = int(st.number_input("Top N suspicious flows", min_value=5, max_value=MAX_TOP_ROWS, value=50))
    show_raw = st.checkbox("Show raw features (large)", value=False)
    highlight_suspicious = st.checkbox("Highlight suspicious rows", value=True)
    run_on_upload = st.checkbox("Auto-run detection on upload", value=True)

    st.markdown("---")
    if st.button("Clear cached model"):
        try:
            clear_model_cache()
            st.sidebar.success("Cache cleared (attempted).")
        except Exception as e:
            st.sidebar.error(f"Failed to clear cache: {e}")
        try:
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
        except Exception:
            try:
                if hasattr(st, "rerun"):
                    st.rerun()
            except Exception:
                st.sidebar.info("Could not programmatically rerun the app. Reload the browser page as fallback.")


# ---------- Inject CSS & theme wrapper ----------
st.markdown(_COMMON_CSS, unsafe_allow_html=True)
if ui_theme.startswith("Soft"):
    st.markdown(_DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(_LIGHT_CSS, unsafe_allow_html=True)

if background_style.startswith("Professional"):
    st.markdown("""
    <style>
    body > div[role="application"] { background-image: radial-gradient(circle at 10% 10%, rgba(11,118,255,0.03), transparent 10%), linear-gradient(180deg, rgba(3,10,20,0.6), rgba(3,10,20,0.85)); background-attachment: fixed; }
    </style>
    """, unsafe_allow_html=True)

# ---------- helpers ----------
def section_card(title: str, body_callable):
    st.markdown(f"<div class='section-card'><div class='section-header-card'><h3>{title}</h3></div>", unsafe_allow_html=True)
    try:
        body_callable()
    except Exception as e:
        st.error(f"Rendering section '{title}' failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

def small_kpi(title: str, value: str, sub: str, kclass=""):
    return f"""
    <div class='summary-card kpi {kclass}'>
      <div class='title'>{title}</div>
      <div class='value'>{value}</div>
      <div class='sub'>{sub}</div>
    </div>
    """

# ---------- Load model ----------
@st.cache_resource
def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

try:
    model = load_model()
except FileNotFoundError:
    st.sidebar.error(f"Model missing at {MODEL_PATH}. Upload or place the model in 'models'.")
    model = None
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model = None

# optional AgGrid
try:
    from st_aggrid import AgGrid, GridOptionsBuilder
    aggrid_available = True
except Exception:
    aggrid_available = False

# ---------- Page title ----------
st.markdown(
    "<div class='page-title'><div><div class='headline'>🛡️ AI-Powered Intrusion Detection System</div>"
    "<div class='sub'>Streamlit · PCAP → features → model</div></div></div>",
    unsafe_allow_html=True
)
st.markdown("Upload a PCAP, choose label scheme and run detection. Use controls on the left.")

# ---------- Layout ----------
left, right = st.columns([2, 1])

with left:
    def render_pcap_source():
        sample_list = find_sample_pcaps()
        source_mode = st.radio("Choose PCAP source:", ["Use default sample (recommended)", "Upload custom PCAP"], index=0)
        pcap_path = None
        if source_mode.startswith("Use default"):
            if sample_list:
                filenames = [os.path.basename(p) for p in sample_list]
                selected_idx = st.session_state.get("selected_default_idx", 0)
                if selected_idx >= len(filenames):
                    selected_idx = 0
                sel = st.selectbox("Select default sample:", options=filenames, index=selected_idx)
                sel_idx = filenames.index(sel)
                st.session_state["selected_default_idx"] = sel_idx
                pcap_path = sample_list[sel_idx]
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
                    st.session_state["_tmp_pcap"] = tmp.name
                    pcap_path = tmp.name
        else:
            uploaded = st.file_uploader("Upload PCAP (.pcap)", type=["pcap"])
            if uploaded is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcap")
                tmp.write(uploaded.read())
                tmp.flush()
                tmp.close()
                st.session_state["_tmp_pcap"] = tmp.name
                pcap_path = tmp.name
            else:
                st.info("No custom PCAP uploaded yet. Choose 'Use default sample' to run immediately using a sample file.")

        st.markdown("**PCAP path:**")
        st.write(pcap_path if pcap_path else "No PCAP selected")
        st.markdown("Press **R** to re-run inference (keyboard shortcut)")
        st.session_state["_pcap_path"] = pcap_path

    section_card("1) PCAP Source", render_pcap_source)

with right:
    def render_samples_and_model():
        sample_list = find_sample_pcaps()
        st.markdown("<div style='display:flex;flex-direction:column;gap:8px'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700;margin-bottom:6px'>Samples</div>", unsafe_allow_html=True)
        if sample_list:
            for sp in sample_list:
                s = get_sample_summary(sp)
                highlight = (st.session_state.get("_pcap_path") == s.get("path"))
                card_html = f"""
                <div class='sample-card' style='border: 1px solid {"#0b76ff" if highlight else "rgba(255,255,255,0.02)"};'>
                  <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='font-weight:700'>{s.get('name')}</div>
                    <div class='meta'>{s.get('num_flows', '?')} flows</div>
                  </div>
                  <div style='font-size:0.85rem; margin-top:6px;'>Size: {s.get('size_bytes', '?')} bytes<br>Modified: {s.get('modified_time', '-')}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
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

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700;margin-bottom:6px'>Model & Quick Info</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-card'><div style='font-weight:700'>Model Path</div><div style='padding-top:6px'>{os.path.basename(MODEL_PATH)}</div><div style='color:var(--muted-dark);padding-top:4px'>Threshold: {threshold:.2f}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    section_card("Samples & Model", render_samples_and_model)

# ---------- Run logic ----------
pcap_path = st.session_state.get("_pcap_path", None)
do_run = (run_on_upload and pcap_path is not None) or st.button("Extract & Predict ▶️")

if not pcap_path:
    st.info("No PCAP selected — use the default sample or upload a custom file to run detection.")

if do_run and pcap_path and model is not None:
    with st.spinner("Extracting features from PCAP... ⛏️"):
        try:
            num_flows = int(st.sidebar.number_input("Max flows to extract from PCAP", min_value=10, max_value=500, value=50, step=10))
            if extract_features_from_pcap is None:
                st.error("PCAP feature extractor not available (pcap_feature_extractor.py not importable).")
                st.stop()
            raw_df = extract_features_from_pcap(pcap_path)
            if raw_df is None or getattr(raw_df, "shape", (0, 0))[0] == 0:
                st.warning("No flows parsed from PCAP. Check the file content.")
                st.stop()
            if raw_df.shape[0] > num_flows:
                raw_df = raw_df.head(num_flows).reset_index(drop=True)
        except Exception as e:
            st.exception(f"Feature extraction failed: {e}")
            st.stop()

    try:
        if HAVE_ALIGN and align_pcap_df_to_model:
            aligned_df = align_pcap_df_to_model(raw_df, model, train_csv="data/UNSW_NB15_training-set.csv")
        else:
            aligned_df = fallback_align_pcap_df_to_model(raw_df, model, train_csv="data/UNSW_NB15_training-set.csv")
    except Exception as e:
        st.warning(f"Failed to align PCAP features with model automatically: {e}. Applying fallback alignment.")
        aligned_df = fallback_align_pcap_df_to_model(raw_df, model, train_csv="data/UNSW_NB15_training-set.csv")

    df = pad_features(aligned_df, EXPECTED_FEATURES)

    with st.spinner("Running model inference... 🤖"):
        try:
            if predict_traffic is None:
                raise RuntimeError("predict_traffic not available from src.predict")
            results_list = predict_traffic(df, threshold=threshold) if callable(predict_traffic) else []
            probs = []
            preds_int = []
            preds_label = []
            for r in results_list:
                p = None
                if isinstance(r, dict):
                    p = r.get("attack_probability") or r.get("malicious_probability") or r.get("malicious_prob") or r.get("mal_prob")
                try:
                    p = float(p) if p is not None else 0.0
                except Exception:
                    p = 0.0
                probs.append(p)
                pred_label_val = None
                if isinstance(r, dict):
                    pred_label_val = r.get("prediction") or r.get("label")
                pred_label_val = str(pred_label_val).upper() if pred_label_val is not None else ""
                if pred_label_val in ("ATTACK", "MALICIOUS", "1", "TRUE"):
                    preds_int.append(1)
                    preds_label.append("ATTACK")
                else:
                    preds_int.append(0)
                    preds_label.append("NORMAL")
        except Exception as e:
            st.error(f"Inference failed: {e}")
            try:
                mdl = load_model()
                X_proc = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
                if hasattr(mdl, "predict_proba"):
                    probs_arr = mdl.predict_proba(X_proc)
                    if probs_arr.ndim == 1:
                        probs = [float(x) for x in probs_arr]
                    else:
                        try:
                            probs = [float(x) for x in probs_arr[:, 1]]
                        except Exception:
                            probs = [float(x) for x in probs_arr[:, -1]]
                else:
                    preds_raw = mdl.predict(X_proc)
                    probs = [1.0 if int(x) == 1 else 0.0 for x in preds_raw]
                preds_int = [1 if p >= threshold else 0 for p in probs]
                preds_label = ["ATTACK" if p >= threshold else "NORMAL" for p in probs]
            except Exception as e2:
                st.error(f"Fallback prediction failed: {e2}")
                probs = [0.0] * len(df)
                preds_int = [0] * len(df)
                preds_label = ["NORMAL"] * len(df)

    results = df.copy().reset_index(drop=True)
    # normalized/compat columns
    results["malicious_probability"] = [float(p) for p in probs]
    results["attack_probability"] = results["malicious_probability"]
    results["prediction_int"] = [int(x) for x in preds_int]
    results["prediction"] = preds_label
    results["is_suspicious"] = results["malicious_probability"] >= threshold

    # label column for legacy usage
    if label_preset == "Binary (BENIGN / MALICIOUS)":
        lbl0, lbl1 = "BENIGN", "MALICIOUS"
        results["label"] = results["prediction_int"].map({0: lbl0, 1: lbl1})
    elif label_preset == "Severity (NORMAL/LOW/MEDIUM/HIGH)":
        results["label"] = results["malicious_probability"].apply(map_severity_by_prob)
    else:
        lbl0 = custom_label_0 if custom_label_0 else "BENIGN"
        lbl1 = custom_label_1 if custom_label_1 else "MALICIOUS"
        results["label"] = results["prediction_int"].map({0: lbl0, 1: lbl1})

    # ---- NEW: canonical severity column used for chart color mapping ----
    # Always derive severity from probability to ensure consistent coloring:
    results["severity"] = results["malicious_probability"].apply(map_severity_by_prob)

    total = len(results)
    attacks = int(results["prediction_int"].sum())
    normals = total - attacks
    attack_ratio = attacks / total if total else 0.0

    # Threat level text + color (reuse your compact script logic)
    if attack_ratio < 0.1:
        threat_level = "LOW"
        threat_color = "#4caf50"  # green
    elif attack_ratio < 0.3:
        threat_level = "MEDIUM"
        threat_color = "#ff9800"  # orange
    else:
        threat_level = "HIGH"
        threat_color = "#ff4c4c"  # red

    def render_threat_card():
        html = f"""
        <div style='padding:14px; border-radius:10px; background-color: rgba(0,0,0,0.05);'>
          <h2 style='color:{threat_color}; margin:0; font-weight:900;'>● {threat_level} THREAT LEVEL</h2>
          <p style='color:var(--muted-dark); margin:6px 0 0 0;'>{attack_ratio*100:.1f}% of flows flagged as suspicious</p>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    section_card("Threat Level", render_threat_card)

    # ---- KPI cards ----
    def render_summary():
        st.markdown("<div class='card-grid'>", unsafe_allow_html=True)
        kpis = [
            ("Total Flows", total, "Total parsed flows"),
            ("Benign / Normal", normals, "Non-alert flows"),
            ("Suspicious / Alerts", attacks, "Detected suspicious flows"),
            ("Threat Level", f"{threat_level}", f"{attack_ratio*100:.2f}% of flows"),
        ]
        for title, value, sub in kpis:
            st.markdown(small_kpi(title, value, sub), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    section_card("Summary", render_summary)

    # -------------------------
    # Charts & Table (updated for colored charts based on severity)
    # -------------------------
    def render_charts_and_table():
        st.markdown("### Traffic Overview")

        # canonical color map for severity
        severity_color_map = {
            "HIGH": "#ff4c4c",    # red
            "MEDIUM": "#ff9800",  # orange
            "LOW": "#4caf50",     # green
            "NORMAL": "#6ec6ff"   # soft blue for normal/ok
        }
        # Attempt Altair charts with colors; fallback to simple charts if Altair not available
        try:
            import altair as alt

            # 1) Traffic Overview: group by severity (consistent coloring)
            counts = results.groupby("severity").size().reset_index(name="count")
            # ensure deterministic ordering
            severity_order = ["HIGH", "MEDIUM", "LOW", "NORMAL"]
            counts["severity"] = pd.Categorical(counts["severity"], categories=severity_order, ordered=True)
            counts = counts.sort_values("severity")

            bar = (
                alt.Chart(counts)
                .mark_bar()
                .encode(
                    x=alt.X("severity:N", title="Severity", sort=severity_order),
                    y=alt.Y("count:Q", title="Count"),
                    color=alt.Color("severity:N", scale=alt.Scale(domain=list(severity_color_map.keys()), range=list(severity_color_map.values())), legend=None),
                    tooltip=["severity", "count"]
                )
                .properties(height=220)
            )
            st.altair_chart(bar, use_container_width=True)

            st.markdown("### Probability Distribution")
            # 2) Probability Distribution: stacked histogram by severity (bins colored by severity)
            # create a severity column already present; use it to color/stack bins
            hist = (
                alt.Chart(results)
                .mark_bar()
                .encode(
                    x=alt.X("malicious_probability:Q", bin=alt.Bin(maxbins=40), title="Malicious probability"),
                    y=alt.Y("count():Q", title="Flows"),
                    color=alt.Color("severity:N", title="Severity", scale=alt.Scale(domain=list(severity_color_map.keys()), range=list(severity_color_map.values()))),
                    tooltip=[alt.Tooltip("count()", title="Flows")]
                )
                .properties(height=220)
            )
            st.altair_chart(hist, use_container_width=True)

            st.markdown("### Attack Probability Over Sample")
            # 3) Attack Probability Over Sample: use row order (sample index) and color by severity
            # create a row index column for plotting
            plot_df = results.reset_index().rename(columns={"index": "row_index"})
            # ensure severity is categorical with our ordering for consistent legend
            plot_df["severity"] = pd.Categorical(plot_df["severity"], categories=severity_order, ordered=True)

            line = (
                alt.Chart(plot_df.head(200))
                .mark_line(point=True)
                .encode(
                    x=alt.X("row_index:Q", title="Sample index"),
                    y=alt.Y("malicious_probability:Q", title="Malicious probability"),
                    color=alt.Color("severity:N", scale=alt.Scale(domain=list(severity_color_map.keys()), range=list(severity_color_map.values())), legend=alt.Legend(title="Severity")),
                    tooltip=["row_index", alt.Tooltip("malicious_probability", format=".3f"), "severity"]
                )
                .properties(height=220)
            )
            st.altair_chart(line, use_container_width=True)

        except Exception:
            # Altair unavailable or failed — fallback to simple Streamlit charts (colors not customizable here)
            # Traffic Overview fallback (severity counts)
            st.bar_chart(results["severity"].value_counts().loc[["HIGH", "MEDIUM", "LOW", "NORMAL"]].fillna(0))

            st.markdown("### Probability Distribution")
            st.bar_chart(results["malicious_probability"].value_counts().sort_index())

            st.markdown("### Attack Probability Over Sample")
            st.line_chart(results["malicious_probability"].head(200))

        # Detected Flows — Interactive View (left unchanged as requested)
        st.markdown("### Detected Flows — Interactive View")
        filtered = results[results["malicious_probability"] >= prob_filter].sort_values("malicious_probability", ascending=False)
        if filtered.shape[0] == 0:
            st.info("No flows match the current probability filter. Lower the filter or choose a different PCAP.")
        else:
            top = filtered.head(top_n).copy()
            display_cols = ["prediction", "attack_probability", "malicious_probability", "prediction_int"] + [c for c in top.columns if c.startswith("dummy_")][:3]
            display_cols = [c for c in display_cols if c in top.columns]
            display_df = top[display_cols]

            if aggrid_available:
                try:
                    gb = GridOptionsBuilder.from_dataframe(display_df)
                    gb.configure_selection(selection_mode='single', use_checkbox=True)
                    grid_options = gb.build()
                    AgGrid(display_df, gridOptions=grid_options, height=350)
                except Exception:
                    st.dataframe(display_df, height=400)
            else:
                # styled dataframe: highlight attack rows (UNCHANGED)
                def _highlight(row):
                    return ["background-color: #ff4c4c; color: white; font-weight:700" if row["prediction"] == "ATTACK" else "background-color: #28a745; color: white; font-weight:700" for _ in row]
                try:
                    styled = display_df.style.apply(lambda r: _highlight(r), axis=1)
                    st.dataframe(styled, height=400)
                except Exception:
                    st.dataframe(display_df, height=400)

    section_card("Traffic & Tables", render_charts_and_table)

    # Inspect / Export / Top suspicious flows
    def render_inspect_and_export():
        st.markdown("### Inspect a single flow / JSON view")
        idx = st.number_input("Row index (0-based)", 0, max(0, total - 1), 0)
        if total > 0:
            row = results.iloc[int(idx)].to_dict()
            st.json(row)

        st.markdown("### Top Suspicious Flows")
        if attacks > 0:
            top_attacks = results[results["prediction"]=="ATTACK"].sort_values("attack_probability", ascending=False).head(10)
            st.dataframe(top_attacks, use_container_width=True)
        else:
            st.info("No ATTACK flows detected in this sample.")

        st.markdown("### Export results")
        results_df = results.copy()
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        json_str = results_df.to_json(orient="records", indent=2)
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

    section_card("Inspect / Export", render_inspect_and_export)

    st.success("Detection completed ✅")

else:
    def render_welcome():
        st.markdown("This dashboard extracts features from PCAPs and runs a trained ML model to detect suspicious activity. Use the controls on the left to configure labels, thresholds and run detection.")
    section_card("Welcome", render_welcome)

# ---------- Keyboard shortcut ----------
st.markdown("""
<script>
window.addEventListener('keydown', function(e) {
  if (e.key === 'r' || e.key === 'R') {
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
st.markdown("Made with ❤️ by the Feras team")