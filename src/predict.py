"""
Real-time intrusion prediction module
"""

import joblib
import pandas as pd

# UNSW-NB15 exact feature order your model expects
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

# Load once at module level — not on every call
_model = None

def load_model(model_path="models/ids_model.pkl"):
    global _model
    if _model is None:
        _model = joblib.load(model_path)
        print("Model loaded from:", model_path)
    return _model


def predict_traffic(sample_data, threshold=0.6):
    model = load_model()

    # Align columns to exact training feature order
    # Add missing columns as 0, drop unexpected columns
    sample_data = sample_data.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # Validate
    if sample_data.empty:
        print("Warning: Empty input data.")
        return []

    predictions = model.predict(sample_data)
    probabilities = model.predict_proba(sample_data)[:, 1]

    results = []

    for i in range(len(predictions)):
        prob = float(probabilities[i])

        # Use threshold instead of raw prediction for better control
        label = "ATTACK" if prob >= threshold else "NORMAL"

        results.append({
            "flow_id":            i + 1,
            "prediction":         label,
            "attack_probability": round(prob, 4),
            "confidence":         "HIGH" if prob > 0.8 or prob < 0.2 else "MEDIUM"
        })

    return results