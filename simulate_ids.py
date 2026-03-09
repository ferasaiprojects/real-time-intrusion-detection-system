#!/usr/bin/env python3
"""
simulate_ids.py

Robust simulator for running IDS predictions on a CSV sample or a real PCAP.
This version is tolerant of different extractor signatures (with/without num_packets).
Saves a CSV to results/simulated_pcap_results.csv by default.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from typing import Optional

# optional: use the same extractor if you want to test using a PCAP
try:
    from pcap_feature_extractor import extract_features_from_pcap
    HAVE_PCAP_EXTRACTOR = True
except Exception:
    extract_features_from_pcap = None
    HAVE_PCAP_EXTRACTOR = False

DEFAULT_MODEL = "models/ids_model.pkl"
DEFAULT_PIPELINE = "models/pipeline_ids.pkl"
TRAINING_CSV = "data/UNSW_NB15_testing-set.csv"  # used for sampling if no pcap provided
OUTPUT_DIR = "results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "simulated_pcap_results.csv")

def load_best_model(pipeline_path=DEFAULT_PIPELINE, model_path=DEFAULT_MODEL):
    """
    Prefer a saved pipeline (preprocessor+model). If not found, load a raw model.
    Returns the loaded object and a boolean flag 'is_pipeline'.
    """
    if os.path.exists(pipeline_path):
        m = joblib.load(pipeline_path)
        return m, True
    if os.path.exists(model_path):
        m = joblib.load(model_path)
        return m, False
    raise FileNotFoundError(f"Neither pipeline ({pipeline_path}) nor model ({model_path}) found.")

def get_expected_feature_names(model):
    """
    Try several ways to find expected feature names used at training time.
    Returns list (may be empty).
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # LightGBM booster
    if hasattr(model, "booster_"):
        try:
            names = list(model.booster_.feature_name())
            return names
        except Exception:
            pass
    # sklearn pipeline: get final estimator inside pipeline
    try:
        if hasattr(model, "named_steps"):
            final = model.named_steps.get(list(model.named_steps.keys())[-1])
            if final is not None and hasattr(final, "feature_names_in_"):
                return list(final.feature_names_in_)
            if final is not None and hasattr(final, "booster_"):
                try:
                    return list(final.booster_.feature_name())
                except Exception:
                    pass
    except Exception:
        pass
    return []

def malicious_probs_from_model(model, X):
    """
    Return malicious-class probabilities robustly.
    If predict_proba exists, pick the column matching class '1' if present,
    otherwise fallback to last column. If not available, map predict -> 1/0.
    """
    if not hasattr(model, "predict_proba"):
        preds = model.predict(X)
        return np.array([1.0 if int(p) == 1 else 0.0 for p in preds], dtype=float)
    probs = model.predict_proba(X)
    # handle 1D vs 2D outputs
    if probs.ndim == 1:
        return np.asarray(probs).astype(float)
    classes = list(getattr(model, "classes_", []))
    try:
        if len(classes) == probs.shape[1] and len(classes) > 0:
            if 1 in classes:
                idx = classes.index(1)
            elif "MALICIOUS" in classes:
                idx = classes.index("MALICIOUS")
            else:
                idx = probs.shape[1] - 1
            return np.asarray(probs)[:, idx].astype(float)
    except Exception:
        pass
    # fallback: last column
    return np.asarray(probs)[:, -1].astype(float)

def prepare_sample_from_csv(csv_path, n=10):
    df = pd.read_csv(csv_path)
    sample = df.sample(n=min(n, len(df)), random_state=42)
    # Drop typical meta columns if present
    for col in ["id", "attack_cat", "label"]:
        if col in sample.columns:
            sample.drop(columns=[col], inplace=True)
    # encode categorical columns similarly to training step
    for col in ["proto", "service", "state"]:
        if col in sample.columns:
            sample[col] = sample[col].astype("category").cat.codes
    return sample.reset_index(drop=True)

def align_to_expected(X: pd.DataFrame, expected_cols):
    """Return DataFrame with exactly expected_cols (in order), filling missing columns with 0."""
    if expected_cols is None or len(expected_cols) == 0:
        return X.copy()
    aligned = pd.DataFrame(0, index=range(len(X)), columns=expected_cols)
    for c in X.columns:
        if c in aligned.columns:
            aligned[c] = X[c].values
    return aligned

def _normalize_extractor_output(x):
    """
    Accept DataFrame, list-of-dicts, or numpy array and return a DataFrame.
    """
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.DataFrame):
        return x.copy().reset_index(drop=True)
    if isinstance(x, (list, tuple)):
        try:
            return pd.DataFrame(x).reset_index(drop=True)
        except Exception:
            pass
    # try to coerce
    try:
        return pd.DataFrame(x).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def main(args):
    # 1) load model/pipeline
    try:
        model, is_pipeline = load_best_model()
        print(f"Loaded {'pipeline' if is_pipeline else 'model'} successfully.")
    except Exception as e:
        print("ERROR loading model/pipeline:", e)
        return

    print("Model type:", type(model))
    print("Has predict_proba:", hasattr(model, "predict_proba"))
    print("Model.classes_ (if present):", getattr(model, "classes_", None))

    # 2) prepare sample (CSV sampling or PCAP extraction)
    X = pd.DataFrame()
    if args.pcap:
        if not HAVE_PCAP_EXTRACTOR or extract_features_from_pcap is None:
            print("PCAP extractor not available (cannot import pcap_feature_extractor). Abort.")
            return
        print("Extracting features from PCAP:", args.pcap)
        # attempt to call extractor signature with num_packets first, else fallback
        try:
            # some extractor versions support num_packets / max_flows
            raw = extract_features_from_pcap(args.pcap, num_packets=args.n)
        except TypeError:
            # fallback: single-argument extractor and then .head()
            try:
                raw = extract_features_from_pcap(args.pcap)
            except Exception as e:
                print("PCAP extractor raised an error during fallback call:", e)
                return
            # truncate to n if possible after normalization
            df_raw = _normalize_extractor_output(raw)
            if df_raw.shape[0] == 0:
                print("Extractor returned no flows.")
                return
            if args.n and df_raw.shape[0] > args.n:
                df_raw = df_raw.head(args.n).reset_index(drop=True)
            X = df_raw
        except Exception as e:
            print("PCAP feature extraction FAILED:", e)
            return
        else:
            # we succeeded calling with num_packets; normalize
            X = _normalize_extractor_output(raw)
            if X.shape[0] == 0:
                print("Extractor returned no flows.")
                return
            if args.n and X.shape[0] > args.n:
                X = X.head(args.n).reset_index(drop=True)
    else:
        print("Sampling rows from CSV:", TRAINING_CSV)
        X = prepare_sample_from_csv(TRAINING_CSV, n=args.n)

    print("Sample shape:", X.shape)
    print("Sample dtypes:\n", X.dtypes.head(20))

    if X.shape[0] == 0:
        print("No data to predict on. Exiting.")
        return

    # 3) align features if model exposes expected names
    expected = get_expected_feature_names(model)
    if expected:
        print(f"Model exposes {len(expected)} expected feature names. Aligning sample...")
    else:
        print("Model did not expose expected feature names. Will attempt to use sample columns as-is.")
    X_aligned = align_to_expected(X, expected)

    # short diagnostics
    common = [c for c in X.columns if c in expected] if expected else X.columns.tolist()
    print("Common columns count (sample vs expected):", len(common))
    if expected:
        missing = [c for c in expected if c not in X.columns]
        print(f"Missing expected columns (count {len(missing)}), first 12: {missing[:12]}")
    print("Aligned shape:", X_aligned.shape)

    # 4) attempt prediction
    try:
        probs = malicious_probs_from_model(model, X_aligned)
    except Exception as e:
        print("Prediction failed:", e)
        return

    # 5) results summary
    threshold = args.threshold
    preds = (probs >= threshold).astype(int)
    n_attack = int(preds.sum())
    n_total = len(probs)
    print(f"\nResults (threshold={threshold:.2f}): {n_attack}/{n_total} flagged as ATTACK ({n_attack/n_total:.2%})\n")

    # show per-row results (compact)
    out = []
    for i, p in enumerate(probs):
        out.append({"row": i, "malicious_probability": float(p), "label": "ATTACK" if p >= threshold else "BENIGN"})
    df_out = pd.DataFrame(out)
    pd.set_option("display.max_rows", None)
    print(df_out)

    # ensure results dir exists and save CSV
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved results to: {OUTPUT_CSV}")
    except Exception as e:
        print("Failed to save results CSV:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate real-time IDS predictions")
    parser.add_argument("--n", type=int, default=10, help="Number of sample rows (or num_packets for pcap)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold to flag ATTACK")
    parser.add_argument("--pcap", type=str, default=None, help="Optional: path to a PCAP to extract features from (requires extractor)")
    args = parser.parse_args()
    main(args)