# src/parser.py
"""
Log and file parsing utilities
"""

import pandas as pd
import os


def load_csv(file_path):
    """
    Load any CSV file and return a DataFrame
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df


def load_pcap_results(results_path="ids_detection_results.csv"):
    """
    Load saved detection results CSV from dashboard export
    """
    return load_csv(results_path)


def load_unsw_sample(data_dir="data"):
    """
    Load a small sample from UNSW-NB15 test set for quick simulation
    """
    path = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")

    if not os.path.exists(path):
        print(f"UNSW-NB15 test set not found at: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Drop non-feature columns
    for col in ["id", "attack_cat", "label"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Encode categoricals
    for col in ["proto", "service", "state"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    print(f"Loaded UNSW-NB15 sample: {len(df)} rows")
    return df