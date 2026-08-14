from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
import requests
import streamlit as st
import os
from pandas import json_normalize
from datetime import datetime 
localtime = f"{datetime.now().day}-{datetime.now().month}-{datetime.now().year}"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""
Population Stability Index (PSI) for data drift detection.
PSI < 0.1: No major change.
0.1 <= PSI < 0.2: Moderate change, use judgement.
PSI >= 0.2: Significant change, retraining may be required.
"""


@dataclass
class PSIResult:
    feature: str
    psi: float
    status: str  # "stable" | "moderate" | "significant"


def _bucket_edges(reference: np.ndarray, bins: int, bucket_type: str) -> np.ndarray:
    if bucket_type == "quantile":
        quantiles = np.linspace(0, 100, bins + 1)
        edges = np.percentile(reference, quantiles)
    elif bucket_type == "uniform":
        edges = np.linspace(reference.min(), reference.max(), bins + 1)
    else:
        raise ValueError("bucket_type must be 'quantile' or 'uniform'")

    edges = np.unique(edges)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def calculate_psi(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    bins: int = 10,
    bucket_type: str = "quantile",
    epsilon: float = 1e-4,
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    edges = _bucket_edges(reference, bins, bucket_type)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def calculate_psi_categorical(
    reference: pd.Series,
    current: pd.Series,
    epsilon: float = 1e-4,
) -> float:
    ref_freq = reference.value_counts(normalize=True)
    cur_freq = current.value_counts(normalize=True)

    categories = set(ref_freq.index) | set(cur_freq.index)

    psi = 0.0
    for cat in categories:
        ref_pct = max(ref_freq.get(cat, epsilon), epsilon)
        cur_pct = max(cur_freq.get(cat, epsilon), epsilon)
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

    return float(psi)


def classify_psi(psi_value: float) -> str:
    if psi_value < 0.1:
        return "stable"
    elif psi_value < 0.2:
        return "moderate"
    else:
        return "significant"


def psi_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    categorical_features: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    categorical_features = set(categorical_features or [])
    results = []

    for feature in features:
        if feature in categorical_features:
            psi_value = calculate_psi_categorical(reference_df[feature], current_df[feature])
        else:
            psi_value = calculate_psi(reference_df[feature], current_df[feature], bins=bins)

        results.append(PSIResult(feature=feature, psi=psi_value, status=classify_psi(psi_value)))

    report_df = pd.DataFrame([r.__dict__ for r in results])
    return report_df.sort_values("psi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Script logic (fetch live data + run the real drift check). This block is
# ONLY guarded to run when the file is executed directly — e.g.
# `python scripts/check_data_drift.py` in a scheduled Actions job — never
# when the module is merely imported (e.g. by pytest for its functions).
# ---------------------------------------------------------------------------
def fetch_eod_data(ticker: str = "VFS") -> pd.DataFrame:
    eod_key = st.secrets["fin_historical_data"]
    resp = requests.get(f"https://eodhd.com/api/eod/{ticker}?api_token={eod_key}&fmt=json")
    resp.raise_for_status()  # fail loudly on 4xx/5xx instead of silently parsing an error payload
    data = resp.json()

    eod_data = json_normalize(data)

    expected_cols = {"adjusted_close", "volume", "high", "low", "open", "close"}
    missing = expected_cols - set(eod_data.columns)
    if missing:
        raise ValueError(
            f"EOD response is missing expected columns {missing}. "
            f"Got columns: {list(eod_data.columns)}. "
            f"Raw response (truncated): {str(data)[:300]}"
        )

    return eod_data


def main() -> None:
    print("Requesting from the EOD API...")
    eod_data = fetch_eod_data("VFS")

    print("Loading the original saved dataset...")
    df = pd.read_csv(f"{BASE_DIR}/data/vinfast_data_cleaned.csv")
    print ("================ DEBUG COLUMNS") 
    print (df.columns)
    print ("================= END DEBUG")
    print("Investigating Data Drift...")
    report = psi_report(
        df,
        eod_data,
        features=["adjusted_close", "volume", "high", "low", "open", "close"],
    )
    print(report)

    significant = report[report["status"] == "significant"]
    if not significant.empty:
        print("Significant drift detected in:", significant["feature"].tolist())
        # raise SystemExit(1)  # uncomment to fail a CI/CD job on drift
    # How Do i send an email to myself? 

        with open(f"{BASE_DIR}/logs/vinfast_data_drift_logs.txt", 'a') as f:
            f.write("----------------------------------\n")

            f.write(f"Date: {localtime}")
            f.write(f"Data Drift: DETECTED\n")
            f.write(f"{significant['feature'].tolist()}\n")
            f.write(f"WARNING: Update the model")
            f.write("\n\n\n")
            f.close() 
        
    
    else: 
        with open(f"{BASE_DIR}/logs/vinfast_data_drift_logs.txt", 'a') as f:
            f.write("----------------------------------\n")

            f.write(f"Current Date: {localtime}\n")
            f.write(f"Data Drift: NOT DETECTED\n")
            f.write("\n\n\n")
            f.close()         


if __name__ == "__main__":
    main()