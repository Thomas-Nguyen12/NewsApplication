from __future__ import annotations
from dataclasses import dataclass
import pandas as pd 
import numpy as np 
import scipy.stats as stats 
import sys 
import os 
import requests 
import streamlit as st 
import smtplib
from getpass import getpass
import ssl 
from pandas import json_normalize


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



# checking for data drift using the PSI test 
# This is hte population stsability index 

# loading the api keys 
print ("Loading API keys...")
eod_key = st.secrets["fin_historical_data"]


# loading the eod data 
print ("Requesting from the EOD API...")
eod_request = requests.get(f"https://eodhd.com/api/eod/VFS?api_token={eod_key}&fmt=json")
eod_data = json_normalize(eod_request.json())

print (f"---------------------- EOD DATA") 
print (eod_data)

# loading the original dataset
print ("Loading the original saved dataset...")
df = pd.read_csv(f"{BASE_DIR}/data/vinfast_data_cleaned.csv") 


# comparing the datasets 
print ("Investigating Data Drift...") 

"""
PSI < 0.1: No major change, you can continue with the current model.
PSI < 0.2: Moderate population change, use your best judgement.

PSI >= 0.2: Significant population change, model retraining may be required.

"""

"""
Population Stability Index (PSI) for data drift detection.

PSI compares the distribution of a feature (or model score) between a
reference dataset (e.g. training data) and a current dataset (e.g. new
production/incoming data), to flag when the distribution has shifted
enough to warrant investigation or retraining.

Rule-of-thumb thresholds:
    PSI < 0.1   -> no significant shift
    0.1 <= PSI < 0.2 -> moderate shift, worth monitoring
    PSI >= 0.2  -> significant shift, investigate / consider retraining

Usage:
    from psi_drift_test import calculate_psi, psi_report

    psi_value = calculate_psi(reference_series, current_series)
    report = psi_report(reference_df, current_df, features=["age", "income"])
"""



@dataclass
class PSIResult:
    feature: str
    psi: float
    status: str  # "stable" | "moderate" | "significant"


def _bucket_edges(reference: np.ndarray, bins: int, bucket_type: str) -> np.ndarray:
    """Compute bin edges from the reference distribution only.

    Bin edges must come from the reference set — using the current set
    (or a pooled set) leaks drift information into the binning itself
    and biases PSI toward zero.
    """
    if bucket_type == "quantile":
        # Quantile-based binning: equal population per bucket in the reference set.
        quantiles = np.linspace(0, 100, bins + 1)
        edges = np.percentile(reference, quantiles)
    elif bucket_type == "uniform":
        # Equal-width binning across the reference range.
        edges = np.linspace(reference.min(), reference.max(), bins + 1)
    else:
        raise ValueError("bucket_type must be 'quantile' or 'uniform'")

    edges = np.unique(edges)  # guard against duplicate edges (e.g. many repeated values)
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
    """
    Calculate PSI between a reference distribution and a current distribution.

    Works on continuous numeric features. For categorical features use
    calculate_psi_categorical below.

    Args:
        reference: baseline values (e.g. training set feature).
        current: new values to compare against the baseline.
        bins: number of buckets.
        bucket_type: "quantile" (recommended, robust to skew) or "uniform".
        epsilon: small constant added to avoid divide-by-zero / log(0)
                 when a bucket has zero observations in either sample.

    Returns:
        PSI value (float). Higher = more drift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    edges = _bucket_edges(reference, bins, bucket_type)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # Replace zero buckets with a tiny epsilon so PSI stays finite.
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def calculate_psi_categorical(
    reference: pd.Series,
    current: pd.Series,
    epsilon: float = 1e-4,
) -> float:
    """PSI for categorical / discrete features, using observed category frequencies."""
    ref_freq = reference.value_counts(normalize=True)
    cur_freq = current.value_counts(normalize=True)

    categories = set(ref_freq.index) | set(cur_freq.index)

    psi = 0.0
    for cat in categories:
        ref_pct = ref_freq.get(cat, epsilon)
        cur_pct = cur_freq.get(cat, epsilon)
        ref_pct = max(ref_pct, epsilon)
        cur_pct = max(cur_pct, epsilon)
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
    """
    Run PSI across multiple features and return a summary DataFrame,
    sorted by PSI descending (worst drift first).
    """
    categorical_features = set(categorical_features or [])
    results = []

    for feature in features:
        if feature in categorical_features:
            psi_value = calculate_psi_categorical(
                reference_df[feature], current_df[feature]
            )
        else:
            psi_value = calculate_psi(
                reference_df[feature], current_df[feature], bins=bins
            )

        results.append(
            PSIResult(feature=feature, psi=psi_value, status=classify_psi(psi_value))
        )

    report_df = pd.DataFrame([r.__dict__ for r in results])
    return report_df.sort_values("psi", ascending=False).reset_index(drop=True)




# calculating the data drift
# old dataframe: df
# new dataframe: eod_data
# I have to specify the column name

psi = calculate_psi(reference=df['adjusted_close'], current=eod_data['adjusted_close']) 
print (f"PSI for adjusted close: {psi}") 

report = psi_report(
    df,
    eod_data,
    features=["adjusted_close", "volume", "high", 'low', 'open', 'close'],

)
print(report)

# I'll need to send the report to myself 









# ---------------------------------------------------------------------------
# Pytest-style tests you can drop into a CI pipeline (e.g. GitHub Actions)
# to fail a build/deploy when drift crosses a threshold.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    def test_psi_identical_distributions_is_near_zero():
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 5000)
        psi = calculate_psi(data, data.copy())
        assert psi < 0.01, f"Expected near-zero PSI for identical data, got {psi}"

    def test_psi_detects_mean_shift():
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, 5000)
        current = rng.normal(1.5, 1, 5000)  # meaningful mean shift
        psi = calculate_psi(reference, current)
        assert psi >= 0.2, f"Expected significant PSI for shifted data, got {psi}"

    def test_psi_categorical_new_category():
        reference = pd.Series(["a"] * 500 + ["b"] * 500)
        current = pd.Series(["a"] * 300 + ["b"] * 300 + ["c"] * 400)
        psi = calculate_psi_categorical(reference, current)
        assert psi > 0.1, f"Expected drift when a new category appears, got {psi}"

    def test_psi_report_threshold_gate():
        """Example of a CI gate: fail if ANY monitored feature exceeds 0.2."""
        rng = np.random.default_rng(0)
        reference_df = pd.DataFrame(
            {
                "feature_stable": rng.normal(0, 1, 2000),
                "feature_drifted": rng.normal(0, 1, 2000),
            }
        )
        current_df = pd.DataFrame(
            {
                "feature_stable": rng.normal(0, 1, 2000),
                "feature_drifted": rng.normal(2, 1, 2000),
            }
        )
        report = psi_report(reference_df, current_df, features=["feature_stable", "feature_drifted"])
        significant = report[report["status"] == "significant"]
        assert "feature_drifted" in significant["feature"].values
        assert "feature_stable" not in significant["feature"].values

    tests = [
        test_psi_identical_distributions_is_near_zero,
        test_psi_detects_mean_shift,
        test_psi_categorical_new_category,
        test_psi_report_threshold_gate,
    ]

    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {t.__name__} -> {e}")

    sys.exit(1 if failures else 0)