import sys
import os
import pandas as pd
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f"{BASE_DIR}/scripts/")

from check_data_drift import calculate_psi, calculate_psi_categorical, psi_report


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
