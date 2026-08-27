import numpy as np
import pandas as pd

from src.monitoring import drift_report, population_stability_index


def test_psi_identical_distribution_is_near_zero():
    x = np.linspace(-2, 2, 500)
    psi = population_stability_index(x, x.copy())
    assert abs(psi) < 1e-10


def test_psi_detects_strong_shift():
    rng = np.random.default_rng(42)
    reference = rng.normal(0.0, 1.0, 2000)
    current = rng.normal(1.5, 1.0, 2000)
    assert population_stability_index(reference, current) > 0.25


def test_drift_report_returns_status_per_feature():
    reference = pd.DataFrame({"income": [1, 2, 3, 4, 5] * 20})
    current = pd.DataFrame({"income": [1, 2, 3, 4, 5] * 20})
    report = drift_report(reference, current)
    assert report.loc[0, "feature"] == "income"
    assert report.loc[0, "status"] == "stable"
