"""Small production-oriented monitoring utilities for feature drift."""
from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(reference, current, bins: int = 10) -> float:
    """Population Stability Index using quantile bins from the reference sample."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, numeric_columns=None) -> pd.DataFrame:
    columns = numeric_columns or reference.select_dtypes(include="number").columns.intersection(current.columns).tolist()
    rows = []
    for col in columns:
        psi = population_stability_index(reference[col].dropna(), current[col].dropna())
        rows.append({"feature": col, "psi": psi, "status": "high" if psi >= 0.25 else "watch" if psi >= 0.10 else "stable"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
