"""Central configuration for the scoring service."""
from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_DIR = Path(os.getenv("MODEL_ARTIFACT_DIR", "artifacts"))
MODEL_PATH = ARTIFACT_DIR / "modelo_scoring.joblib"
MANIFEST_PATH = ARTIFACT_DIR / "model_manifest.json"
DRIFT_REFERENCE_PATH = ARTIFACT_DIR / "reference_profile.json"
DEFAULT_THRESHOLD = float(os.getenv("SCORING_THRESHOLD", "0.50"))
