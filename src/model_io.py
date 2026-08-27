"""Model serialization, manifests and stable artifact loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_model_bundle(model: Any, model_path: Path, manifest_path: Path, metadata: dict) -> dict:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    manifest = {**metadata, "artifact_sha256": sha256_file(model_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_model_bundle(model_path: Path, manifest_path: Path):
    model = joblib.load(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256_file(model_path)
    if actual != manifest.get("artifact_sha256"):
        raise ValueError("Model artifact checksum does not match the manifest")
    return model, manifest
