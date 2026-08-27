"""Train and serialize a reproducible synthetic credit-scoring pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 50311
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def make_data(n: int = 3000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "edad": rng.integers(18, 76, n),
        "ingreso_mensual": rng.lognormal(np.log(6500), 0.55, n),
        "ratio_deuda_ingreso": np.clip(rng.beta(2.1, 5.2, n) * 1.25, 0, 1.6),
        "atrasos_12m": np.clip(rng.poisson(0.7, n), 0, 12),
        "utilizacion_credito": np.clip(rng.beta(2.2, 3.8, n) * 1.35, 0, 1.5),
        "antiguedad_laboral": np.clip(rng.gamma(2.4, 2.1, n), 0, 35),
        "tipo_empleo": rng.choice(["asalariado", "independiente", "temporal"], n,
                                  p=[0.66, 0.25, 0.09]),
    })
    logit = (
        -2.55 + 3.0 * df["ratio_deuda_ingreso"] + 0.48 * df["atrasos_12m"]
        + 2.1 * df["utilizacion_credito"] - 0.000055 * df["ingreso_mensual"]
        - 0.035 * df["antiguedad_laboral"]
        + 0.35 * (df["tipo_empleo"] == "temporal").astype(float)
        + rng.normal(0, 0.65, n)
    )
    df["incumplimiento"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


def build_pipeline() -> Pipeline:
    numeric = ["edad", "ingreso_mensual", "ratio_deuda_ingreso", "atrasos_12m",
               "utilizacion_credito", "antiguedad_laboral"]
    categorical = ["tipo_empleo"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", prep),
                     ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def main() -> None:
    data = make_data()
    X = data.drop(columns="incumplimiento")
    y = data["incumplimiento"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    model = build_pipeline().fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, prob))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS / "modelo_scoring.joblib")
    manifest = {"model_version": "1.0.0", "decision_threshold": 0.50,
                "roc_auc_test": round(auc, 4), "seed": SEED}
    (ARTIFACTS / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
