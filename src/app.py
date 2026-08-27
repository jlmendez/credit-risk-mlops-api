from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "modelo_scoring.joblib"
MANIFEST_PATH = BASE_DIR / "artifacts" / "model_manifest.json"

if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
    raise RuntimeError("Run src/train_model.py first to generate model artifacts under artifacts/.")

model = joblib.load(MODEL_PATH)
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
THRESHOLD = float(manifest["decision_threshold"])
app = FastAPI(title="Credit Scoring API", version=manifest.get("model_version", "0.1"))


class ScoringRequest(BaseModel):
    edad: int = Field(ge=18, le=85)
    ingreso_mensual: float = Field(gt=0)
    ratio_deuda_ingreso: float = Field(ge=0, le=2)
    atrasos_12m: int = Field(ge=0, le=20)
    utilizacion_credito: float = Field(ge=0, le=1.5)
    antiguedad_laboral: float = Field(ge=0, le=60)
    tipo_empleo: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": manifest.get("model_version")}


@app.post("/predict")
def predict(data: ScoringRequest):
    try:
        frame = pd.DataFrame([data.model_dump()])
        probability = float(model.predict_proba(frame)[0, 1])
        return {
            "default_probability": round(probability, 6),
            "high_risk": probability >= THRESHOLD,
            "threshold": THRESHOLD,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
