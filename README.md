# Credit Scoring MLOps: FastAPI, Docker & Drift Monitoring

A compact machine-learning deployment workflow that takes a credit-scoring model from reproducible training to serialization, API serving, containerization, and monitoring.

## Highlights

- Reproducible synthetic credit-scoring data
- End-to-end scikit-learn preprocessing and model pipeline
- Serialized model artifact and versioned manifest
- FastAPI prediction and health endpoints
- Docker packaging
- Clear separation between training and serving code

## Tech stack

Python · pandas · NumPy · scikit-learn · joblib · FastAPI · Uvicorn · Docker

## Repository structure

- `src/train_model.py` — trains and serializes the scoring pipeline
- `src/app.py` — FastAPI inference service
- `deploy/Dockerfile` — container definition
- `requirements.txt` — runtime dependencies

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
uvicorn src.app:app --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive API documentation.

The dataset is synthetic; no private financial data are included.
