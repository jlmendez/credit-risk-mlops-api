# Credit Scoring MLOps: FastAPI, Docker & Drift Monitoring

[![CI](https://github.com/jlmendez/credit-risk-mlops-api/actions/workflows/ci.yml/badge.svg)](https://github.com/jlmendez/credit-risk-mlops-api/actions/workflows/ci.yml)

A compact, production-oriented machine-learning workflow that takes a synthetic credit-scoring model from **reproducible training** to **artifact management, API serving, containerization and drift monitoring**.

## Architecture

```mermaid
flowchart LR
    A[Synthetic credit data] --> B[Preprocessing pipeline]
    B --> C[Logistic Regression]
    C --> D[Serialized model]
    C --> E[Model manifest]
    D --> F[FastAPI service]
    E --> F
    F --> G[/health]
    F --> H[/predict]
    I[Reference data] --> J[PSI drift monitor]
    K[Current data] --> J
    J --> L[Stable / Watch / High]
    F --> M[Docker image]
```

The project separates the main MLOps responsibilities rather than placing the full lifecycle in one script.

## What this demonstrates

- reproducible synthetic data generation and model training;
- end-to-end scikit-learn preprocessing and scoring;
- versioned model artifacts and manifest validation;
- typed request/response contracts with Pydantic;
- FastAPI health and prediction endpoints;
- Docker packaging for portable serving;
- Population Stability Index (PSI) monitoring for numeric feature drift;
- separation of configuration, training, serving, serialization and monitoring concerns.

## Validation signals

The automated test suite checks properties that should remain true as the project evolves:

| Check | Expected behavior |
|---|---|
| PSI on identical distributions | approximately zero |
| PSI after a strong distribution shift | larger than the baseline PSI |
| Drift report | returns a feature-level PSI and status |
| CI | installs dependencies and runs `pytest` on every push / pull request |

## Repository structure

```text
.
├── deploy/
│   └── Dockerfile
├── notebooks/
│   ├── README.md
│   └── mlops_pipeline_walkthrough.ipynb
├── src/
│   ├── app.py
│   ├── config.py
│   ├── model_io.py
│   ├── monitoring.py
│   ├── schemas.py
│   └── train_model.py
├── tests/
│   └── test_monitoring.py
├── .github/workflows/
│   └── ci.yml
└── requirements.txt
```

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

Run the automated checks with:

```bash
pip install pytest
pytest -q
```

## Deployment path

```text
Train → Validate → Serialize → Serve → Containerize → Monitor
```

The dataset is fully synthetic; no private financial data or credentials are included. This repository is intended as a concise demonstration of the engineering path around an ML model, not merely the model itself.
