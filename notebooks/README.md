# Analytical notebook

`mlops_pipeline_walkthrough.ipynb` is a compact portfolio edition of the larger notebook-to-production workflow. It connects model training and persistence with the API, Docker packaging, checksummed model manifests and a simple drift-monitoring example.

The reusable implementation remains in `src/`; the notebook is intentionally orchestration-focused rather than duplicating application code.
