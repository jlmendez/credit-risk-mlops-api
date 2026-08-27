"""Pydantic request/response contracts for the FastAPI layer."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    income: float = Field(gt=0)
    debt_ratio: float = Field(ge=0, le=1)
    age: int = Field(ge=18, le=100)
    employment_years: float = Field(ge=0)
    arrears_12m: int = Field(ge=0)
    requested_amount: float = Field(gt=0)


class ScoreResponse(BaseModel):
    default_probability: float
    predicted_default: int
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
