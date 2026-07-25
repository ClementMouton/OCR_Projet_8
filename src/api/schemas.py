from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str | None = None
    decision_threshold: float | None = None
    model_loaded: bool
    feature_count: int | None = None


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Variables client attendues par le pipeline.",
    )


class PredictionResponse(BaseModel):
    default_probability: float
    prediction: int
    decision_threshold: float
    decision_label: str
