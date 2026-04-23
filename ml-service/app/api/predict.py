"""
Prediction API endpoints.

POST /predict      — Single-user churn prediction.
POST /predict/batch — Batch churn prediction (up to 1000 users per call).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logger import get_logger
from app.features.validators import BatchScoringRequest, RawUserActivity
from app.services.inference_service import inference_service

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])

# ── Response schemas ──────────────────────────────────────────────────────────


class PredictResponse(BaseModel):
    """Churn prediction output with enterprise metadata."""

    customer_id: str
    name: Optional[str] = None
    segment: Optional[str] = None
    subscription_plan: Optional[str] = None
    current_status: Optional[str] = None
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    predicted_revenue_loss: Optional[str] = None
    last_active_date: Optional[str] = None
    forecast_month: Optional[str] = None
    recommended_action: Optional[str] = None
    source: str = "ML-Engine-v2"

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "C-001",
                "churn_probability": 0.78,
                "risk_level": "HIGH",
                "predicted_revenue_loss": "$1,200",
                "recommended_action": "Incentivize renewal with 20% discount"
            }
        }
    }


class BatchPredictResponse(BaseModel):
    """Batch prediction response envelope."""

    predictions: list[PredictResponse]
    total: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("", response_model=PredictResponse, summary="Single-user churn prediction")
def predict_single(body: RawUserActivity) -> PredictResponse:
    """
    Predict churn probability for a single user.

    **Request body**: A `RawUserActivity` JSON object.
    **Response**: `churn_score` (float) + `risk_level` (LOW/MEDIUM/HIGH).

    Node.js compatible — pure JSON in/out.
    """
    try:
        result = inference_service.predict_one(body.model_dump())
        return PredictResponse(**result)
    except RuntimeError as exc:
        logger.error("Prediction endpoint error", extra={"error": str(exc)})
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected prediction error", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Internal inference error") from exc


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Batch churn prediction",
)
def predict_batch_endpoint(body: BatchScoringRequest) -> BatchPredictResponse:
    """
    Score a batch of users in a single API call.

    Accepts up to **1 000 users** per request.  For larger datasets use the
    standalone batch scoring pipeline (`pipelines/batch_scoring.py`).
    """
    if len(body.users) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeds 1000. Use the batch scoring pipeline for larger datasets.",
        )

    try:
        records = [u.model_dump() for u in body.users]
        results = inference_service.predict_many(records)

        predictions = [PredictResponse(**r) for r in results]
        high   = sum(1 for p in predictions if p.risk_level == "HIGH")
        medium = sum(1 for p in predictions if p.risk_level == "MEDIUM")
        low    = sum(1 for p in predictions if p.risk_level == "LOW")

        return BatchPredictResponse(
            predictions=predictions,
            total=len(predictions),
            high_risk_count=high,
            medium_risk_count=medium,
            low_risk_count=low,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Batch prediction endpoint error", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Internal batch inference error") from exc
