"""
Inference service.

Orchestrates feature engineering + prediction for both single and batch
requests.  The API layer calls this service; no ML logic lives in the routes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.logger import get_logger
from app.features.builder import build_feature_dataframe, build_single_feature_dict
from app.models.predictor import predict_batch, predict_single

logger = get_logger(__name__)


class InferenceService:
    """
    Service layer for churn inference.

    Keeps API handlers thin by centralising feature engineering and prediction
    orchestration in one place.
    """

    def predict_one(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Predict churn for a single user.

        Args:
            raw: Validated raw activity dict (from :class:`~app.features.validators.RawUserActivity`).

        Returns:
            ``{"user_id": str, "churn_score": float, "risk_level": str}``
        """
        user_id = raw.get("user_id", "unknown")
        logger.info("Single inference request", extra={"user_id": user_id})

        result = predict_single(raw)
        return {"user_id": user_id, **result}

    def predict_many(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Predict churn for a batch of users.

        Args:
            records: List of validated raw activity dicts.

        Returns:
            List of ``{"user_id": str, "churn_score": float, "risk_level": str}``.
        """
        logger.info("Batch inference request", extra={"n_users": len(records)})

        user_ids = [r.get("user_id", f"user_{i}") for i, r in enumerate(records)]
        feature_df = build_feature_dataframe(records)
        preds = predict_batch(feature_df)

        return [{"user_id": uid, **pred} for uid, pred in zip(user_ids, preds)]


# Module-level singleton for DI into routes
inference_service = InferenceService()
