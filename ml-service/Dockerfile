# ── Builder stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="mlops-team"
LABEL description="ChurnPredictionService - FastAPI + XGBoost + MLflow"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY pipelines/ ./pipelines/

# Create data directories
RUN mkdir -p data/batch_results mlflow

# Non-root user for security
RUN useradd -m -u 1001 mlservice && chown -R mlservice:mlservice /app
USER mlservice

# ── Environment defaults (override via docker run -e or compose env_file) ─────
ENV APP_NAME="ChurnPredictionService" \
    APP_VERSION="1.0.0" \
    ENV="production" \
    DEBUG="false" \
    API_HOST="0.0.0.0" \
    API_PORT="8000" \
    LOG_LEVEL="INFO" \
    LOG_FORMAT="json" \
    MLFLOW_TRACKING_URI="http://mlflow:5000" \
    MLFLOW_MODEL_NAME="ChurnModel" \
    MLFLOW_MODEL_STAGE="Production" \
    CHURN_WINDOW_DAYS="30" \
    RISK_LOW_THRESHOLD="0.3" \
    RISK_HIGH_THRESHOLD="0.7"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Use exec form so signals propagate correctly to uvicorn
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--no-access-log"]
