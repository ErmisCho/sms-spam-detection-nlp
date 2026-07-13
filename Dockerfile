FROM ghcr.io/astral-sh/uv:0.11.21 AS uv

FROM node:20.20-alpine AS frontend

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    SMS_SPAM_MODEL_PATH=/models/tfidf_classifier.joblib \
    SMS_SPAM_FRONTEND_DIST=/app/frontend/dist \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY --from=frontend /frontend/dist ./frontend/dist

RUN uv sync --locked --no-dev --no-editable \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"]

CMD ["uvicorn", "sms_spam_ham_analysis.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
