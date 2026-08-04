# Multi-stage: model weights are baked in at build time so the container has no
# cold-start download and no runtime dependency on HuggingFace being reachable.
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[serve,models]"

FROM python:3.11-slim
RUN useradd --create-home --uid 10001 app
COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=app:app src ./src
COPY --chown=app:app evals ./evals
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    FINRAG_EMBEDDING_BACKEND=hash
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "finrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
