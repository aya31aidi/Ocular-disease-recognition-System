FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements-ml.txt

COPY . .
RUN useradd --create-home appuser && mkdir -p instance && chown -R appuser /app
USER appuser

EXPOSE 8000
# One worker keeps a single copy of the ~160 MB model in memory; threads handle concurrency.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
