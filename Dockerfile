FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt

# Dados NLTK usados pelo topic_clusterer (stopwords PT-BR)
RUN python -m nltk.downloader stopwords punkt punkt_tab

# Apenas os módulos necessários para a API
COPY api/ api/
COPY aggregation/ aggregation/
COPY ideological/ ideological/
COPY scripts/ scripts/

EXPOSE 8080

CMD ["gunicorn", "api.app:app", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--threads", "4", \
     "--worker-class", "gthread", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
