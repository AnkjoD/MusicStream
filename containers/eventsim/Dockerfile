# ============================================================
# Eventsim - Python Event Producer
# Replaces legacy Scala eventsim at the Docker level.
# Original eventsim source code is UNTOUCHED (reference only).
# 
# Build context: streamlify/ (root)
# ============================================================

FROM python:3.11-slim

# System deps (curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps first for layer caching
COPY containers/eventsim/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Producer script
COPY containers/eventsim/producer.py ./producer.py
COPY containers/eventsim/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["./entrypoint.sh"]