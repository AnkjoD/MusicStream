#!/bin/bash
# ============================================================
# Eventsim Entrypoint - Python Producer
# NOTE: Original Scala eventsim code is kept as reference only.
# This container runs a Python-based producer that mimics
# eventsim's output schema exactly.
# ============================================================
set -e

KAFKA_BROKER="${KAFKA_BROKER:-kafka:9092}"
KAFKA_TOPIC="${KAFKA_TOPIC:-eventsim}"
NUM_USERS="${NUM_USERS:-200}"
GROWTH_RATE="${GROWTH_RATE:-0.0}"
EVENT_DELAY_MS="${EVENT_DELAY_MS:-200}"
REAL_TIME="${REAL_TIME:-true}"

echo "🎵 [Eventsim-Python] Starting music event simulation..."
echo "   Kafka Broker : ${KAFKA_BROKER}"
echo "   Kafka Topic  : ${KAFKA_TOPIC}"
echo "   Users        : ${NUM_USERS}"
echo "   Event delay  : ${EVENT_DELAY_MS}ms"
echo "   Real-time    : ${REAL_TIME}"

# Wait for Kafka TCP port (bash built-in, no netcat needed)
echo "⏳ Waiting for Kafka at ${KAFKA_BROKER}..."
HOST=$(echo "$KAFKA_BROKER" | cut -d: -f1)
PORT=$(echo "$KAFKA_BROKER" | cut -d: -f2)

until (echo > /dev/tcp/$HOST/$PORT) 2>/dev/null; do
    echo "   Kafka not ready, retrying in 5s..."
    sleep 5
done
echo "✅ Kafka is reachable! Waiting 3s for broker stabilization..."
sleep 3

# Export env vars for producer.py
export KAFKA_BROKER KAFKA_TOPIC NUM_USERS GROWTH_RATE EVENT_DELAY_MS REAL_TIME

echo "🚀 Starting Python producer..."
exec python3 /app/producer.py
