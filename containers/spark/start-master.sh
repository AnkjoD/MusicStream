#!/bin/bash
# ============================================================
# Spark Master Auto-Start Script
# Chạy Master + tự submit Streaming Job sau khi sẵn sàng
# ============================================================
set -e

echo "🚀 [Spark Master] Đang khởi động..."

# Khởi động Spark Master (background)
/opt/spark/bin/spark-class org.apache.spark.deploy.master.Master &
MASTER_PID=$!

# Đợi Master sẵn sàng (kiểm tra port 7077)
echo "⏳ Đợi Spark Master sẵn sàng..."
for i in $(seq 1 30); do
    if /opt/spark/bin/spark-class org.apache.spark.deploy.client.SparkSubmit \
        --status "driver-dummy" \
        --master spark://spark-master:7077 2>/dev/null | grep -q "Error" 2>/dev/null || \
       nc -z spark-master 7077 2>/dev/null; then
        echo "✅ Spark Master đã sẵn sàng!"
        break
    fi
    echo "   Chờ ${i}s..."
    sleep 2
done

# Thêm 5s buffer để Workers connect
sleep 5

echo "🎯 [Spark Submit] Đang submit Kafka→MinIO Streaming Job..."
# JARs đã được pre-baked vào image, không cần --packages nữa!
/opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --deploy-mode client \
    --driver-memory 1g \
    --executor-memory 1g \
    --executor-cores 1 \
    --conf spark.jars.ivy=/tmp/.ivy2 \
    --name "KafkaToMinIO_Streaming" \
    /data/jobs/streaming/kafka_to_minio.py &

echo "✅ Streaming job đã được submit!"
echo "📊 Xem logs tại: http://spark-master:8080"

# Giữ Master tiếp tục chạy (blocking)
wait $MASTER_PID
