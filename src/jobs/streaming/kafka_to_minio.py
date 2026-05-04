from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp, from_unixtime
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType
import os

# Schema events từ eventsim (khớp với producer.py)
SCHEMA = StructType([
    StructField("ts",            LongType(),   True),
    StructField("userId",        StringType(), True),
    StructField("sessionId",     LongType(),   True),
    StructField("page",          StringType(), True),
    StructField("auth",          StringType(), True),
    StructField("method",        StringType(), True),
    StructField("status",        LongType(),   True),
    StructField("level",         StringType(), True),
    StructField("itemInSession", LongType(),   True),
    StructField("location",      StringType(), True),
    StructField("userAgent",     StringType(), True),
    StructField("lastName",      StringType(), True),
    StructField("firstName",     StringType(), True),
    StructField("registration",  LongType(),   True),
    StructField("gender",        StringType(), True),
    StructField("artist",        StringType(), True),
    StructField("song",          StringType(), True),
    StructField("duration",      DoubleType(), True),
])

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT",  "http://minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "homura_madoka")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "homura123")
KAFKA_BROKER   = os.getenv("KAFKA_BROKER",     "kafka:9092")
KAFKA_TOPIC    = os.getenv("KAFKA_TOPIC",      "eventsim")


def create_spark() -> SparkSession:
    """Tạo SparkSession với S3A + AQE config."""
    return (
        SparkSession.builder
        .appName("KafkaToMinIO_Streaming_Distributed")
        # S3A / MinIO
        .config("spark.hadoop.fs.s3a.endpoint",             MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key",           MINIO_ACCESS)
        .config("spark.hadoop.fs.s3a.secret.key",           MINIO_SECRET)
        .config("spark.hadoop.fs.s3a.path.style.access",    "true")
        .config("spark.hadoop.fs.s3a.impl",                 "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.fast.upload",          "true")
        .config("spark.hadoop.fs.s3a.multipart.size",       "67108864")
        # Streaming
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        # AQE
        .config("spark.sql.adaptive.enabled",                      "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",   "true")
        # Parallelism: 3 workers × 2 cores = 6
        .config("spark.default.parallelism",    "6")
        .config("spark.sql.shuffle.partitions", "6")
        .getOrCreate()
    )


def main():
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    # Đọc từ Kafka
    kafka_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe",               KAFKA_TOPIC)
        .option("startingOffsets",         "earliest")
        .option("failOnDataLoss",          "false")
        .option("maxOffsetsPerTrigger",    "10000")
        .load()
    )

    # Parse JSON
    parsed_df = (
        kafka_df
        .selectExpr("CAST(value AS STRING)")
        .select(from_json(col("value"), SCHEMA).alias("data"))
        .select("data.*")
        .withColumn("ingestion_time", current_timestamp())
        .withColumn("event_time", from_unixtime(col("ts") / 1000).cast("timestamp"))
        .filter(col("ts").isNotNull())
    )

    # Ghi vào MinIO Bronze (Parquet, partitioned by page)
    query_minio = (
        parsed_df.writeStream
        .format("parquet")
        .option("path",               "s3a://bronze-zone/datalake/raw/eventsim")
        .option("checkpointLocation", "s3a://bronze-zone/datalake/checkpoints/eventsim_raw")
        .partitionBy("page")
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .start()
    )

    print(f"✅ [Streaming] Kafka→MinIO pipeline đang chạy...")
    print(f"   Kafka: {KAFKA_BROKER} | Topic: {KAFKA_TOPIC}")
    print(f"   MinIO: {MINIO_ENDPOINT} | Bucket: bronze-zone")
    query_minio.awaitTermination()


if __name__ == "__main__":
    main()
