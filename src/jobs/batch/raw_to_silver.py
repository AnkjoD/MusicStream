from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, split, regexp_extract
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "homura_madoka")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "homura123")


def create_spark(app_name: str) -> SparkSession:
    """Factory tạo SparkSession với cấu hình Distributed chuẩn."""
    return (
        SparkSession.builder
        .appName(app_name)
        # S3A / MinIO
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        # AQE: Adaptive Query Execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # Tận dụng 3 workers × 2 cores = 6 partitions
        .config("spark.sql.shuffle.partitions", "6")
        .config("spark.default.parallelism", "6")
        .getOrCreate()
    )


def main():
    spark = create_spark("RawToSilver_Distributed")
    spark.sparkContext.setLogLevel("WARN")

    raw_path    = "s3a://bronze-zone/datalake/raw/eventsim"
    silver_path = "s3a://silver-zone/datalake/silver/eventsim"

    print(f"📦 [Bronze→Silver] Đọc từ: {raw_path}")

    try:
        raw_df = spark.read.parquet(raw_path)

        # Dùng .rdd.isEmpty() thay vì .count() để tránh collect về Driver
        if raw_df.rdd.isEmpty():
            print("⚠️  Bronze layer trống. Chờ Streaming job ghi data...")
            return

        # ── Tất cả transformations chạy distributed trên Executors ──

        # a. Deduplication - Distributed sort + hash
        silver_df = raw_df.dropDuplicates(["userId", "ts", "sessionId"])

        # b. Time conversion - vectorized trên mỗi partition
        silver_df = silver_df \
            .withColumn("event_time",
                        from_unixtime(col("ts") / 1000).cast("timestamp")) \
            .withColumn("registration_time",
                        from_unixtime(col("registration") / 1000).cast("timestamp"))

        # c. Location parsing - Catalyst optimizer sẽ fuse thành 1 pass
        loc_split = split(col("location"), ", ")
        silver_df = silver_df \
            .withColumn("city",  loc_split.getItem(0)) \
            .withColumn("state", loc_split.getItem(1))

        # d. UserAgent parsing
        silver_df = silver_df \
            .withColumn("browser",
                        regexp_extract(col("userAgent"),
                                       r"(Firefox|Chrome|Safari|Opera|Edge)", 1)) \
            .withColumn("os",
                        regexp_extract(col("userAgent"),
                                       r"(Windows|Macintosh|Android|iPhone|Linux)", 1)) \
            .na.fill({"browser": "Unknown", "os": "Unknown"})

        # ── Ghi song song tất cả partitions → MinIO ──
        print(f"💾 [Bronze→Silver] Ghi vào Silver: {silver_path}")
        silver_df.write \
            .mode("append") \
            .partitionBy("page") \
            .parquet(silver_path)

        print("✅ [Bronze→Silver] Hoàn tất!")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
