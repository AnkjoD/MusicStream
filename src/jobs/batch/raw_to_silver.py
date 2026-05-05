from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, split, regexp_extract, max as spark_max
from pyspark.sql.utils import AnalysisException
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "homura_madoka")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "homura123")


def create_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.hadoop.fs.s3a.endpoint",          MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key",        MINIO_ACCESS)
        .config("spark.hadoop.fs.s3a.secret.key",        MINIO_SECRET)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl",              "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.fast.upload",       "true")
        .config("spark.sql.adaptive.enabled",                    "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled",           "true")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism",    "4")
        .config("spark.sql.parquet.mergeSchema", "false")
        .getOrCreate()
    )


def get_max_ts(spark: SparkSession, silver_path: str) -> int:
    """
    Lấy timestamp lớn nhất trong Silver.
    Trả về 0 nếu Silver chưa tồn tại (lần đầu chạy).
    KHÔNG dùng isEmpty/count — tránh EOFException.
    """
    try:
        row = spark.read.parquet(silver_path).agg(spark_max("ts").alias("max_ts")).collect()
        val = row[0]["max_ts"]
        return int(val) if val is not None else 0
    except Exception:
        return 0


def transform(df):
    loc_split = split(col("location"), ", ")
    return (
        df
        .filter(col("userId").isNotNull() & col("ts").isNotNull())
        .dropDuplicates(["userId", "ts", "sessionId"])
        .withColumn("event_time",        from_unixtime(col("ts") / 1000).cast("timestamp"))
        .withColumn("registration_time", from_unixtime(col("registration") / 1000).cast("timestamp"))
        .withColumn("city",    loc_split.getItem(0))
        .withColumn("state",   loc_split.getItem(1))
        .withColumn("browser", regexp_extract(col("userAgent"), r"(Firefox|Chrome|Safari|Opera|Edge)", 1))
        .withColumn("os",      regexp_extract(col("userAgent"), r"(Windows|Macintosh|Android|iPhone|Linux)", 1))
        .na.fill({"browser": "Unknown", "os": "Unknown"})
    )


def main():
    spark = create_spark("RawToSilver_Incremental")
    spark.sparkContext.setLogLevel("WARN")

    raw_path    = "s3a://bronze-zone/datalake/raw/eventsim"
    silver_path = "s3a://silver-zone/datalake/silver/eventsim"

    print(f"📦 [Bronze→Silver] Đọc từ: {raw_path}")

    # ── Đọc Bronze ──
    try:
        raw_df = spark.read.parquet(raw_path)
    except AnalysisException:
        print("⚠️  Bronze layer chưa có dữ liệu. Streaming job chưa ghi file.")
        spark.stop()
        return

    # ── Lấy max_ts từ Silver (thuần SQL agg, không scan toàn bộ) ──
    max_ts = get_max_ts(spark, silver_path)

    if max_ts == 0:
        print("🆕 [Silver] Lần đầu chạy — nạp toàn bộ Bronze vào Silver.")
        new_df = raw_df
    else:
        print(f"🔍 [Silver] Đã có dữ liệu tới ts={max_ts}. Chỉ nạp phần mới hơn.")
        new_df = raw_df.filter(col("ts") > max_ts)

    # ── Transform (Catalyst optimizer fuse thành 1 pass) ──
    silver_df = transform(new_df)

    # ── Ghi vào Silver — append mode hoạt động ngay cả khi thư mục chưa tồn tại ──
    print(f"⏳ [Silver] Ghi (append) vào: {silver_path}")

    (
        silver_df.write
        .mode("append")
        .partitionBy("page")
        .parquet(silver_path)
    )

    print("✅ [Silver] Hoàn tất!")
    spark.stop()


if __name__ == "__main__":
    main()
