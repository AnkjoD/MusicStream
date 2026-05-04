from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "homura_madoka")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "homura123")


def main():
    spark = SparkSession.builder \
        .appName("SongCountReport_Distributed") \
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT) \
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS) \
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "6") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    silver_path = "s3a://silver-zone/datalake/silver/eventsim"
    gold_path   = "s3a://gold-zone/datalake/gold/song_rankings"

    try:
        silver_df = spark.read.parquet(silver_path)

        # ── Aggregation chạy hoàn toàn distributed ──
        # Spark tự chia data giữa các Executors, mỗi Executor tính partial count
        # rồi shuffle và reduce → không có Driver bottleneck
        song_rankings = silver_df \
            .filter(col("song").isNotNull() & col("artist").isNotNull()) \
            .groupBy("artist", "song") \
            .agg(count("*").alias("play_count")) \
            .orderBy(col("play_count").desc())

        # Ghi toàn bộ kết quả distributed (không dùng .show() trong production)
        song_rankings.write.mode("overwrite").parquet(gold_path)
        print(f"✅ [Silver→Gold] Song rankings đã được cập nhật tại: {gold_path}")

        # Log top 10 nhẹ nhàng (lấy 10 dòng thôi)
        top10 = song_rankings.limit(10).collect()
        for i, row in enumerate(top10, 1):
            print(f"   #{i}: {row['artist']} - {row['song']} ({row['play_count']} plays)")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
