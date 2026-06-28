from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.sql.utils import AnalysisException
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioadmin")


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
        silver_df = spark.read.option("recursiveFileLookup", "true").parquet(silver_path)

        # ── Gom nhóm và tính toán phân tán (distributed aggregation) ──
        # Spark sẽ tự xé nhỏ data cho các Executor tính toán cục bộ trước, 
        # sau đó mới shuffle và gom lại kết quả cuối cùng. Cách này tránh bị nghẽn ở nút Driver.
        song_rankings = silver_df \
            .filter(col("song").isNotNull() & col("artist").isNotNull()) \
            .groupBy("artist", "song") \
            .agg(count("*").alias("play_count")) \
            .orderBy(col("play_count").desc())
            
        print(f"⏳ [Gold] Đang tổng hợp Bảng Xếp Hạng vào {gold_path} (Tốc độ tối đa)...")

        # Ghi thẳng kết quả xuống tầng chứa (Gold Layer).
        # Tuyệt đối không gọi .show() hay .collect() toàn bộ dữ liệu ở môi trường production vì sẽ gây sập Driver.
        song_rankings.write.mode("overwrite").parquet(gold_path)
        print(f"✅ [Gold] Chúc mừng! Đã hoàn tất ghi Bảng xếp hạng vào Gold Layer.")

        # Chỉ in ra Top 10 bài hát hot nhất để debug nhanh trên console.
        top10 = song_rankings.limit(10).collect()
        for i, row in enumerate(top10, 1):
            print(f"   #{i}: {row['artist']} - {row['song']} ({row['play_count']} plays)")

    except AnalysisException as e:
        print(f"⚠️  Silver layer chưa có dữ liệu. Chờ Bronze→Silver chạy xong trước. ({e})")
        return
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
