from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, count, abs as spark_abs, hash as spark_hash
from pyspark.sql.utils import AnalysisException
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "homura_madoka")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "homura123")


def main():
    # SparkSession - Distributed ML config
    spark = (
        SparkSession.builder
        .appName("MusicRecommendation_ALS_Distributed")
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # ALS cần nhiều memory cho matrix factorization
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "6")
        # Kryo serializer: nhanh hơn Java default serializer
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        # Suppress WARN SparkStringUtils: plan truncated
        .config("spark.sql.debug.maxToStringFields", "50")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # ── Paths: ĐÃ SỬA từ HDFS sang MinIO ──
    silver_path = "s3a://silver-zone/datalake/silver/eventsim"
    gold_path   = "s3a://gold-zone/datalake/gold/recommendations"
    model_path  = "s3a://gold-zone/models/als_recommendation"

    print(f"🎵 [ALS] Đọc dữ liệu từ Silver Layer: {silver_path}")

    try:
        df = spark.read.parquet(silver_path)

        # ── Tính implicit rating: số lần user nghe bài ──
        # Chạy distributed GroupBy trên tất cả Executors
        rating_df = df \
            .filter(col("userId").isNotNull() & col("song").isNotNull()) \
            .groupBy("userId", "song") \
            .agg(count("*").alias("play_count"))

        # ── Hash-based Integer Index (thay thế StringIndexer) ──
        # StringIndexer cần collect() TẤT CẢ unique values → EOFException khi data lớn
        # Hash-based: 100% distributed, không collect() gì cả
        user_mapping = rating_df.select(
            (spark_abs(spark_hash(col("userId"))) % 100000).cast("int").alias("user_idx"),
            "userId"
        ).dropDuplicates(["user_idx"])

        song_mapping = rating_df.select(
            (spark_abs(spark_hash(col("song"))) % 100000).cast("int").alias("song_idx"),
            "song"
        ).dropDuplicates(["song_idx"])

        model_data = rating_df.select(
            (spark_abs(spark_hash(col("userId"))) % 100000).cast("int").alias("user_idx"),
            (spark_abs(spark_hash(col("song")))   % 100000).cast("int").alias("song_idx"),
            col("play_count").cast("float")
        )


        # ── ALS: Matrix Factorization - Fully Distributed ──
        # ALS chia user matrix và item matrix ra các Executors
        # Mỗi Executor giữ một "block" của ma trận → không cần 1 máy chứa hết
        als = ALS(
            maxIter=10,
            regParam=0.1,
            rank=10,
            userCol="user_idx",
            itemCol="song_idx",
            ratingCol="play_count",
            coldStartStrategy="drop",
            nonnegative=True,
        )

        train_df, test_df = model_data.randomSplit([0.8, 0.2], seed=42)

        print("🚀 [ALS] Bắt đầu Matrix Factorization distributed...")
        model = als.fit(train_df)

        # Đánh giá
        predictions = model.transform(test_df)
        evaluator = RegressionEvaluator(
            metricName="rmse", labelCol="play_count", predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)
        print(f"   📈 RMSE: {rmse:.4f}")

        # Top 10 gợi ý cho tất cả users - distributed map operation
        print(f"💾 [ALS] Lưu gợi ý vào: {gold_path}")
        recommendations = model.recommendForAllUsers(10)
        recommendations.write.mode("overwrite").parquet(gold_path)
        
        # Lưu mapping để Dashboard dịch ngược ID ra tên thật
        user_mapping.write.mode("overwrite").parquet("s3a://gold-zone/datalake/gold/user_mapping")
        song_mapping.write.mode("overwrite").parquet("s3a://gold-zone/datalake/gold/song_mapping")

        # Lưu model để inference
        model.write().overwrite().save(model_path)

        print(f"✅ [ALS] Training hoàn tất! RMSE={rmse:.4f}")

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
