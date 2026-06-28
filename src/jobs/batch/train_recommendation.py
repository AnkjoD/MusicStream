from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, count, abs as spark_abs, hash as spark_hash
from pyspark.sql.utils import AnalysisException
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioadmin")


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
        # ALS cần kha khá RAM cho việc phân rã ma trận (matrix factorization)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "6")
        # Dùng Kryo serializer để tuần tự hóa dữ liệu nhanh hơn hẳn so với serializer mặc định của Java
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        # Giới hạn in log string tránh làm trôi màn hình console
        .config("spark.sql.debug.maxToStringFields", "50")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # ── Định nghĩa các đường dẫn lưu trữ trên MinIO ──
    silver_path = "s3a://silver-zone/datalake/silver/eventsim"
    gold_path   = "s3a://gold-zone/datalake/gold/recommendations"
    model_path  = "s3a://gold-zone/models/als_recommendation"

    print(f"🎵 [ALS] Đọc dữ liệu từ Silver Layer: {silver_path}")

    try:
        df = spark.read.parquet(silver_path)

        # ── Tính rating ngầm (implicit): User nghe bài hát bao nhiêu lần thì coi như thích bấy nhiêu ──
        # Gom nhóm phân tán trên toàn bộ các Executor
        rating_df = df \
            .filter(col("userId").isNotNull() & col("song").isNotNull()) \
            .groupBy("userId", "song") \
            .agg(count("*").alias("play_count"))

        # ── Dùng hàm Hash để tự sinh ID số nguyên thay cho StringIndexer ──
        # StringIndexer của Spark bắt buộc phải gom (collect) toàn bộ giá trị về máy Driver để gán nhãn, dễ gây tràn RAM/lỗi EOF.
        # Dùng hash thì tự tính toán song song hoàn toàn ngay trên các Worker, không cần gom về Driver.
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


        # ── Thuật toán ALS: Phân rã ma trận phân tán ──
        # Spark sẽ tự xé nhỏ ma trận User và ma trận Song ra các Executor để xử lý độc lập.
        # Nhờ vậy không có máy nào phải chứa toàn bộ ma trận khổng lồ.
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

        # Gợi ý Top 10 bài hát cho toàn bộ danh sách user (hoạt động map phân tán)
        print(f"💾 [ALS] Lưu gợi ý vào: {gold_path}")
        recommendations = model.recommendForAllUsers(10)
        recommendations.write.mode("overwrite").parquet(gold_path)
        
        # Lưu lại bảng mapping ID để Dashboard Streamlit sau này dịch ngược ID số thành tên người dùng và bài hát thật
        user_mapping.write.mode("overwrite").parquet("s3a://gold-zone/datalake/gold/user_mapping")
        song_mapping.write.mode("overwrite").parquet("s3a://gold-zone/datalake/gold/song_mapping")

        # Export model đã train để phục vụ đợt gợi ý tiếp theo
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
