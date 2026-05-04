"""
Churn Prediction với XGBoost Distributed trên Spark Cluster
============================================================

MÔ HÌNH NÀY LÀM GÌ?
    XGBoost dự đoán user nào có nguy cơ "downgrade" từ PAID → FREE
    (tức là "churn" - rời bỏ gói Premium)

    Input features (từ Silver Layer):
        - Số lần nghe nhạc trong 7 ngày
        - Tỷ lệ nghe hoàn thành bài (vs skip)
        - Số session trung bình mỗi ngày
        - Tỷ lệ dùng Thumbs Down
        - Số lần xem trang Settings / Help
        - Giới tính, OS, Browser
    
    Output: Xác suất churn (0-1), lưu vào Gold Layer
    Use case thực tế: Gửi ưu đãi cho user sắp rời đi

FULLY DISTRIBUTED VỚI SPARKXGB:
    - Sử dụng xgboost4j-spark (native Spark distribution)
    - Dữ liệu được chia đều giữa các Spark Workers
    - Training song song thực sự, không bottleneck ở Driver
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, datediff,
    max as spark_max, min as spark_min, lit, when, to_date
)
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import os


def create_spark_session():
    """Tạo SparkSession với cấu hình tối ưu cho Distributed ML."""
    return (
        SparkSession.builder
        .appName("ChurnPrediction_XGBoost_Distributed")
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("MINIO_ENDPOINT", "http://minio:9000"))
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY", "homura_madoka"))
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY", "homura123"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # Tối ưu cho distributed training
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def build_user_features(silver_df):
    """
    Feature Engineering: Tổng hợp hành vi user từ event logs.
    Mỗi user → 1 dòng với các feature aggregated.
    """
    # Tham chiếu ngày cuối cùng trong dataset
    max_date = silver_df.agg(spark_max("event_time")).collect()[0][0]

    user_features = silver_df.groupBy("userId", "level", "gender").agg(
        # Tổng số bài đã nghe
        count(when(col("page") == "NextSong", True)).alias("total_songs"),
        # Số lần "Thumbs Down" (bài không thích)
        count(when(col("page") == "Thumbs Down", True)).alias("thumbs_down"),
        # Số lần "Thumbs Up"
        count(when(col("page") == "Thumbs Up", True)).alias("thumbs_up"),
        # Số lần vào trang Settings (dấu hiệu muốn hủy)
        count(when(col("page") == "Settings", True)).alias("settings_visits"),
        # Số lần vào trang Help
        count(when(col("page") == "Help", True)).alias("help_visits"),
        # Số lần vào trang Cancel (tín hiệu churn mạnh nhất)
        count(when(col("page") == "Cancellation Confirmation", True)).alias("cancel_count"),
        # Số session phân biệt
        count("sessionId").alias("total_sessions"),
        # Số ngày active
        (datediff(
            to_date(lit(max_date)),
            to_date(spark_min("event_time"))
        ) + 1).alias("days_active"),
    )

    # Tỷ lệ bài hát không thích (thumbs_down / total_songs)
    user_features = user_features.withColumn(
        "dislike_ratio",
        when(col("total_songs") > 0,
             col("thumbs_down") / col("total_songs")).otherwise(0.0)
    ).withColumn(
        # Sessions trung bình mỗi ngày
        "avg_sessions_per_day",
        when(col("days_active") > 0,
             col("total_sessions") / col("days_active")).otherwise(0.0)
    )

    return user_features


def create_churn_label(user_features):
    """
    Label: user có bị churn không?
    Logic: Nếu cancel_count > 0 HOẶC level == 'free' và từng là paid → churn=1
    (Dùng cancel_count vì eventsim sinh ra sự kiện 'Cancellation Confirmation')
    """
    return user_features.withColumn(
        "churn",
        when(
            (col("cancel_count") > 0) | (col("level") == "free"),
            lit(1)
        ).otherwise(lit(0))
    )


def build_ml_pipeline(feature_cols):
    """
    Pipeline ML:
    1. Encode categorical features (gender)
    2. Assemble thành feature vector
    3. XGBoost Classifier (Distributed)
    """
    gender_indexer = StringIndexer(
        inputCol="gender",
        outputCol="gender_idx",
        handleInvalid="keep"  # Không crash khi gặp giá trị lạ
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep"
    )

    # XGBoost Distributed via SparkXGBClassifier
    # Nếu dùng xgboost4j-spark (recommended):
    try:
        from xgboost.spark import SparkXGBClassifier
        xgb = SparkXGBClassifier(
            features_col="features",
            label_col="churn",
            num_workers=2,          # Số Spark workers tham gia training
            use_gpu=False,          # Set True nếu có GPU
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            early_stopping_rounds=10,
            verbosity=1,
        )
    except ImportError:
        # Fallback: Dùng Spark MLlib GBTClassifier (built-in, fully distributed)
        print("⚠️  xgboost4j-spark chưa cài, dùng GBTClassifier thay thế (cũng distributed).")
        from pyspark.ml.classification import GBTClassifier
        xgb = GBTClassifier(
            featuresCol="features",
            labelCol="churn",
            maxIter=100,
            maxDepth=6,
            stepSize=0.1,
            subsamplingRate=0.8,
            featureSubsetStrategy="0.8",
        )

    return Pipeline(stages=[gender_indexer, assembler, xgb])


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    silver_path = "s3a://silver-zone/datalake/silver/eventsim"
    gold_path = "s3a://gold-zone/datalake/gold/churn_predictions"
    model_path = "s3a://gold-zone/models/churn_xgboost"

    print("🎯 [Churn Model] Đang đọc dữ liệu Silver Layer...")

    try:
        silver_df = spark.read.parquet(silver_path)

        if silver_df.rdd.isEmpty():
            print("⚠️  Không có dữ liệu để train. Hãy chạy raw_to_silver trước.")
            return

        # 1. Feature Engineering
        print("⚙️  [Churn Model] Đang tổng hợp features từ event logs...")
        user_features = build_user_features(silver_df)
        labeled_df = create_churn_label(user_features)

        print(f"   📊 Tổng số users: {labeled_df.count()}")
        churn_rate = labeled_df.filter(col("churn") == 1).count() / labeled_df.count()
        print(f"   📉 Tỷ lệ churn: {churn_rate:.1%}")

        # 2. Chuẩn bị features
        feature_cols = [
            "gender_idx",           # Encoded
            "total_songs",
            "thumbs_down",
            "thumbs_up",
            "settings_visits",
            "help_visits",
            "total_sessions",
            "days_active",
            "dislike_ratio",
            "avg_sessions_per_day",
        ]

        # 3. Train/Test split (80/20)
        train_df, test_df = labeled_df.randomSplit([0.8, 0.2], seed=42)
        print(f"   🏋️  Train: {train_df.count()} | Test: {test_df.count()}")

        # 4. Build và Train Pipeline (Distributed!)
        print("🚀 [Churn Model] Bắt đầu training phân tán trên Spark Cluster...")
        pipeline = build_ml_pipeline(feature_cols)
        model = pipeline.fit(train_df)
        print("✅ Training hoàn tất!")

        # 5. Đánh giá mô hình
        predictions = model.transform(test_df)
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        evaluator = BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        print(f"   📈 AUC Score: {auc:.4f}")

        # 6. Dự đoán tất cả users và lưu vào Gold Layer
        print(f"💾 [Churn Model] Lưu kết quả dự đoán vào: {gold_path}")
        all_predictions = model.transform(labeled_df)
        all_predictions.select(
            "userId", "churn", "probability", "prediction",
            "total_songs", "dislike_ratio", "cancel_count"
        ).write.mode("overwrite").parquet(gold_path)

        # 7. Lưu model để dùng lại (inference)
        print(f"💾 [Churn Model] Lưu model vào: {model_path}")
        model.write().overwrite().save(model_path)

        print(f"""
╔══════════════════════════════════════╗
║  ✅ CHURN MODEL TRAINING COMPLETE!   ║
║  AUC Score : {auc:.4f}               ║
║  Output    : {gold_path[:30]}...     ║
╚══════════════════════════════════════╝
        """)

    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
