"""
Xây dựng mô hình dự đoán Churn sử dụng XGBoost phân tán trên Spark.

Nhiệm vụ của mô hình:
    Dự đoán xem người dùng nào đang xài gói trả phí (PAID) có xu hướng hạ cấp xuống gói miễn phí (FREE)
    để mình kịp thời tung ra các chương trình khuyến mãi/ưu đãi giữ chân họ.

    Các đặc trưng (Features) đầu vào gom từ tầng Silver:
        - Tần suất nghe nhạc (tổng số bài hát đã nghe).
        - Mức độ hài lòng (tỷ lệ Thumbs Up / Thumbs Down).
        - Tần suất truy cập trang cấu hình Settings/Help (dấu hiệu muốn hủy gói).
        - Hoạt động tương tác (số session, số ngày active, tần suất truy cập trung bình).
        - Thông tin cơ bản: Giới tính, OS, Browser đang dùng.

    Đầu ra (Output):
        - Xác suất người dùng đó sẽ churn (từ 0 đến 1), ghi thẳng vào Gold Layer.

Cơ chế phân tán với SparkXGB:
    - Train mô hình trực tiếp trên Spark Cluster qua thư viện xgboost4j-spark.
    - Dữ liệu được xé nhỏ ra các Worker để train song song, Driver chỉ nhận kết quả cuối, tránh nghẽn cổ chai.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, datediff,
    max as spark_max, min as spark_min, lit, when, to_date
)
from pyspark.sql.utils import AnalysisException
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import os
import sys


def create_spark_session():
    """Khởi động SparkSession, cấu hình tối ưu nhất cho việc chạy học máy phân tán."""
    return (
        SparkSession.builder
        .appName("ChurnPrediction_XGBoost_Distributed")
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("MINIO_ENDPOINT", "http://minio:9000"))
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY", "minioadmin"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # Chia đều dữ liệu thành 200 partition để Spark xử lý song song mượt mà hơn
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def build_user_features(silver_df, spark):
    """
    Gom các event log lại theo từng user để tạo tính năng (feature engineering).
    Mỗi user sẽ được tổng hợp thành đúng 1 dòng dữ liệu chứa toàn bộ hành vi của họ.
    """
    # Lấy ngày cuối cùng có dữ liệu trong dataset để làm mốc tính thời gian active.
    # Dùng câu lệnh SQL của Spark chạy trực tiếp dưới JVM cho nhanh, tránh gọi đi gọi lại qua Python Driver.
    silver_df.createOrReplaceTempView("silver_events")
    max_date_row = spark.sql(
        "SELECT MAX(event_time) AS max_dt FROM silver_events"
    ).collect()
    max_date = max_date_row[0]["max_dt"] if max_date_row and max_date_row[0]["max_dt"] else None

    # Bắt đầu gom nhóm và tính toán các chỉ số hành vi
    user_features = silver_df.groupBy("userId", "level", "gender").agg(
        # Đếm tổng số bài hát đã nghe hoàn chỉnh
        count(when(col("page") == "NextSong", True)).alias("total_songs"),
        # Đếm số lần bấm Thumbs Down (thể hiện không thích bài hát)
        count(when(col("page") == "Thumbs Down", True)).alias("thumbs_down"),
        # Đếm số lần bấm Thumbs Up (yêu thích bài hát)
        count(when(col("page") == "Thumbs Up", True)).alias("thumbs_up"),
        # Số lần truy cập trang Settings (đang phân vân cấu hình hoặc tìm nút hủy gói)
        count(when(col("page") == "Settings", True)).alias("settings_visits"),
        # Số lần mò vào trang trợ giúp Help
        count(when(col("page") == "Help", True)).alias("help_visits"),
        # Sự kiện bấm Downgrade (đây là tín hiệu Churn rõ nhất của bộ giả lập eventsim)
        count(when(col("page") == "Downgrade", True)).alias("downgrade_count"),
        # Đếm thêm sự kiện Cancel thật phòng khi cấu trúc log sau này thay đổi
        count(when(col("page") == "Cancellation Confirmation", True)).alias("cancel_count"),
        # Số session sử dụng ứng dụng
        count("sessionId").alias("total_sessions"),
        # Số ngày hoạt động thực tế
        (datediff(
            to_date(lit(max_date)),
            to_date(spark_min("event_time"))
        ) + 1).alias("days_active"),
    )

    # Tính tỷ lệ không thích nhạc trên tổng số bài đã nghe
    user_features = user_features.withColumn(
        "dislike_ratio",
        when(col("total_songs") > 0,
             col("thumbs_down") / col("total_songs")).otherwise(0.0)
    ).withColumn(
        # Tính số session trung bình mỗi ngày hoạt động
        "avg_sessions_per_day",
        when(col("days_active") > 0,
             col("total_sessions") / col("days_active")).otherwise(0.0)
    )

    return user_features


def create_churn_label(user_features):
    """
    Tạo nhãn Churn (0 hoặc 1).
    Logic chuẩn: User bị coi là Churn nếu họ chủ động bấm Downgrade hoặc Cancel gói.
    Không được dùng trực tiếp 'level=free' để làm nhãn vì nhiều người dùng chỉ dùng free từ đầu 
    (chưa bao giờ trả phí thì không gọi là churn được, nếu đưa vào sẽ bị hiện tượng rò rỉ nhãn - label leakage).
    """
    return user_features.withColumn(
        "churn",
        when(
            (col("cancel_count") > 0) | (col("downgrade_count") > 0),
            lit(1)
        ).otherwise(lit(0))
    )


def build_ml_pipeline(feature_cols):
    """
    Thiết lập luồng xử lý (Pipeline) cho học máy:
    1. Mã hóa giới tính (gender) từ text sang số index.
    2. Gom tất cả các cột đặc trưng lại thành 1 cột vector duy nhất (features).
    3. Đưa vào mô hình phân loại XGBoost chạy phân tán.
    """
    gender_indexer = StringIndexer(
        inputCol="gender",
        outputCol="gender_idx",
        handleInvalid="keep"  # Để 'keep' để không bị crash nếu lỡ sau này có thêm giới tính khác lạ
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep"
    )

    # Ưu tiên dùng XGBoost phân tán nếu môi trường có sẵn thư viện
    try:
        from xgboost.spark import SparkXGBClassifier
        xgb = SparkXGBClassifier(
            features_col="features",
            label_col="churn",
            num_workers=2,          # Phân bổ cho 2 worker chạy song song
            use_gpu=False,          # Chạy CPU cho lành, đổi thành True nếu worker có gắn card rời
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
        # Backup plan: Nếu chưa cài xgboost cho Spark, dùng GBTClassifier có sẵn của Spark MLlib (cũng chạy phân tán tốt)
        print("⚠️ Không import được xgboost.spark, tự động chuyển sang mô hình dự phòng GBTClassifier.")
        from pyspark.ml.classification import GBTClassifier
        xgb = GBTClassifier(
            featuresCol="features",
            labelCol="churn",
            maxIter=10,        # Giảm số vòng lặp để tránh quá tải bộ nhớ
            maxDepth=4,        # Giới hạn độ sâu của cây để tối ưu hiệu năng
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
        # Xóa recursiveFileLookup để Spark nhận diện cột partition 'page'
        silver_df = spark.read.parquet(silver_path)

        # 1. Feature Engineering
        print("⚙️  [Churn Model] Đang tổng hợp features từ event logs...")
        user_features = build_user_features(silver_df, spark)
        labeled_df = create_churn_label(user_features)

        # CHÚ Ý: Bỏ các lệnh count() trung gian để tối ưu.
        # Việc gọi count() không cần thiết sẽ ép Spark phải tính toán lại toàn bộ đồ thị, dễ gây lỗi OOM.

        # 2. Chuẩn bị features
        # LƯU Ý: Tuyệt đối không nhét cột 'level' hiện tại vào bộ đặc trưng đầu vào. 
        # Cột level này tương quan trực tiếp với việc bấm nút hạ cấp (Downgrade) dễ gây rò rỉ thông tin trước cho mô hình (data leakage).
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

        # 3. Chia tập dữ liệu thành Train (80%) và Test (20%) để đánh giá khách quan
        train_df, test_df = labeled_df.randomSplit([0.8, 0.2], seed=42)
        # Không gọi count() ở đây luôn để tránh tốn RAM vô ích

        # 4. Tạo pipeline và khởi động quá trình Train phân tán trên Spark cluster
        print("🚀 [Churn Model] Bắt đầu training phân tán trên Spark Cluster...")
        pipeline = build_ml_pipeline(feature_cols)
        model = pipeline.fit(train_df)
        print("✅ Training hoàn tất!")

        # 5. Đánh giá chất lượng mô hình trên tập Test
        predictions = model.transform(test_df)
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        evaluator = BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        print(f"   📈 AUC Score: {auc:.4f}")

        # 6. Chạy dự báo cho toàn bộ danh sách user rồi lưu kết quả vào Gold Layer
        print(f"💾 [Churn Model] Lưu kết quả dự đoán vào: {gold_path}")
        all_predictions = model.transform(labeled_df)
        all_predictions.select(
            "userId", "churn", "probability", "prediction",
            "total_songs", "dislike_ratio", "cancel_count"
        ).write.mode("overwrite").parquet(gold_path)

        # 7. Lưu mô hình (export model) để phục vụ cho các đợt inference sau này
        print(f"💾 [Churn Model] Lưu model vào: {model_path}")
        model.write().overwrite().save(model_path)

        print(f"""
╔══════════════════════════════════════╗
║  ✅ CHURN MODEL TRAINING COMPLETE!   ║
║  AUC Score : {auc:.4f}               ║
║  Output    : {gold_path[:30]}...     ║
╚══════════════════════════════════════╝
        """)

    except AnalysisException as e:
        print(f"⚠️  [Churn Model] Silver layer chưa có dữ liệu (hoặc đang rỗng). Vui lòng chờ luồng ETL chạy xong. Chi tiết: {e}")
        return
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()