"""
Luồng train các model ML (ALS gợi ý nhạc và XGBoost dự đoán khách rời bỏ - Churn)
Chạy tự động lúc 2h sáng mỗi ngày — giờ này ít người dùng nên tha hồ train mà không lo nghẽn mạng.

Chiến lược thực hiện:
- Hốt dữ liệu tầng Silver đã được dọn dẹp sạch sẽ (nhờ luồng etl_fast chạy trước đó).
- Output (file gợi ý và tỉ lệ dự đoán churn) sẽ ném trực tiếp lên MinIO (Gold Layer).
- Trang Dashboard streamlit sẽ tự động quét file này để hiển thị, không cần cập nhật real-time.
"""
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

default_args = {
    'owner': 'homura_madoka',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

SPARK_CONF_BASE = {
    'spark.master': 'spark://spark-master:7077',
    'spark.submit.deployMode': 'client',
    'spark.executor.instances': '1',
    'spark.executor.cores': '2',
    'spark.cores.max': '2',
    'spark.executor.memory': '2g',
    'spark.driver.memory': '1g',
    'spark.driver.cores': '1',
    'spark.default.parallelism': '1',
    'spark.sql.shuffle.partitions': '1',
    'spark.hadoop.fs.s3a.endpoint': MINIO_ENDPOINT,
    'spark.hadoop.fs.s3a.access.key': MINIO_ACCESS_KEY,
    'spark.hadoop.fs.s3a.secret.key': MINIO_SECRET_KEY,
    'spark.hadoop.fs.s3a.path.style.access': 'true',
    'spark.hadoop.fs.s3a.impl': 'org.apache.hadoop.fs.s3a.S3AFileSystem',
    'spark.hadoop.fs.s3a.fast.upload': 'true',
    'spark.eventLog.enabled': 'false',
    'spark.sql.debug.maxToStringFields': '50',
    'spark.jars.ivy': '/tmp/.ivy2',
    'spark.driver.extraJavaOptions': (
        '-Dlog4j.logger.org.apache.hadoop.metrics2.impl.MetricsConfig=ERROR '
        '-Dlog4j.logger.org.apache.hadoop.util.NativeCodeLoader=ERROR'
    ),
    'spark.executor.extraJavaOptions': (
        '-Dlog4j.logger.org.apache.hadoop.metrics2.impl.MetricsConfig=ERROR '
        '-Dlog4j.logger.org.apache.hadoop.util.NativeCodeLoader=ERROR'
    ),
}

# Lưu ý quan trọng cho ALS: Phải tắt AQE (Adaptive Query Execution). 
# Nếu bật, Spark sẽ tự động broadcast các bảng trung gian gây tràn bộ nhớ (OOM) và lỗi EOFException.
SPARK_CONF_ALS = {
    **SPARK_CONF_BASE,
    'spark.sql.adaptive.enabled': 'false',
    'spark.sql.adaptive.coalescePartitions.enabled': 'false',
    'spark.sql.autoBroadcastJoinThreshold': '-1',
}

# Model dự đoán Churn cũng dùng chung cấu hình tắt AQE như ALS để giữ an toàn cho bộ nhớ worker.
SPARK_CONF_CHURN = {
    **SPARK_CONF_ALS
}

with DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='ML Training hàng ngày: ALS Recommendation + Churn Prediction (2:00 AM)',
    schedule='0 2 * * *',  # Chạy lúc 2:00 AM mỗi ngày
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,  # Chỉ cho phép chạy đúng 1 instance tại một thời điểm, tránh việc chạy đè lên nhau gây sập cụm Spark.
    tags=['spark', 'ml', 'training', 'als', 'churn'],
) as dag:

    # Bước 1: Train model gợi ý nhạc ALS
    # Đọc dữ liệu Silver -> Phân tích ma trận tương tác User x Song -> Xuất Top 10 bài hát gợi ý lên Gold Layer
    train_recommendation = SparkSubmitOperator(
        task_id='train_als_recommendation',
        application='/opt/airflow/src/jobs/batch/train_recommendation.py',
        conn_id='spark_default',
        conf=SPARK_CONF_ALS,
        name='spark_train_recommendation',
        execution_timeout=timedelta(minutes=40),
    )

    # Bước 2: Train model dự đoán Churn (Khách sắp hủy gói Premium)
    # Phân tích hành vi tương tác để phát hiện ai chuẩn bị chuyển từ gói PAID sang FREE, lưu kết quả lên Gold
    train_churn = SparkSubmitOperator(
        task_id='train_churn_prediction',
        application='/opt/airflow/src/jobs/batch/churn_prediction.py',
        conn_id='spark_default',
        conf=SPARK_CONF_CHURN,
        name='spark_train_churn',
        execution_timeout=timedelta(minutes=40),
    )

    # ── Thứ tự: Phải chạy xong ALS rồi mới kích hoạt Churn để tránh quá tải bộ nhớ (OOM) ──
    # (Tránh chạy song song 2 job Spark đồng thời trên cùng một worker vì sẽ gây tràn RAM của container)
    train_recommendation >> train_churn
