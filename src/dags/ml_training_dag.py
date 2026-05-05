"""
ML Training Pipeline DAG
=========================
Pipeline: Silver → ML Models (ALS Recommendation + Churn Prediction)
Chạy 1 lần/ngày lúc 2:00 sáng — tác vụ tốn thời gian nhưng không urgent.

Chiến lược:
- Đọc dữ liệu Silver ĐÃ SẠCH (do etl_fast_pipeline xử lý liên tục) để train.
- Kết quả (Gold/recommendations + Gold/churn_predictions) được lưu vào MinIO.
- Dashboard tự đọc kết quả từ MinIO, không cần real-time.
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

# ALS cần tắt AQE để tránh BroadcastExchangeExec → OOM → EOFException
SPARK_CONF_ALS = {
    **SPARK_CONF_BASE,
    'spark.sql.adaptive.enabled': 'false',
    'spark.sql.adaptive.coalescePartitions.enabled': 'false',
    'spark.sql.autoBroadcastJoinThreshold': '-1',
}

# Churn dùng chung cấu hình với ALS (tắt AQE) để tránh OOM
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
    max_active_runs=1,  # Không cho 2 training jobs chạy đồng thời
    tags=['spark', 'ml', 'training', 'als', 'churn'],
) as dag:

    # TASK 1: Train ALS Recommendation Model
    # Đọc Silver → Học ma trận user×song → Lưu Top-10 recommendations vào Gold
    train_recommendation = SparkSubmitOperator(
        task_id='train_als_recommendation',
        application='/opt/airflow/src/jobs/batch/train_recommendation.py',
        conn_id='spark_default',
        conf=SPARK_CONF_ALS,
        name='spark_train_recommendation',
        execution_timeout=timedelta(minutes=40),
    )

    # TASK 2: Train Churn Prediction Model
    # Đọc Silver → Học hành vi nghe nhạc → Dự đoán khả năng churn vào Gold
    train_churn = SparkSubmitOperator(
        task_id='train_churn_prediction',
        application='/opt/airflow/src/jobs/batch/churn_prediction.py',
        conn_id='spark_default',
        conf=SPARK_CONF_CHURN,
        name='spark_train_churn',
        execution_timeout=timedelta(minutes=40),
    )

    # ── Flow: ALS xong rồi mới train Churn để không OOM ──
    # (2 Spark jobs chạy song song = 1024MB RAM → vượt giới hạn 1200MB của worker)
    train_recommendation >> train_churn
