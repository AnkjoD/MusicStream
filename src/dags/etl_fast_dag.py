"""
Luồng ETL nhanh: Chạy dồn dập 15 phút một lần để đẩy data từ Bronze lên Silver, 
sau đó gom Gold (xếp hạng nhạc) rồi nhảy qua ML Monitoring để check drift ngay.
"""
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import os

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY",  "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY",  "minioadmin")
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
PYTHON           = os.getenv("DL_PYTHON", "/opt/dl_venv/bin/python")

default_args = {
    'owner': 'homura_madoka',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

SPARK_CONF = {
    'spark.master': 'spark://spark-master:7077',
    'spark.submit.deployMode': 'client',
    'spark.executor.instances': '1',
    'spark.executor.cores': '2',
    'spark.cores.max': '2',
    'spark.executor.memory': '2g',
    'spark.driver.memory': '1g',
    'spark.driver.cores': '1',
    'spark.sql.adaptive.enabled': 'true',
    'spark.sql.adaptive.coalescePartitions.enabled': 'true',
    'spark.sql.adaptive.skewJoin.enabled': 'true',
    'spark.default.parallelism': '4',
    'spark.sql.shuffle.partitions': '4',
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

# Chạy ML Monitoring mỗi 15 phút thì hơi spam report nên mình chỉ cho chạy 1 lần lúc đầu giờ thôi.
# Hàm Python dưới này sẽ check xem có đúng đầu giờ không để rẽ nhánh.
def should_run_monitoring(**context):
    """
    Check xem có thuộc 15 phút đầu tiên của giờ không để trigger monitoring.
    Làm vậy cho đỡ spam báo cáo.
    """
    execution_minute = context["logical_date"].minute
    if execution_minute < 15:   # Chỉ chạy lần đầu của mỗi giờ (0-14 phút)
        return "run_ml_monitoring"
    return "skip_monitoring"


with DAG(
    dag_id='etl_fast_pipeline',
    default_args=default_args,
    description='ETL nhanh: Bronze→Silver→Gold→Monitoring (15 phút/lần)',
    schedule='*/15 * * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['spark', 'etl', 'medallion', 'fast', 'monitoring'],
) as dag:

    # Bước 1: Hốt dữ liệu thô từ Bronze sang Silver
    raw_to_silver = SparkSubmitOperator(
        task_id='bronze_to_silver',
        application='/opt/airflow/src/jobs/batch/raw_to_silver.py',
        conn_id='spark_default',
        conf=SPARK_CONF,
        name='spark_bronze_to_silver',
        execution_timeout=timedelta(minutes=10),
    )

    # Bước 2: Từ Silver làm sạch, gom tiếp lên Gold để xếp hạng bài hát
    silver_to_gold = SparkSubmitOperator(
        task_id='silver_to_gold_rankings',
        application='/opt/airflow/src/jobs/batch/song_count_report.py',
        conn_id='spark_default',
        conf=SPARK_CONF,
        name='spark_silver_to_gold',
        execution_timeout=timedelta(minutes=10),
    )

    # Bước 3: Rẽ nhánh - check xem có cần chạy ML Monitoring đợt này không
    check_monitoring = BranchPythonOperator(
        task_id='check_should_monitor',
        python_callable=should_run_monitoring,
        provide_context=True,
    )

    # Nhánh 4a: Chạy script ML Monitoring
    run_monitoring = BashOperator(
        task_id='run_ml_monitoring',
        bash_command=(
            f"MINIO_ENDPOINT={MINIO_ENDPOINT} "
            f"MINIO_ACCESS_KEY={MINIO_ACCESS_KEY} "
            f"MINIO_SECRET_KEY={MINIO_SECRET_KEY} "
            f"MLFLOW_TRACKING_URI={MLFLOW_URI} "
            f"DRIFT_THRESHOLD=0.3 "
            f"{PYTHON} /opt/airflow/src/jobs/monitoring/ml_monitoring.py"
        ),
        execution_timeout=timedelta(minutes=10),
        # Cho phép bỏ qua lỗi nếu monitoring tèo, vì đây chỉ là giám sát chứ không phải luồng chính.
        retries=0,
    )

    # Nhánh 4b: Không làm gì cả (skip)
    skip_monitoring = BashOperator(
        task_id='skip_monitoring',
        bash_command='echo "Skipping monitoring this run"',
    )

    # Định nghĩa luồng chạy của các task
    raw_to_silver >> silver_to_gold >> check_monitoring
    check_monitoring >> [run_monitoring, skip_monitoring]
