"""
ETL Fast Pipeline DAG
=====================
Pipeline: Bronze → Silver → Gold (Rankings)
Chạy mỗi 15 phút — chỉ ETL thuần túy, không training AI.

Mục tiêu: Dashboard luôn có dữ liệu bảng xếp hạng cập nhật gần real-time.
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
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# ── Cấu hình Spark chung ──
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

with DAG(
    dag_id='etl_fast_pipeline',
    default_args=default_args,
    description='ETL nhanh: Bronze→Silver→Gold (15 phút/lần, không train AI)',
    schedule='*/15 * * * *',  # Chạy mỗi 15 phút
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['spark', 'etl', 'medallion', 'fast'],
) as dag:

    # TASK 1: Bronze → Silver (Normalize + Deduplicate)
    raw_to_silver = SparkSubmitOperator(
        task_id='bronze_to_silver',
        application='/opt/airflow/src/jobs/batch/raw_to_silver.py',
        conn_id='spark_default',
        conf=SPARK_CONF,
        name='spark_bronze_to_silver',
        execution_timeout=timedelta(minutes=10),
    )

    # TASK 2: Silver → Gold (Song Rankings Report)
    silver_to_gold = SparkSubmitOperator(
        task_id='silver_to_gold_rankings',
        application='/opt/airflow/src/jobs/batch/song_count_report.py',
        conn_id='spark_default',
        conf=SPARK_CONF,
        name='spark_silver_to_gold',
        execution_timeout=timedelta(minutes=10),
    )

    # ── Flow: ETL tuần tự ──
    raw_to_silver >> silver_to_gold
