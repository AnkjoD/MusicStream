"""
Medallion Batch Pipeline DAG
============================
Pipeline: Bronze → Silver → Gold → ML Training
Chạy mỗi 30 phút với cấu hình Spark Distributed đầy đủ.

Cluster: 1 Master + 3 Workers × 2 cores = 6 cores tổng
"""
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'homura_madoka',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

# ── Cấu hình Spark chung (dùng cho tất cả tasks) ──
SPARK_CONF = {
    # ── Cluster ──
    'spark.master': 'spark://spark-master:7077',
    'spark.submit.deployMode': 'client',

    # ── Resources - tận dụng 3 workers ──
    'spark.executor.instances': '3',        # 1 executor per worker
    'spark.executor.cores': '2',            # 2 cores per executor
    'spark.executor.memory': '1g',
    'spark.driver.memory': '1g',
    'spark.driver.cores': '1',

    # ── AQE: Adaptive Query Execution ──
    'spark.sql.adaptive.enabled': 'true',
    'spark.sql.adaptive.coalescePartitions.enabled': 'true',
    'spark.sql.adaptive.skewJoin.enabled': 'true',

    # ── Parallelism: 3 workers × 2 cores = 6 ──
    'spark.default.parallelism': '6',
    'spark.sql.shuffle.partitions': '6',

    # ── Serialization ──
    'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
    'spark.kryoserializer.buffer.max': '256m',

    # ── MinIO (S3A) ──
    'spark.hadoop.fs.s3a.endpoint': 'http://minio:9000',
    'spark.hadoop.fs.s3a.access.key': 'homura_madoka',
    'spark.hadoop.fs.s3a.secret.key': 'homura123',
    'spark.hadoop.fs.s3a.path.style.access': 'true',
    'spark.hadoop.fs.s3a.impl': 'org.apache.hadoop.fs.s3a.S3AFileSystem',
    'spark.hadoop.fs.s3a.fast.upload': 'true',

    # ── Event Log để History Server có thể đọc ──
    'spark.eventLog.enabled': 'true',
    'spark.eventLog.dir': 's3a://bronze-zone/spark-events',
}

# Tắt việc download JAR từ Maven (JARs đã pre-baked trong Spark image)
SPARK_CONF_EXTRA = {
    'spark.jars.ivy': '/tmp/.ivy2',
}

with DAG(
    dag_id='spark_medallion_batch_pipeline',
    default_args=default_args,
    description='Pipeline phân tán: Bronze→Silver→Gold→ML Training (3 Spark Workers)',
    schedule='*/30 * * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['spark', 'distributed', 'medallion', 'minio'],
) as dag:

    # TASK 1: Bronze → Silver (Normalize + Deduplicate)
    raw_to_silver = SparkSubmitOperator(
        task_id='bronze_to_silver',
        application='/opt/airflow/src/jobs/batch/raw_to_silver.py',
        conn_id='spark_default',
        conf={**SPARK_CONF, **SPARK_CONF_EXTRA},
        name='spark_bronze_to_silver',
        execution_timeout=timedelta(minutes=20),
    )

    # TASK 2: Silver → Gold (Song Rankings Report)
    silver_to_gold = SparkSubmitOperator(
        task_id='silver_to_gold_rankings',
        application='/opt/airflow/src/jobs/batch/song_count_report.py',
        conn_id='spark_default',
        conf={**SPARK_CONF, **SPARK_CONF_EXTRA},
        name='spark_silver_to_gold',
        execution_timeout=timedelta(minutes=15),
    )

    # TASK 3: Train Recommendation Model (ALS)
    train_recommendation = SparkSubmitOperator(
        task_id='train_als_recommendation',
        application='/opt/airflow/src/jobs/batch/train_recommendation.py',
        conn_id='spark_default',
        conf={
            **SPARK_CONF,
            **SPARK_CONF_EXTRA,
            'spark.executor.memory': '2g',
            'spark.driver.memory': '2g',
        },
        name='spark_train_recommendation',
        execution_timeout=timedelta(minutes=30),
    )

    # TASK 4: Churn Prediction (GBT)
    train_churn = SparkSubmitOperator(
        task_id='train_churn_prediction',
        application='/opt/airflow/src/jobs/batch/churn_prediction.py',
        conn_id='spark_default',
        conf={
            **SPARK_CONF,
            **SPARK_CONF_EXTRA,
            'spark.executor.memory': '2g',
            'spark.driver.memory': '2g',
        },
        name='spark_train_churn',
        execution_timeout=timedelta(minutes=30),
    )

    # ── DAG Flow: Sequential với dependency rõ ràng ──
    # Bronze→Silver phải xong trước khi làm Gold
    # Gold phải xong trước khi train (cần clean data)
    raw_to_silver >> silver_to_gold >> [train_recommendation, train_churn]
