from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# 1. Cấu hình mặc định cho các Task
default_args = {
    'owner': 'homura_madoka',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# 2. Khởi tạo DAG
with DAG(
    dag_id='spark_kafka_to_minio_bronze',
    default_args=default_args,
    description='Tự động đẩy dữ liệu từ Kafka vào MinIO Bronze Layer (S3-Native)',
    schedule=None, # Streaming Job chạy liên tục
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['spark', 'streaming', 'kafka', 'minio', 'bronze'],
) as dag:

    # 3. Định nghĩa Task SparkSubmit
    # QUAN TRỌNG: Không dùng 'packages' vì JARs đã pre-baked trong Spark image.
    # Khai báo lại sẽ khiến Spark cố download từ Maven → thất bại trong môi trường Docker.
    run_spark_job = SparkSubmitOperator(
        task_id='run_kafka_to_minio_streaming',
        application='/opt/airflow/src/jobs/streaming/kafka_to_minio.py',
        conn_id='spark_default',
        conf={
            'spark.master': 'spark://spark-master:7077',
            'spark.submit.deployMode': 'client',
            'spark.executor.memory': '512m',
            'spark.executor.cores': '1',
            'spark.cores.max': '1',
            'spark.driver.memory': '512m',
            'spark.app.name': 'Airflow_Kafka_To_MinIO_Streaming',
            # S3A / MinIO
            'spark.hadoop.fs.s3a.endpoint': MINIO_ENDPOINT,
            'spark.hadoop.fs.s3a.access.key': MINIO_ACCESS_KEY,
            'spark.hadoop.fs.s3a.secret.key': MINIO_SECRET_KEY,
            'spark.hadoop.fs.s3a.path.style.access': 'true',
            'spark.hadoop.fs.s3a.impl': 'org.apache.hadoop.fs.s3a.S3AFileSystem',
            'spark.hadoop.fs.s3a.connection.ssl.enabled': 'false',
            'spark.hadoop.fs.s3a.fast.upload': 'true',
            # Tắt Maven download (JARs đã có sẵn)
            'spark.jars.ivy': '/tmp/.ivy2',
            # Tắt event log (không có bucket spark-events)
            'spark.eventLog.enabled': 'false',
            # Dùng FileSystem-based checkpoint — tương thích với S3A (không cần rename atomic)
            'spark.sql.streaming.checkpointFileManagerClass': 'org.apache.spark.sql.execution.streaming.FileSystemBasedCheckpointFileManager',
            # Suppress WARN vô nghĩa: MetricsConfig + NativeCodeLoader + AdminClientConfig
            'spark.driver.extraJavaOptions': (
                '-Dlog4j.logger.org.apache.hadoop.metrics2.impl.MetricsConfig=ERROR '
                '-Dlog4j.logger.org.apache.hadoop.util.NativeCodeLoader=ERROR '
                '-Dlog4j.logger.org.apache.kafka.clients.admin.AdminClientConfig=ERROR'
            ),
            'spark.executor.extraJavaOptions': (
                '-Dlog4j.logger.org.apache.hadoop.metrics2.impl.MetricsConfig=ERROR '
                '-Dlog4j.logger.org.apache.hadoop.util.NativeCodeLoader=ERROR '
                '-Dlog4j.logger.org.apache.kafka.clients.admin.AdminClientConfig=ERROR'
            ),
        },
        name='spark_kafka_to_minio_job',
        verbose=True,
        execution_timeout=timedelta(hours=24),  # Streaming job chạy liên tục
    )

    run_spark_job
