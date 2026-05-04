from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

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
            'spark.executor.memory': '1G',
            'spark.executor.cores': '1',
            'spark.driver.memory': '1G',
            'spark.app.name': 'Airflow_Kafka_To_MinIO_Streaming',
            # S3A / MinIO
            'spark.hadoop.fs.s3a.endpoint': 'http://minio:9000',
            'spark.hadoop.fs.s3a.access.key': 'homura_madoka',
            'spark.hadoop.fs.s3a.secret.key': 'homura123',
            'spark.hadoop.fs.s3a.path.style.access': 'true',
            'spark.hadoop.fs.s3a.impl': 'org.apache.hadoop.fs.s3a.S3AFileSystem',
            'spark.hadoop.fs.s3a.connection.ssl.enabled': 'false',
            'spark.hadoop.fs.s3a.fast.upload': 'true',
            # Tắt Maven download (JARs đã có sẵn)
            'spark.jars.ivy': '/tmp/.ivy2',
            # Event log
            'spark.eventLog.enabled': 'true',
            'spark.eventLog.dir': 's3a://bronze-zone/spark-events',
        },
        name='spark_kafka_to_minio_job',
        verbose=True,
        execution_timeout=timedelta(hours=24),  # Streaming job chạy liên tục
    )

    run_spark_job
