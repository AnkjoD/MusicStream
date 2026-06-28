from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# Cấu hình mặc định áp dụng cho tất cả task trong DAG này
default_args = {
    'owner': 'homura_madoka',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Khởi tạo luồng DAG
with DAG(
    dag_id='spark_kafka_to_minio_bronze',
    default_args=default_args,
    description='Tự động đẩy dữ liệu từ Kafka vào MinIO Bronze Layer (S3-Native)',
    schedule=None, # Set schedule=None vì đây là luồng Streaming chạy liên tục 24/7, không chạy theo lịch cố định
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['spark', 'streaming', 'kafka', 'minio', 'bronze'],
) as dag:

    # Định nghĩa Task submit job Spark
    # LƯU Ý LỚN: Tuyệt đối không dùng 'packages' ở đây để tải thêm thư viện.
    # Vì toàn bộ các file JAR cần thiết đã được cài sẵn (pre-baked) trong image Spark rồi.
    # Nếu khai báo packages, Spark sẽ cố lên Maven tải lại và sẽ lỗi vì môi trường Docker không có mạng ngoài hoặc tải rất chậm.
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
            # Kết nối MinIO qua giao thức S3A
            'spark.hadoop.fs.s3a.endpoint': MINIO_ENDPOINT,
            'spark.hadoop.fs.s3a.access.key': MINIO_ACCESS_KEY,
            'spark.hadoop.fs.s3a.secret.key': MINIO_SECRET_KEY,
            'spark.hadoop.fs.s3a.path.style.access': 'true',
            'spark.hadoop.fs.s3a.impl': 'org.apache.hadoop.fs.s3a.S3AFileSystem',
            'spark.hadoop.fs.s3a.connection.ssl.enabled': 'false',
            'spark.hadoop.fs.s3a.fast.upload': 'true',
            # Tránh tải linh tinh từ Maven (do JARs đã được cài sẵn trong image rồi)
            'spark.jars.ivy': '/tmp/.ivy2',
            # Tắt ghi log sự kiện (vì cụm Spark local không có sẵn bucket spark-events)
            'spark.eventLog.enabled': 'false',
            # Dùng cơ chế checkpoint dựa trên FileSystem để tương thích tốt với MinIO/S3 (tránh lỗi rename atomic do S3 không hỗ trợ thực sự)
            'spark.sql.streaming.checkpointFileManagerClass': 'org.apache.spark.sql.execution.streaming.FileSystemBasedCheckpointFileManager',
            # Tắt mấy cảnh báo vô nghĩa ở console cho đỡ rối mắt (của MetricsConfig, NativeCodeLoader, AdminClientConfig)
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
