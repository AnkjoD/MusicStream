"""
Luồng train Deep Learning (NCF và LSTM Autoencoder)
Chạy vào lúc 3h sáng Chủ Nhật hàng tuần. Tách riêng ra để tránh đụng độ tài nguyên với các luồng ML truyền thống (ALS, XGBoost).

Các bước chạy:
    1. Trích xuất dữ liệu (export_training_data)
    2. Train model gợi ý NCF (train_ncf)
    3. Train model phát hiện bất thường LSTM AE (train_lstm_ae)

Lưu ý quan trọng:
    - Bắt buộc phải chạy tuần tự (sequential) từng task một để tránh việc chiếm dụng tài nguyên GPU quá tải
      gây lỗi tràn bộ nhớ (VRAM) hoặc crash hệ thống.
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT",   "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY",  "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY",  "minioadmin")
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Trỏ đến môi trường Python có cài sẵn PyTorch (venv riêng hoặc python hệ thống)
PYTHON = os.getenv("DL_PYTHON", "/opt/dl_venv/bin/python")

default_args = {
    "owner":           "minioadmin",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries":         1,
    "retry_delay":     timedelta(minutes=10),
}

# Gom các biến môi trường cần thiết để truyền vào script training
TRAIN_ENV = (
    f"MINIO_ENDPOINT={MINIO_ENDPOINT} "
    f"MINIO_ACCESS_KEY={MINIO_ACCESS_KEY} "
    f"MINIO_SECRET_KEY={MINIO_SECRET_KEY} "
    f"MLFLOW_TRACKING_URI={MLFLOW_URI} "
)

with DAG(
    dag_id="dl_training_pipeline",
    default_args=default_args,
    description="DL Training: NCF Recommendation + LSTM AE Anomaly Detection (weekly)",
    schedule="0 3 * * 0",   # Sunday 3:00 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["dl", "ncf", "lstm", "anomaly", "pytorch"],
) as dag:

    # Bước 1: Trích xuất và chuẩn bị dữ liệu từ Silver sang định dạng PyTorch cần
    export_data = BashOperator(
        task_id="export_training_data",
        bash_command=(
            f"{TRAIN_ENV} "
            f"{PYTHON} /opt/airflow/src/jobs/dl/data_prep/export_training_data.py"
        ),
        execution_timeout=timedelta(minutes=20),
    )

    # Bước 2: Bắt đầu train model gợi ý bài hát NCF (có cắn GPU)
    train_ncf = BashOperator(
        task_id="train_ncf",
        bash_command=(
            f"{TRAIN_ENV} "
            f"NCF_EMB_DIM=64 NCF_EPOCHS=20 NCF_BATCH=2048 "
            f"{PYTHON} /opt/airflow/src/jobs/dl/models/train_ncf.py"
        ),
        execution_timeout=timedelta(hours=2),
    )

    # Bước 3: Train model LSTM Autoencoder để phát hiện bất thường hành vi (cũng cắn GPU)
    train_ae = BashOperator(
        task_id="train_lstm_ae",
        bash_command=(
            f"{TRAIN_ENV} "
            f"AE_HIDDEN=64 AE_LATENT=16 AE_EPOCHS=30 AE_BATCH=256 "
            f"{PYTHON} /opt/airflow/src/jobs/dl/models/train_lstm_ae.py"
        ),
        execution_timeout=timedelta(hours=2),
    )

    # Thứ tự chạy: Export dữ liệu xong -> Train NCF -> Train LSTM AE
    export_data >> train_ncf >> train_ae
