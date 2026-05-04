from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import json
import logging

# Cấu hình mặc định
default_args = {
    'owner': 'homura_madoka',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def submit_recommendation_job():
    """Gửi yêu cầu chạy Spark Recommendation Job tới Spark Master qua REST API."""
    url = "http://spark-master:6066/v1/submissions/create"
    
    payload = {
        "action": "CreateSubmissionRequest",
        "appArgs": [],
        "appResource": "file:/opt/spark/jobs/batch/train_recommendation.py",
        "clientSparkVersion": "3.5.0",
        "mainClass": "org.apache.spark.deploy.PythonRunner",
        "environmentVariables": {
            "SPARK_ENV_LOADED": "1"
        },
        "sparkProperties": {
            "spark.driver.supervise": "false",
            "spark.app.name": "MusicRecommendationTraining",
            "spark.submit.deployMode": "cluster",
            "spark.master": "spark://spark-master:7077",
            "spark.executor.memory": "1g",
            "spark.driver.memory": "1g",
            "spark.jars.packages": "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"
        }
    }
    
    logging.info(f"Gửi yêu cầu tới Spark Master: {url}")
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        logging.info(f"Phản hồi từ Spark Master: {json.dumps(result, indent=2)}")
        
        if result.get("success"):
            logging.info(f"Job đã được gửi thành công! Submission ID: {result.get('submissionId')}")
        else:
            raise Exception(f"Spark Master báo lỗi: {result.get('message')}")
            
    except Exception as e:
        logging.error(f"Lỗi khi gọi Spark REST API: {str(e)}")
        raise

# Định nghĩa DAG
with DAG(
    dag_id='music_recommendation_training',
    default_args=default_args,
    description='Huấn luyện AI gợi ý bài hát định kỳ (via REST API)',
    schedule=None, # Bạn có thể đổi thành '@daily' để chạy hàng ngày
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['spark', 'ml', 'recommendation'],
) as dag:

    train_model = PythonOperator(
        task_id='submit_ml_training_via_rest',
        python_callable=submit_recommendation_job,
    )
