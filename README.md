# 🎵 Streamlify: Nền tảng Phân tích Streaming Nhạc Thời gian Thực

**Streamlify** là một nền tảng Data & ML Engineering hiện đại, xử lý dữ liệu streaming thời gian thực dựa trên hành vi người dùng (tương tự Spotify). Dự án áp dụng kiến trúc **Medallion Architecture** (Bronze → Silver → Gold) kết hợp với **Deep Learning** để xây dựng luồng dữ liệu hoàn chỉnh — từ thu thập sự kiện thô đến serving ML model qua REST API.

---

## 🏗️ Kiến trúc Hệ thống

```
Eventsim (1,000 users)
        │
        ▼ Kafka (KRaft)
        │
        ▼ Spark Structured Streaming
  ┌─────────────────────┐
  │   Bronze Zone       │  ← Raw events, partitioned by page
  │   (MinIO S3)        │
  └─────────┬───────────┘
            │ Airflow (every 15 min)
            ▼ Spark Batch ETL
  ┌─────────────────────┐
  │   Silver Zone       │  ← Cleaned, deduplicated, enriched
  │   (MinIO S3)        │
  └─────────┬───────────┘
            │
     ┌──────┼──────────────────────────┐
     ▼      ▼                          ▼
  Song    ALS + XGBoost            NCF + LSTM AE
Rankings  (Spark MLlib)            (PyTorch, GPU)
  Gold      Gold                      Gold
     │         │                        │
     └──────────────────┬───────────────┘
                        ▼
               FastAPI Serving Layer
               /recommend/{userId}
               /anomaly/realtime
                        │
                        ▼
              Streamlit Dashboard
```

---

## 🛠️ Công nghệ Sử dụng

| Thành phần | Công nghệ | Chi tiết |
|---|---|---|
| **Data Source** | `Eventsim` | Giả lập hành vi người dùng Spotify thời gian thực |
| **Message Broker** | `Apache Kafka 3.7` | KRaft mode, không cần Zookeeper |
| **Stream Processing** | `Apache Spark Streaming` | Kafka → Bronze (Structured Streaming) |
| **Batch Processing** | `Apache Spark 3.5` | ETL, Spark MLlib, distributed training |
| **Data Lake** | `MinIO` | S3-compatible, medallion architecture |
| **Orchestration** | `Apache Airflow 2.10` | CeleryExecutor, 4 DAGs |
| **Classical ML** | `Spark MLlib ALS`, `XGBoost` | Recommendation + Churn prediction |
| **Deep Learning** | `PyTorch` | NCF (NeuMF) + LSTM Autoencoder |
| **ML Serving** | `FastAPI` | REST API, real-time inference |
| **Experiment Tracking** | `MLflow` | Metrics, params, artifacts |
| **ML Monitoring** | `scipy KS-test` | Data drift detection tự động |
| **Dashboard** | `Streamlit` | Analytics + Monitoring UI |
| **Infrastructure** | `Docker Compose` | 12+ services containerized |

---

## 🤖 ML Models

### Neural Collaborative Filtering (NCF / NeuMF)
- **Kiến trúc**: GMF branch + MLP branch (He et al., WWW 2017)
- **Training**: Implicit feedback với negative sampling ratio = 4
- **Metric**: HR@10 = 0.409
- **Serving**: Real-time inference `/recommend/{userId}`

### LSTM Autoencoder — Anomaly Detection
- **Input**: Chuỗi page events trong 1 session (tối đa 50 events)
- **Training**: Unsupervised, học reconstruction của session bình thường
- **Threshold**: P95 reconstruction error = 0.042
- **Anomaly rate**: ~5% sessions bị flag
- **Serving**: Real-time scoring `/anomaly/realtime`

### ALS Recommendation (Baseline)
- Spark MLlib Alternating Least Squares
- Giữ làm baseline để so sánh với NCF

### Churn Prediction
- XGBoost / GBTClassifier phân tán trên Spark
- Features: behavioral aggregates từ Silver layer
- Label: user bấm Downgrade hoặc Cancellation

---

## 📁 Cấu trúc Thư mục

```
streamlify/
├── containers/
│   ├── docker-compose.yml      # Toàn bộ stack (12+ services)
│   ├── .env                    # Biến môi trường
│   ├── airflow/                # Custom Airflow image
│   ├── spark/                  # Custom Spark image (pre-baked JARs)
│   ├── dashboard/              # Streamlit image
│   ├── eventsim/               # Data generator image
│   └── serving/                # FastAPI image
└── src/
    ├── dags/
    │   ├── etl_fast_dag.py         # Bronze→Silver→Gold (15 phút/lần)
    │   ├── ml_training_dag.py      # ALS + XGBoost (hàng ngày 2AM)
    │   ├── kafka_to_minio_dag.py   # Kafka→Bronze Streaming
    │   └── dl_training_dag.py      # NCF + LSTM AE (hàng tuần)
    ├── jobs/
    │   ├── batch/
    │   │   ├── raw_to_silver.py
    │   │   ├── song_count_report.py
    │   │   ├── churn_prediction.py
    │   │   └── train_recommendation.py
    │   ├── streaming/
    │   │   └── kafka_to_minio.py
    │   ├── dl/
    │   │   ├── data_prep/export_training_data.py
    │   │   ├── models/train_ncf.py
    │   │   ├── models/train_lstm_ae.py
    │   │   ├── serving/api.py
    │   │   └── utils/minio_utils.py
    │   └── monitoring/
    │       └── ml_monitoring.py
    └── dashboard/
        ├── app.py
        └── monitoring_page.py
```

---

## 🚀 Luồng Dữ liệu

1. **Bronze Layer**: Spark Streaming đọc liên tục từ Kafka → ghi raw Parquet vào `s3a://bronze-zone/`
2. **Silver Layer**: Airflow trigger Spark Batch mỗi 15 phút → clean, deduplicate, enrich → `s3a://silver-zone/`
3. **Gold Layer**:
   - Song Rankings: Top bài hát theo play count
   - ALS Recommendations: Collaborative Filtering phân tán
   - Churn Predictions: XGBoost dự đoán user sắp rời bỏ
   - NCF Recommendations: Neural CF học non-linear patterns
   - Anomaly Scores: LSTM AE flag session bất thường
4. **Serving**: FastAPI load models từ MinIO → expose REST endpoints
5. **Monitoring**: KS-test so sánh data distribution hàng giờ → alert nếu drift

---

## 🔌 API Endpoints

```bash
GET  /health                     # Trạng thái models
GET  /recommend/{userId}?k=10    # Top-K songs (NCF real-time)
GET  /recommend/als/{userId}     # Top-K songs (ALS pre-computed)
POST /anomaly/realtime           # Score session đang diễn ra
GET  /anomaly/session/{id}       # Anomaly score pre-computed
GET  /churn/{userId}             # Xác suất churn
```

**Ví dụ:**
```bash
# NCF Recommendation
curl "http://localhost:8000/recommend/100?k=5"
# → {"userId":"100","model":"ncf","songs":[{"rank":1,"song":"Stairway to Heaven","score":0.61},...]}

# Realtime Anomaly Detection
curl -X POST "http://localhost:8000/anomaly/realtime" \
  -H "Content-Type: application/json" \
  -d '["Home","NextSong","Thumbs Down","Settings","Downgrade"]'
# → {"recon_error":0.012,"threshold":0.042,"is_anomaly":false,"anomaly_score":0.29}
```

---

## ⏰ Airflow DAGs

| DAG | Lịch chạy | Mô tả |
|---|---|---|
| `spark_kafka_to_minio_bronze` | Always-on | Kafka → Bronze (Streaming) |
| `etl_fast_pipeline` | Mỗi 15 phút | Bronze → Silver → Gold + Drift monitoring |
| `ml_training_pipeline` | Hàng ngày 2:00 AM | Retrain ALS + XGBoost |
| `dl_training_pipeline` | Chủ nhật 3:00 AM | Retrain NCF + LSTM AE |

---

## 📊 ML Monitoring

Chạy tự động 1 lần/giờ qua Airflow. So sánh distribution của Silver data hiện tại với baseline bằng **Kolmogorov-Smirnov test** (p-value < 0.05 = drift detected).

Output:
- HTML report → `gold-zone/monitoring/reports/YYYY-MM-DD.html`
- JSON summary → `gold-zone/monitoring/summary/YYYY-MM-DD.json`
- Metrics → MLflow experiment `streamlify_monitoring`

---

## 🔧 Hướng dẫn Cài đặt

**Yêu cầu**: Docker Desktop, RAM 8GB+, (tuỳ chọn) NVIDIA GPU để train DL

```bash
# 1. Clone repo
git clone https://github.com/AnkjoD/MusicStream.git
cd MusicStream/containers

# 2. Setup environment
cp .env.example .env

# 3. Khởi động toàn bộ stack
docker compose up -d

# 4. Truy cập services
# Airflow:   http://localhost:8888
# Spark UI:  http://localhost:8082
# Kafka UI:  http://localhost:8081
# MinIO:     http://localhost:9001
# MLflow:    http://localhost:5000
# Dashboard: http://localhost:8501
# API:       http://localhost:8000
```

**Train DL models** (cần có Silver data trước):
```bash
cd src/jobs/dl

python data_prep/export_training_data.py  # Export Silver → PyTorch dataset
python models/train_ncf.py                # Train NCF (tự động dùng GPU nếu có)
python models/train_lstm_ae.py            # Train LSTM Autoencoder

# Reload models vào serving
docker compose restart ml-serving
```

---

## 🎯 Quyết định Thiết kế

**Tại sao MinIO thay vì HDFS?** Setup 1 node cho môi trường dev, S3-compatible API đảm bảo zero code change khi deploy lên AWS S3.

**Tại sao NCF bên cạnh ALS?** ALS học linear interaction (dot product). NCF thêm MLP layers để học non-linear patterns. Giữ cả 2 để A/B comparison trong production.

**Tại sao LSTM Autoencoder cho anomaly?** Session sequences ngắn (10-50 events) — LSTM đủ mạnh và nhẹ hơn Transformer. Unsupervised approach không cần labeled anomaly data.

**Tại sao KS-test thay vì Evidently?** Implement trực tiếp bằng scipy tránh dependency conflicts, minh bạch hơn về underlying statistics.

---

*Phát triển bởi **AnkjoD**.*
