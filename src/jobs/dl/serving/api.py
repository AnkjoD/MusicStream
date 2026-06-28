"""
Bước 4: Thiết lập FastAPI Model Serving.

Nạp cả 4 mô hình khi khởi động server, cung cấp các REST API:

    GET /recommend/{userId}          -> Trả về Top-K bài hát gợi ý từ model NCF deep learning.
    GET /recommend/als/{userId}      -> Trả về Top-K bài hát gợi ý từ model ALS baseline.
    GET /anomaly/session/{sessionId} -> Trả về điểm bất thường của phiên (LSTM AE).
    GET /churn/{userId}              -> Trả về xác suất người dùng rời bỏ dịch vụ (XGBoost).
    GET /health                      -> Kiểm tra trạng thái hoạt động của server.

Ý tưởng thiết kế:
    - Các mô hình được tải đúng 1 lần từ MinIO khi khởi động (startup event) và lưu trực tiếp trên RAM để phản hồi siêu tốc.
    - Dùng FastAPI bất đồng bộ (async) để phục vụ nhiều request đồng thời mà không bị nghẽn (non-blocking).
    - Các mô hình deep learning (NCF, LSTM AE) chạy dự đoán trên CPU, vẫn dư sức đáp ứng độ trễ cực thấp (< 50ms).
    - Sử dụng mapping để dịch ngược từ ID số nguyên (song_idx) thành tên bài hát cụ thể.
"""

import os
import io
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from minio import Minio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT",  "minio:9000").replace("http://", "")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioadmin")
GOLD_BUCKET    = "gold-zone"

DEVICE = torch.device("cpu")  # Serving dùng CPU — GPU để train


# ── Paste lại model classes (cần để load state_dict) ─────────────────────────
# Trong project thật: import từ models/ncf.py và models/lstm_ae.py

class NCF(torch.nn.Module):
    def __init__(self, n_users, n_songs, emb_dim=64, mlp_layers=None, dropout=0.2):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 128, 64]
        self.gmf_user_emb = torch.nn.Embedding(n_users, emb_dim)
        self.gmf_song_emb = torch.nn.Embedding(n_songs, emb_dim)
        self.mlp_user_emb = torch.nn.Embedding(n_users, emb_dim)
        self.mlp_song_emb = torch.nn.Embedding(n_songs, emb_dim)
        mlp, in_dim = [], emb_dim * 2
        for out_dim in mlp_layers:
            mlp.extend([torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout)])
            in_dim = out_dim
        self.mlp = torch.nn.Sequential(*mlp)
        self.predict_layer = torch.nn.Linear(emb_dim + mlp_layers[-1], 1)

    def forward(self, user_idx, song_idx):
        gmf_out = self.gmf_user_emb(user_idx) * self.gmf_song_emb(song_idx)
        mlp_out = self.mlp(torch.cat([self.mlp_user_emb(user_idx), self.mlp_song_emb(song_idx)], dim=-1))
        return torch.sigmoid(self.predict_layer(torch.cat([gmf_out, mlp_out], dim=-1)).squeeze(-1))

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True,
                                   dropout=dropout if n_layers > 1 else 0)
        self.fc = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return torch.tanh(self.fc(h_n[-1]))


class LSTMDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_layers, dropout, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = torch.nn.LSTM(latent_dim, hidden_dim, n_layers, batch_first=True,
                                   dropout=dropout if n_layers > 1 else 0)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z_rep)
        return torch.sigmoid(self.fc(out))


class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, seq_len, hidden_dim=64, latent_dim=16, n_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = LSTMEncoder(1, hidden_dim, latent_dim, n_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, 1, n_layers, dropout, seq_len)

    def reconstruction_error(self, x):
        x_3d = x.unsqueeze(-1)
        z = self.encoder(x_3d)
        recon = self.decoder(z).squeeze(-1)
        return ((x - recon) ** 2).mean(dim=1)

# ── Global state (loaded at startup) ──────────────────────────────────────────

class ModelStore:
    ncf_model:        NCF | None = None
    ncf_config:       dict       = {}
    ae_model:         LSTMAutoencoder | None = None
    ae_config:        dict       = {}
    user_to_idx:      dict       = {}   # userId → int
    idx_to_user:      dict       = {}   # int → userId
    song_to_idx:      dict       = {}   # song → int
    idx_to_song:      dict       = {}   # int → song name
    als_recs:         pd.DataFrame | None = None  # Pre-computed ALS recommendations
    churn_preds:      pd.DataFrame | None = None  # Pre-computed churn predictions
    anomaly_scores:   pd.DataFrame | None = None  # Pre-computed anomaly scores


store = ModelStore()


def get_minio_client():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)


def read_bytes(client, bucket, key) -> bytes:
    resp = client.get_object(bucket, key)
    data = resp.read(); resp.close(); resp.release_conn()
    return data


def read_json_minio(client, bucket, key) -> dict:
    return json.loads(read_bytes(client, bucket, key))


def read_parquet_minio(client, bucket, key) -> pd.DataFrame:
    return pq.read_table(io.BytesIO(read_bytes(client, bucket, key))).to_pandas()


def load_ncf(client: Minio):
    logger.info("Loading NCF model...")
    try:
        config     = read_json_minio(client, GOLD_BUCKET, "models/ncf/config.json")
        state_dict = torch.load(
            io.BytesIO(read_bytes(client, GOLD_BUCKET, "models/ncf/model.pt")),
            map_location=DEVICE,
        )
        model = NCF(
            n_users=config["n_users"],
            n_songs=config["n_songs"],
            emb_dim=config.get("emb_dim", 64),
            mlp_layers=config.get("mlp_layers", [256, 128, 64]),
            dropout=config.get("dropout", 0.2),
        ).to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        store.ncf_model  = model
        store.ncf_config = config
        logger.info(f"NCF loaded ✅ | n_users={config['n_users']} | n_songs={config['n_songs']}")
    except Exception as e:
        logger.warning(f"NCF not available: {e}")


def load_lstm_ae(client: Minio):
    logger.info("Loading LSTM Autoencoder...")
    try:
        config     = read_json_minio(client, GOLD_BUCKET, "models/lstm_ae/config.json")
        state_dict = torch.load(
            io.BytesIO(read_bytes(client, GOLD_BUCKET, "models/lstm_ae/model.pt")),
            map_location=DEVICE,
        )
        model = LSTMAutoencoder(
            seq_len=config.get("max_seq_len", 50),
            hidden_dim=config.get("hidden_dim", 64),
            latent_dim=config.get("latent_dim", 16),
            n_layers=config.get("n_layers", 2),
            dropout=config.get("dropout", 0.2),
        ).to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        store.ae_model  = model
        store.ae_config = config
        logger.info(f"LSTM AE loaded ✅ | threshold={config.get('threshold', 'N/A'):.6f}")
    except Exception as e:
        logger.warning(f"LSTM AE not available: {e}")


def load_mappings(client: Minio):
    logger.info("Loading ID mappings...")
    try:
        user_map = read_json_minio(client, GOLD_BUCKET, "datalake/gold/ncf_training/user_mapping.json")
        song_map = read_json_minio(client, GOLD_BUCKET, "datalake/gold/ncf_training/song_mapping.json")
        store.user_to_idx = user_map
        store.idx_to_user = {v: k for k, v in user_map.items()}
        store.song_to_idx = song_map
        store.idx_to_song = {v: k for k, v in song_map.items()}
        logger.info(f"Mappings loaded ✅ | {len(user_map)} users | {len(song_map)} songs")
    except Exception as e:
        logger.warning(f"Mappings not available: {e}")


def load_precomputed(client: Minio):
    """Load pre-computed results (ALS recs, churn, anomaly scores)."""
    logger.info("Loading pre-computed Gold results...")

    for name, bucket, key, attr in [
        ("ALS recs",   GOLD_BUCKET, "datalake/gold/recommendations/part-0.parquet",    "als_recs"),
        ("Churn",      GOLD_BUCKET, "datalake/gold/churn_predictions/part-0.parquet",  "churn_preds"),
        ("Anomaly",    GOLD_BUCKET, "datalake/gold/anomaly_scores/scores.parquet",      "anomaly_scores"),
    ]:
        try:
            # List files in prefix (Spark writes multiple part files)
            prefix = key.rsplit("/", 1)[0] + "/"
            files  = [o.object_name for o in client.list_objects(bucket, prefix=prefix, recursive=True)
                      if o.object_name.endswith(".parquet")]
            if files:
                dfs = [read_parquet_minio(client, bucket, f) for f in files]
                setattr(store, attr, pd.concat(dfs, ignore_index=True))
                logger.info(f"{name} loaded ✅ | {len(getattr(store, attr)):,} rows")
        except Exception as e:
            logger.warning(f"{name} not available: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load tất cả models lúc startup."""
    client = get_minio_client()
    load_ncf(client)
    load_lstm_ae(client)
    load_mappings(client)
    load_precomputed(client)
    logger.info("🚀 FastAPI ready!")
    yield
    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Streamlify ML Serving API",
    description="NCF Recommendations + LSTM AE Anomaly Detection + Churn Prediction",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Response schemas ──────────────────────────────────────────────────────────

class RecommendResponse(BaseModel):
    userId:  str
    model:   str
    top_k:   int
    songs:   list[dict[str, Any]]

class AnomalyResponse(BaseModel):
    sessionId:     str | int
    recon_error:   float
    threshold:     float
    is_anomaly:    bool
    anomaly_score: float

class ChurnResponse(BaseModel):
    userId:       str
    churn_prob:   float
    is_churn:     bool
    risk_level:   str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "ncf":     store.ncf_model is not None,
            "lstm_ae": store.ae_model  is not None,
            "als":     store.als_recs  is not None,
            "churn":   store.churn_preds is not None,
        }
    }


@app.get("/recommend/{userId}", response_model=RecommendResponse)
async def recommend_ncf(userId: str, k: int = 10):
    """
    NCF recommendation: realtime inference.
    Score tất cả songs, trả về top-K.
    """
    if store.ncf_model is None:
        raise HTTPException(503, "NCF model chưa được load")
    if userId not in store.user_to_idx:
        raise HTTPException(404, f"userId '{userId}' không có trong training data")

    user_idx  = store.user_to_idx[userId]
    n_songs   = store.ncf_config["n_songs"]

    with torch.no_grad():
        user_tensor = torch.tensor([user_idx] * n_songs, device=DEVICE)
        song_tensor = torch.arange(n_songs, device=DEVICE)
        scores = store.ncf_model(user_tensor, song_tensor).cpu().numpy()

    top_k_idx = np.argsort(-scores)[:k]
    songs = [
        {
            "rank":     int(i + 1),
            "song":     store.idx_to_song.get(int(idx), f"song_{idx}"),
            "score":    float(scores[idx]),
        }
        for i, idx in enumerate(top_k_idx)
    ]

    return RecommendResponse(userId=userId, model="ncf", top_k=k, songs=songs)


@app.get("/recommend/als/{userId}", response_model=RecommendResponse)
async def recommend_als(userId: str, k: int = 10):
    """
    ALS recommendation: từ pre-computed Gold layer.
    Nhanh hơn NCF vì không cần inference.
    """
    if store.als_recs is None:
        raise HTTPException(503, "ALS recommendations chưa được load")

    # ALS lưu dạng user_idx → list of (song_idx, rating)
    if userId not in store.user_to_idx:
        raise HTTPException(404, f"userId '{userId}' không có trong data")

    user_idx = store.user_to_idx[userId]
    user_row = store.als_recs[store.als_recs["user_idx"] == user_idx]
    if user_row.empty:
        raise HTTPException(404, f"Chưa có ALS rec cho userId '{userId}'")

    # recommendations column là array of structs {song_idx, rating}
    recs = user_row.iloc[0]["recommendations"]
    songs = [
        {
            "rank":   i + 1,
            "song":   store.idx_to_song.get(int(r["song_idx"]), f"song_{r['song_idx']}"),
            "score":  float(r["rating"]),
        }
        for i, r in enumerate(recs[:k])
    ]

    return RecommendResponse(userId=userId, model="als", top_k=k, songs=songs)


@app.get("/anomaly/session/{sessionId}", response_model=AnomalyResponse)
async def anomaly_score_precomputed(sessionId: str):
    """
    Anomaly score từ pre-computed Gold layer.
    Dùng cho dashboard — không cần realtime inference.
    """
    if store.anomaly_scores is None:
        raise HTTPException(503, "Anomaly scores chưa được load")

    row = store.anomaly_scores[store.anomaly_scores["sessionId"].astype(str) == sessionId]
    if row.empty:
        raise HTTPException(404, f"sessionId '{sessionId}' không tìm thấy")

    r = row.iloc[0]
    threshold = store.ae_config.get("threshold", 0.01)
    return AnomalyResponse(
        sessionId=sessionId,
        recon_error=float(r["recon_error"]),
        threshold=float(threshold),
        is_anomaly=bool(r["is_anomaly"]),
        anomaly_score=float(r["anomaly_score"]),
    )


@app.post("/anomaly/realtime")
async def anomaly_realtime(page_sequence: list[str]):
    """
    Realtime anomaly scoring: nhận chuỗi page events, trả về score ngay.
    Dùng khi muốn score session đang diễn ra (streaming use case).
    """
    if store.ae_model is None:
        raise HTTPException(503, "LSTM AE model chưa được load")

    page_to_idx = store.ae_config.get("page_to_idx", {})
    max_seq_len = store.ae_config.get("max_seq_len", 50)

    # Encode sequence
    seq = [page_to_idx.get(p, 0) for p in page_sequence]
    seq_padded = np.zeros(max_seq_len, dtype=np.float32)
    length = min(len(seq), max_seq_len)
    seq_padded[:length] = np.array(seq[:length]) / store.ae_config.get("vocab_size", 23)

    tensor = torch.tensor(seq_padded, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        error = store.ae_model.reconstruction_error(tensor).item()

    threshold = store.ae_config.get("threshold", 0.01)
    return {
        "recon_error":   error,
        "threshold":     threshold,
        "is_anomaly":    error > threshold,
        "anomaly_score": min(error / threshold, 5.0),
        "input_length":  len(page_sequence),
    }


@app.get("/churn/{userId}", response_model=ChurnResponse)
async def churn_score(userId: str):
    """Churn prediction từ pre-computed Gold layer."""
    if store.churn_preds is None:
        raise HTTPException(503, "Churn predictions chưa được load")

    row = store.churn_preds[store.churn_preds["userId"].astype(str) == userId]
    if row.empty:
        raise HTTPException(404, f"userId '{userId}' không tìm thấy")

    r = row.iloc[0]
    # probability column từ XGBoost là array [prob_0, prob_1]
    prob = r["probability"]
    churn_prob = float(prob[1]) if hasattr(prob, "__len__") else float(prob)

    risk_level = "high" if churn_prob > 0.7 else ("medium" if churn_prob > 0.4 else "low")

    return ChurnResponse(
        userId=userId,
        churn_prob=churn_prob,
        is_churn=churn_prob > 0.5,
        risk_level=risk_level,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
