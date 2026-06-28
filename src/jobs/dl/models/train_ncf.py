"""
Train mô hình Gợi ý Nhạc sử dụng mạng Nơ-ron (Neural Collaborative Filtering - NCF).

Kiến trúc mô hình NCF (Dựa trên paper NeuMF nổi tiếng của He và cộng sự, 2017):
    NCF là sự kết hợp của hai nhánh GMF và MLP tạo nên NeuMF.

    Nhánh GMF (Generalized Matrix Factorization):
        user_emb ⊙ item_emb -> lớp tuyến tính -> logit.
        Nhánh này học các mối tương quan tuyến tính (tương tự thuật toán ALS nhưng chạy end-to-end).

    Nhánh MLP (Multi-Layer Perceptron):
        Nối (concatenate) hai vector embedding [user_emb || item_emb] rồi đẩy qua các lớp ẩn (Dense).
        Nhánh này giúp học các tương quan phi tuyến tính (non-linear) phức tạp — điều mà các mô hình Matrix Factorization truyền thống chịu chết.

    Mô hình NeuMF (Neural Matrix Factorization):
        Gộp kết quả của cả hai nhánh [gmf_out || mlp_out] -> lớp Dense(1) -> hàm Sigmoid để ra xác suất thích bài hát.
        Sự kết hợp này mang lại hiệu năng tối ưu nhất trong thực tế.

Tại sao NCF vượt trội hơn ALS?
    ALS dùng phép nhân vô hướng (dot product) mang tính tuyến tính đơn giản.
    NCF dùng mạng nơ-ron học hàm tương tác phi tuyến, giúp nắm bắt các hành vi phức tạp hơn của người dùng 
    (ví dụ: "Thích nghe nhạc Pop sôi động vào buổi sáng để làm việc nhưng tối lại thích Jazz nhẹ nhàng").

Chiến lược huấn luyện (Training Strategy):
    - Dùng nhãn ngầm định (Implicit feedback) với hàm mất mát Binary Cross-Entropy (BCE loss).
    - Tạo mẫu âm (Negative sampling) với tỷ lệ 4:1 để đảm bảo độ chính xác.
    - Huấn luyện trên GPU để tăng tốc độ xử lý.
    - Áp dụng cơ chế dừng sớm (Early stopping) dựa trên chỉ số HR@10 (Hit Rate tại top 10).
    - Đồng bộ ghi log và lưu trữ tham số (metrics, model) thông qua MLflow.
"""

import os
import json
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pyarrow.parquet as pq
from minio import Minio
import mlflow
import mlflow.pytorch
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Root project = streamlify/containers/
root = Path(__file__).resolve().parents[4] / "containers"

# Ưu tiên .env.local nếu tồn tại (local dev)
# Fallback về .env (Docker hoặc không có .env.local)
env_local = root / ".env.local"
env_default = root / ".env"

if env_local.exists():
    load_dotenv(env_local)
else:
    load_dotenv(env_default)
# ── Config ────────────────────────────────────────────────────────────────────

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT",  "minio:9000").replace("http://", "")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioadmin")
GOLD_BUCKET    = "gold-zone"
DATA_PREFIX    = "datalake/gold/ncf_training"
MODEL_PREFIX   = "models/ncf"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Hyperparams — có thể override bằng env vars để sweep
EMB_DIM     = int(os.getenv("NCF_EMB_DIM",    "64"))    # Embedding dimension
MLP_LAYERS  = [256, 128, 64]                              # MLP hidden sizes
DROPOUT     = float(os.getenv("NCF_DROPOUT",  "0.2"))
LR          = float(os.getenv("NCF_LR",       "1e-3"))
BATCH_SIZE  = int(os.getenv("NCF_BATCH",      "2048"))
EPOCHS      = int(os.getenv("NCF_EPOCHS",     "20"))
PATIENCE    = int(os.getenv("NCF_PATIENCE",   "3"))      # Early stopping
TOP_K       = 10                                          # Evaluate HR@10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """
    PyTorch Dataset cho NCF.
    Mỗi sample: (user_idx, song_idx, label)
    """
    def __init__(self, df: pd.DataFrame):
        self.users  = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.songs  = torch.tensor(df["song_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values,    dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.songs[idx], self.labels[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class NCF(nn.Module):
    """
    NeuMF = GMF + MLP combined.

    GMF branch: separate embeddings, element-wise product
    MLP branch: separate embeddings, concatenate → dense layers
    Output: concat [gmf_out, mlp_out] → sigmoid
    """
    def __init__(
        self,
        n_users:    int,
        n_songs:    int,
        emb_dim:    int       = 64,
        mlp_layers: list[int] = None,
        dropout:    float     = 0.2,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [256, 128, 64]

        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(n_users, emb_dim)
        self.gmf_song_emb = nn.Embedding(n_songs, emb_dim)

        # MLP embeddings (riêng biệt — paper gốc dùng riêng để tránh interference)
        self.mlp_user_emb = nn.Embedding(n_users, emb_dim)
        self.mlp_song_emb = nn.Embedding(n_songs, emb_dim)

        # MLP layers
        mlp = []
        in_dim = emb_dim * 2  # concat user + song
        for out_dim in mlp_layers:
            mlp.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp)

        # Final prediction: concat GMF output (emb_dim) + MLP output (last_layer)
        self.predict_layer = nn.Linear(emb_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier init cho embeddings và linear layers."""
        for emb in [self.gmf_user_emb, self.gmf_song_emb,
                    self.mlp_user_emb, self.mlp_song_emb]:
            nn.init.xavier_uniform_(emb.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.predict_layer.weight)

    def forward(self, user_idx: torch.Tensor, song_idx: torch.Tensor) -> torch.Tensor:
        # GMF branch
        gmf_u = self.gmf_user_emb(user_idx)
        gmf_s = self.gmf_song_emb(song_idx)
        gmf_out = gmf_u * gmf_s  # element-wise product

        # MLP branch
        mlp_u = self.mlp_user_emb(user_idx)
        mlp_s = self.mlp_song_emb(song_idx)
        mlp_in  = torch.cat([mlp_u, mlp_s], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # Combine
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        logit = self.predict_layer(combined).squeeze(-1)
        return torch.sigmoid(logit)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_hit_rate(
    model:    nn.Module,
    val_df:   pd.DataFrame,
    n_songs:  int,
    k:        int   = 10,
    n_sample: int   = 500,    # Sample n users để tính nhanh
) -> float:
    """
    HR@K (Hit Rate at K):
    Với mỗi user, lấy 1 positive item + (99 negative items) ngẫu nhiên.
    Rank 100 items, check nếu positive nằm trong top-K.
    HR@K = tỉ lệ users có positive trong top-K.

    Đây là metric chuẩn cho implicit CF evaluation (leave-one-out).
    """
    model.eval()
    rng = np.random.default_rng(42)

    # Chỉ lấy positive samples từ val
    pos_df = val_df[val_df["label"] == 1.0].copy()
    if len(pos_df) == 0:
        return 0.0

    # Sample n_sample users
    users = pos_df["user_idx"].unique()
    sampled_users = rng.choice(users, size=min(n_sample, len(users)), replace=False)

    # Set positives per user
    user_positives = pos_df.groupby("user_idx")["song_idx"].apply(set).to_dict()

    hits = 0
    with torch.no_grad():
        for user_idx in sampled_users:
            pos_songs = list(user_positives.get(user_idx, set()))
            if not pos_songs:
                continue
            # Pick 1 positive
            pos_song = rng.choice(pos_songs)

            # Sample 99 negatives (songs user chưa nghe)
            all_songs = np.arange(n_songs)
            known_songs = np.array(list(user_positives.get(user_idx, set())))
            candidate_negatives = np.setdiff1d(all_songs, known_songs)
            neg_songs = rng.choice(
                candidate_negatives,
                size=min(99, len(candidate_negatives)),
                replace=False
            )

            # Rank 100 candidates
            candidates = np.concatenate([[pos_song], neg_songs])
            user_tensor = torch.tensor([user_idx] * len(candidates), device=DEVICE)
            song_tensor = torch.tensor(candidates, dtype=torch.long, device=DEVICE)

            scores = model(user_tensor, song_tensor).cpu().numpy()
            # argsort descending
            ranked = np.argsort(-scores)
            # Check nếu pos_song (index 0) trong top-K
            rank = np.where(ranked == 0)[0][0]
            if rank < k:
                hits += 1

    return hits / len(sampled_users)


# ── MinIO helpers ─────────────────────────────────────────────────────────────

def get_minio_client():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)


def read_parquet(client: Minio, bucket: str, key: str) -> pd.DataFrame:
    resp = client.get_object(bucket, key)
    data = resp.read(); resp.close(); resp.release_conn()
    return pq.read_table(io.BytesIO(data)).to_pandas()


def read_json(client: Minio, bucket: str, key: str) -> dict:
    resp = client.get_object(bucket, key)
    data = resp.read(); resp.close(); resp.release_conn()
    return json.loads(data)


def upload_model(client: Minio, bucket: str, prefix: str, model: nn.Module, metadata: dict):
    """Save model state_dict + metadata lên MinIO."""
    # state_dict
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    client.put_object(bucket, f"{prefix}/model.pt", buf, buf.getbuffer().nbytes)
    print(f"   ✅ model.pt → s3a://{bucket}/{prefix}/model.pt")

    # metadata (n_users, n_songs, hyperparams)
    content = json.dumps(metadata, indent=2).encode()
    client.put_object(bucket, f"{prefix}/config.json", io.BytesIO(content), len(content))
    print(f"   ✅ config.json → s3a://{bucket}/{prefix}/config.json")


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for users, songs, labels in loader:
        users, songs, labels = users.to(DEVICE), songs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        preds = model(users, songs)
        loss  = criterion(preds, labels)
        loss.backward()
        # Gradient clipping — tránh exploding gradients với large embedding
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def main():
    print(f"🚀 [NCF] Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    client = get_minio_client()

    # ── Load data ──
    print("📦 [NCF] Load training data từ MinIO...")
    train_df = read_parquet(client, GOLD_BUCKET, f"{DATA_PREFIX}/train.parquet")
    val_df   = read_parquet(client, GOLD_BUCKET, f"{DATA_PREFIX}/val.parquet")
    metadata = read_json(client,   GOLD_BUCKET, f"{DATA_PREFIX}/metadata.json")

    n_users = metadata["n_users"]
    n_songs = metadata["n_songs"]
    print(f"   n_users={n_users:,} | n_songs={n_songs:,}")
    print(f"   Train: {len(train_df):,} | Val: {len(val_df):,}")

    # ── DataLoaders ──
    train_dataset = InteractionDataset(train_df)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # ── Model ──
    model = NCF(
        n_users=n_users,
        n_songs=n_songs,
        emb_dim=EMB_DIM,
        mlp_layers=MLP_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params:,}")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    criterion = nn.BCELoss()

    # ── MLflow ──
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("streamlify_ncf")

    with mlflow.start_run(run_name=f"ncf_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        # Log hyperparams
        mlflow.log_params({
            "emb_dim":    EMB_DIM,
            "mlp_layers": str(MLP_LAYERS),
            "dropout":    DROPOUT,
            "lr":         LR,
            "batch_size": BATCH_SIZE,
            "epochs":     EPOCHS,
            "n_users":    n_users,
            "n_songs":    n_songs,
        })

        best_hr = 0.0
        best_state = None
        patience_counter = 0

        # ── Training loop ──
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)

            # Evaluate HR@10 mỗi epoch
            hr = compute_hit_rate(model, val_df, n_songs, k=TOP_K)
            scheduler.step(hr)

            mlflow.log_metrics({"train_loss": train_loss, "val_hr10": hr}, step=epoch)

            print(f"   Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss:.4f} | HR@10: {hr:.4f}")

            # Early stopping + checkpoint
            if hr > best_hr:
                best_hr = hr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"   ✨ New best HR@10: {best_hr:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"   ⏹  Early stopping tại epoch {epoch}")
                    break

        # ── Load best model và save ──
        model.load_state_dict(best_state)

        model_config = {
            "n_users":    n_users,
            "n_songs":    n_songs,
            "emb_dim":    EMB_DIM,
            "mlp_layers": MLP_LAYERS,
            "dropout":    DROPOUT,
            "best_hr10":  best_hr,
            "trained_at": datetime.now().isoformat(),
        }

        print(f"💾 [NCF] Lưu best model lên MinIO (HR@10={best_hr:.4f})...")
        upload_model(client, GOLD_BUCKET, MODEL_PREFIX, model, model_config)

        # Log artifact lên MLflow
        mlflow.log_metric("best_val_hr10", best_hr)
        # mlflow.pytorch.log_model(model, "ncf_model")

        print(f"""
╔══════════════════════════════════════════╗
║  ✅ NCF TRAINING HOÀN TẤT!               ║
║  Best HR@10 : {best_hr:.4f}                   ║
║  Model      : gold-zone/{MODEL_PREFIX}   ║
╚══════════════════════════════════════════╝
        """)


if __name__ == "__main__":
    main()
