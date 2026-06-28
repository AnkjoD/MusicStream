"""
Bước 3: Train model LSTM Autoencoder để phát hiện các phiên (session) bất thường.

Cơ chế hoạt động của mô hình:
    Mô hình học thế nào là một phiên truy cập bình thường (normal session), từ đó phát hiện ra những phiên bất thường (anomalous).

    Đặc trưng đầu vào (Input): Chuỗi hành động trong một session của user.
         Ví dụ: ["Home", "NextSong", "NextSong", "Thumbs Up", "NextSong"]

    Cấu trúc mạng:
        - Bộ mã hóa (Encoder): LSTM nén chuỗi hành động thành một vector ẩn đại diện (latent vector/bottleneck).
        - Bộ giải mã (Decoder): LSTM cố gắng tái tạo lại chuỗi hành động ban đầu từ vector ẩn đó.

    Đánh giá lỗi tái tạo (Reconstruction Error):
        - Nếu lỗi tái tạo cao -> Session bất thường (hành vi lạ).
        - Ví dụ: Bấm liên tục Thumbs Down rồi Downgrade ngay (dấu hiệu Churn), session quá ngắn nhưng mò vào Settings/Help (user đang bực bội), hoặc nghe nhạc liên tục không ngừng nghỉ (pattern giống Bot).

Quy trình xử lý (Pipeline):
    Đọc Silver -> Nhóm theo sessionId -> Mã hóa các trang -> Đồng bộ độ dài chuỗi (padding/truncating)
    -> Train mô hình LSTM AE -> Lấy ngưỡng (threshold) ở phân vị thứ 95 (percentile 95) của lỗi tái tạo
    -> Đánh dấu các session bất thường và lưu lại vào Gold layer.

Tại sao dùng LSTM thay vì Transformer?
    Độ dài chuỗi session ở đây tương đối ngắn (~10-50 hành động) nên LSTM là quá đủ dùng, không cần tốn chi phí cho cơ chế Attention.
    Transformer chỉ phát huy tác dụng khi có lượng dữ liệu cực lớn để học positional patterns.
    LSTM gọn nhẹ hơn, suy luận (inference) nhanh giúp dễ dàng tích hợp real-time scoring sau này.
"""

import os
import io
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import pyarrow.parquet as pq
import pyarrow as pa
from minio import Minio
import mlflow
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
SILVER_BUCKET  = "silver-zone"
SILVER_PREFIX  = "datalake/silver/eventsim"
GOLD_BUCKET    = "gold-zone"
MODEL_PREFIX   = "models/lstm_ae"
OUTPUT_PREFIX  = "datalake/gold/anomaly_scores"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Hyperparams
MAX_SEQ_LEN    = int(os.getenv("AE_MAX_SEQ",     "50"))   # Max events per session (pad/truncate)
HIDDEN_DIM     = int(os.getenv("AE_HIDDEN",      "64"))   # LSTM hidden size
LATENT_DIM     = int(os.getenv("AE_LATENT",      "16"))   # Bottleneck dimension
N_LAYERS       = int(os.getenv("AE_LAYERS",      "2"))    # LSTM layers
DROPOUT        = float(os.getenv("AE_DROPOUT",   "0.2"))
LR             = float(os.getenv("AE_LR",        "1e-3"))
BATCH_SIZE     = int(os.getenv("AE_BATCH",       "256"))
EPOCHS         = int(os.getenv("AE_EPOCHS",      "30"))
PATIENCE       = int(os.getenv("AE_PATIENCE",    "5"))
ANOMALY_PCTILE = float(os.getenv("AE_PCTILE",    "95"))   # Threshold percentile

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tất cả unique page events trong eventsim
ALL_PAGES = [
    "<PAD>",           # 0: padding token
    "Home",            # 1
    "NextSong",        # 2
    "Thumbs Up",       # 3
    "Thumbs Down",     # 4
    "Add to Playlist", # 5
    "Roll Advert",     # 6
    "Add Friend",      # 7
    "Logout",          # 8
    "Settings",        # 9
    "Help",            # 10
    "About",           # 11
    "Login",           # 12
    "Register",        # 13
    "Submit Registration", # 14
    "Error",           # 15
    "Downgrade",       # 16
    "Submit Downgrade",# 17
    "Upgrade",         # 18
    "Submit Upgrade",  # 19
    "Cancellation Confirmation", # 20
    "Cancel",          # 21
    "Save Settings",   # 22
]
PAGE_TO_IDX = {p: i for i, p in enumerate(ALL_PAGES)}
VOCAB_SIZE   = len(ALL_PAGES)


# ── Data ──────────────────────────────────────────────────────────────────────

def get_minio_client():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)


def load_silver(client: Minio) -> pd.DataFrame:
    print("📦 [LSTM AE] Đọc Silver layer...")
    objects = client.list_objects(SILVER_BUCKET, prefix=SILVER_PREFIX, recursive=True)
    files   = [o.object_name for o in objects if o.object_name.endswith(".parquet")]

    dfs = []
    for f in files:
        resp = client.get_object(SILVER_BUCKET, f)
        data = resp.read(); resp.close(); resp.release_conn()
        df = pq.read_table(io.BytesIO(data)).to_pandas()

        # ── Fix partition column ──
        # Silver partition theo page= nên cột page nằm trong folder name
        # phải extract từ path thủ công
        import re
        match = re.search(r'page=([^/]+)/', f)
        if match and 'page' not in df.columns:
            df['page'] = match.group(1)

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"   {len(df):,} events | {df['sessionId'].nunique():,} sessions")
    return df

def build_session_sequences(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Group events theo sessionId, sort theo ts, encode page → int.
    Trả về:
        sequences: (n_sessions, MAX_SEQ_LEN) int array — padded
        session_meta: DataFrame với sessionId, userId, session_length
    """
    print("⚙️  [LSTM AE] Build session sequences...")

    df = df[df["page"].notna() & df["sessionId"].notna()].copy()
    df["page_idx"] = df["page"].map(lambda p: PAGE_TO_IDX.get(p, 0))

    # Sort events trong mỗi session theo thời gian
    df = df.sort_values(["sessionId", "ts"])

    # Group → list of page_idx
    grouped = df.groupby("sessionId").agg(
        userId=("userId",  "first"),
        page_seq=("page_idx", list),
        session_length=("page_idx", "count"),
    ).reset_index()

    # Filter sessions quá ngắn (< 3 events) — không đủ pattern
    grouped = grouped[grouped["session_length"] >= 3].reset_index(drop=True)

    print(f"   Sessions hợp lệ: {len(grouped):,}")
    print(f"   Session length: mean={grouped['session_length'].mean():.1f} | max={grouped['session_length'].max()}")

    # Pad/truncate → (n_sessions, MAX_SEQ_LEN)
    sequences = np.zeros((len(grouped), MAX_SEQ_LEN), dtype=np.int32)
    for i, seq in enumerate(grouped["page_seq"]):
        length = min(len(seq), MAX_SEQ_LEN)
        sequences[i, :length] = seq[:length]

    return sequences, grouped[["sessionId", "userId", "session_length"]]


class SessionDataset(Dataset):
    def __init__(self, sequences: np.ndarray):
        # Normalize thành float [0, 1] — LSTM làm việc tốt hơn với giá trị nhỏ
        self.data = torch.tensor(sequences / VOCAB_SIZE, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    LSTM Encoder: sequence → latent vector.
    Lấy hidden state của timestep cuối làm representation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim=1)
        _, (h_n, _) = self.lstm(x)
        # h_n: (n_layers, batch, hidden_dim) → lấy layer cuối
        h_last = h_n[-1]  # (batch, hidden_dim)
        return torch.tanh(self.fc(h_last))  # (batch, latent_dim)


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder: latent vector → reconstruct sequence.
    Repeat latent MAX_SEQ_LEN lần → feed vào LSTM.
    """
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, n_layers: int, dropout: float, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        # Repeat latent để decoder có input ở mỗi timestep
        z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, latent_dim)
        out, _ = self.lstm(z_repeated)
        return torch.sigmoid(self.fc(out))  # (batch, seq_len, output_dim)


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder: encode + decode.
    Loss = MSE(input, reconstruction) → minimize reconstruction error.
    """
    def __init__(
        self,
        seq_len:    int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_layers:   int = 2,
        dropout:    float = 0.2,
    ):
        super().__init__()
        input_dim = 1  # 1 feature per timestep (normalized page_idx)

        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, n_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, n_layers, dropout, seq_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len) → reshape → (batch, seq_len, 1)
        x_3d = x.unsqueeze(-1)
        z = self.encoder(x_3d)
        recon = self.decoder(z).squeeze(-1)  # (batch, seq_len)
        return recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Trả về MSE per sample (dùng cho anomaly scoring)."""
        recon, _ = self(x)
        return ((x - recon) ** 2).mean(dim=1)  # (batch,)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(batch)
    return total_loss / len(loader.dataset)


def compute_anomaly_scores(model: nn.Module, sequences: np.ndarray) -> np.ndarray:
    """Tính reconstruction error cho tất cả sessions."""
    model.eval()
    dataset = SessionDataset(sequences)
    loader  = DataLoader(dataset, batch_size=512, shuffle=False)
    scores  = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            err = model.reconstruction_error(batch)
            scores.append(err.cpu().numpy())
    return np.concatenate(scores)


# ── Upload helpers ─────────────────────────────────────────────────────────────

def upload_bytes(client, bucket, key, data: bytes):
    client.put_object(bucket, key, io.BytesIO(data), len(data))


def upload_model_ae(client: Minio, model: nn.Module, threshold: float, config: dict):
    print("💾 [LSTM AE] Upload model lên MinIO...")
    # model weights
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    client.put_object(GOLD_BUCKET, f"{MODEL_PREFIX}/model.pt", buf, buf.getbuffer().nbytes)

    # config (bao gồm threshold)
    config["threshold"] = threshold
    upload_bytes(client, GOLD_BUCKET, f"{MODEL_PREFIX}/config.json",
                 json.dumps(config, indent=2).encode())
    print(f"   ✅ model.pt + config.json → s3a://{GOLD_BUCKET}/{MODEL_PREFIX}/")


def upload_anomaly_results(client: Minio, session_meta: pd.DataFrame,
                           scores: np.ndarray, threshold: float):
    """Upload kết quả anomaly detection lên Gold layer."""
    result_df = session_meta.copy()
    result_df["recon_error"] = scores
    result_df["is_anomaly"]  = (scores > threshold).astype(int)
    result_df["anomaly_score"] = (scores / threshold).clip(0, 5)  # Normalize 0-5

    n_anomaly = result_df["is_anomaly"].sum()
    print(f"   Anomalous sessions: {n_anomaly:,} / {len(result_df):,} ({n_anomaly/len(result_df)*100:.1f}%)")

    table = pa.Table.from_pandas(result_df)
    buf   = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    client.put_object(GOLD_BUCKET, f"{OUTPUT_PREFIX}/scores.parquet",
                      buf, buf.getbuffer().nbytes)
    print(f"   ✅ scores.parquet → s3a://{GOLD_BUCKET}/{OUTPUT_PREFIX}/")
    return result_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"🚀 [LSTM AE] Device: {DEVICE}")

    client = get_minio_client()

    # 1. Load Silver + build sequences
    silver_df = load_silver(client)
    sequences, session_meta = build_session_sequences(silver_df)

    # Train/val split (90/10 — unsupervised nên không cần test label)
    n = len(sequences)
    idx = np.random.default_rng(42).permutation(n)
    split = int(n * 0.9)
    train_seqs = sequences[idx[:split]]
    val_seqs   = sequences[idx[split:]]

    train_dataset = SessionDataset(train_seqs)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=(DEVICE.type == "cuda"))

    # 2. Model
    model = LSTMAutoencoder(
        seq_len=MAX_SEQ_LEN,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params:,}")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # 3. MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("streamlify_lstm_ae")

    with mlflow.start_run(run_name=f"lstm_ae_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_params({
            "max_seq_len": MAX_SEQ_LEN,
            "hidden_dim":  HIDDEN_DIM,
            "latent_dim":  LATENT_DIM,
            "n_layers":    N_LAYERS,
            "dropout":     DROPOUT,
            "lr":          LR,
            "batch_size":  BATCH_SIZE,
            "anomaly_pctile": ANOMALY_PCTILE,
        })

        best_val_loss = float("inf")
        best_state    = None
        patience_ctr  = 0

        # 4. Training loop
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)

            # Val loss
            val_scores   = compute_anomaly_scores(model, val_seqs)
            val_loss = val_scores.mean()

            mlflow.log_metrics({"train_loss": train_loss, "val_recon_error": val_loss}, step=epoch)
            print(f"   Epoch {epoch:02d}/{EPOCHS} | Train loss: {train_loss:.6f} | Val recon: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"   ⏹  Early stopping tại epoch {epoch}")
                    break

        # 5. Compute threshold trên toàn bộ training set
        model.load_state_dict(best_state)
        print("📊 [LSTM AE] Tính anomaly threshold trên training data...")
        all_train_scores = compute_anomaly_scores(model, train_seqs)
        threshold = float(np.percentile(all_train_scores, ANOMALY_PCTILE))
        print(f"   Threshold (P{ANOMALY_PCTILE:.0f}): {threshold:.6f}")

        mlflow.log_metric("anomaly_threshold", threshold)

        # 6. Score toàn bộ sessions → Gold
        print("📊 [LSTM AE] Scoring tất cả sessions...")
        all_scores = compute_anomaly_scores(model, sequences)
        result_df  = upload_anomaly_results(client, session_meta, all_scores, threshold)

        # 7. Save model
        config = {
            "max_seq_len":    MAX_SEQ_LEN,
            "hidden_dim":     HIDDEN_DIM,
            "latent_dim":     LATENT_DIM,
            "n_layers":       N_LAYERS,
            "dropout":        DROPOUT,
            "vocab_size":     VOCAB_SIZE,
            "page_to_idx":    PAGE_TO_IDX,
            "best_val_recon": float(best_val_loss),
            "trained_at":     datetime.now().isoformat(),
        }
        upload_model_ae(client, model, threshold, config)

        n_anomaly = result_df["is_anomaly"].sum()
        print(f"""
╔══════════════════════════════════════════════╗
║  ✅ LSTM AUTOENCODER TRAINING HOÀN TẤT!      ║
║  Best val recon : {best_val_loss:.6f}              ║
║  Threshold P{ANOMALY_PCTILE:.0f}  : {threshold:.6f}              ║
║  Anomalous sess : {n_anomaly:,}                    ║
╚══════════════════════════════════════════════╝
        """)


if __name__ == "__main__":
    main()
