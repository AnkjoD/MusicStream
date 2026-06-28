"""
Phase 1: Export Silver Layer → PyTorch Training Data
=====================================================

Job này đọc Silver parquet từ MinIO, tổng hợp interaction matrix
user×song, encode IDs thành integer index, rồi lưu ra 3 file:
    - interactions.parquet  → (user_idx, song_idx, play_count, label)
    - user_mapping.json     → userId → user_idx
    - song_mapping.json     → song   → song_idx

Chạy độc lập với Spark — Python + pandas thôi.
Lý do không dùng Spark: dataset sau aggregate ~vài MB, không cần distribute.
"""

import os
import json
import pandas as pd
import numpy as np
from minio import Minio
from minio.error import S3Error
import io
import pyarrow.parquet as pq
import pyarrow as pa
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

MINIO_ENDPOINT  = os.getenv("MINIO_ENDPOINT",  "minio:9000").replace("http://", "")
MINIO_ACCESS    = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET    = os.getenv("MINIO_SECRET_KEY", "minioadmin")
SILVER_BUCKET   = "silver-zone"
SILVER_PREFIX   = "datalake/silver/eventsim"
GOLD_BUCKET     = "gold-zone"
OUTPUT_PREFIX   = "datalake/gold/ncf_training"


# ── MinIO client ──────────────────────────────────────────────────────────────

def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=False,
    )


def list_parquet_files(client: Minio, bucket: str, prefix: str) -> list[str]:
    """List tất cả .parquet files dưới prefix."""
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects if obj.object_name.endswith(".parquet")]


def read_parquet_from_minio(client: Minio, bucket: str, object_name: str) -> pd.DataFrame:
    response = client.get_object(bucket, object_name)
    data = response.read()
    response.close()
    response.release_conn()
    buf = io.BytesIO(data)
    df = pq.read_table(buf).to_pandas()
    
    # Extract partition value từ path
    # Path dạng: datalake/silver/eventsim/page=NextSong/part-0.parquet
    import re
    match = re.search(r'page=([^/]+)/', object_name)
    if match and 'page' not in df.columns:
        df['page'] = match.group(1)
    
    return df


def upload_parquet_to_minio(client: Minio, bucket: str, object_name: str, df: pd.DataFrame):
    """Upload pandas DataFrame lên MinIO dưới dạng parquet."""
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    size = buf.getbuffer().nbytes
    client.put_object(bucket, object_name, buf, size)
    print(f"   ✅ Uploaded: s3a://{bucket}/{object_name} ({size/1024:.1f} KB)")


def upload_json_to_minio(client: Minio, bucket: str, object_name: str, data: dict):
    """Upload JSON lên MinIO."""
    content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    buf = io.BytesIO(content)
    client.put_object(bucket, object_name, buf, len(content))
    print(f"   ✅ Uploaded: s3a://{bucket}/{object_name} ({len(content)/1024:.1f} KB)")


# ── Core logic ────────────────────────────────────────────────────────────────

def load_silver(client: Minio) -> pd.DataFrame:
    """
    Đọc toàn bộ dữ liệu từ tầng Silver trên MinIO.
    Do Silver được phân mảnh (partition) theo trường 'page' nên bắt buộc phải quét thư mục đệ quy (recursive).
    """
    print("📦 [Data Prep] Đọc Silver layer từ MinIO...")
    files = list_parquet_files(client, SILVER_BUCKET, SILVER_PREFIX)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy parquet trong {SILVER_BUCKET}/{SILVER_PREFIX}")
    print(f"   Tìm thấy {len(files)} parquet files")

    dfs = []
    for f in files:
        try:
            df = read_parquet_from_minio(client, SILVER_BUCKET, f)
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️  Bỏ qua {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"   Tổng: {len(combined):,} events")
    return combined


def build_interaction_matrix(silver_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo ma trận tương tác (interaction matrix): đếm xem mỗi cặp (user, song) có bao nhiêu lượt nghe.
    Chúng ta chỉ lọc các sự kiện 'NextSong' vì lúc này người dùng thực sự nghe nhạc.
    """
    print("⚙️  [Data Prep] Tổng hợp interaction matrix...")

    nextsong = silver_df[
        silver_df["page"] == "NextSong"
    ][["userId", "song", "artist"]].dropna()

    interactions = (
        nextsong
        .groupby(["userId", "song", "artist"])
        .size()
        .reset_index(name="play_count")
    )

    # Lọc bỏ nhiễu: Loại các user/song có ít hơn 2 tương tác.
    # Đây là mẹo chuẩn trong Collaborative Filtering để tránh lỗi khởi đầu lạnh cực đoan (cold-start extreme).
    user_counts = interactions.groupby("userId")["play_count"].sum()
    song_counts = interactions.groupby("song")["play_count"].sum()

    active_users = user_counts[user_counts >= 2].index
    active_songs = song_counts[song_counts >= 2].index

    interactions = interactions[
        interactions["userId"].isin(active_users) &
        interactions["song"].isin(active_songs)
    ].reset_index(drop=True)

    print(f"   Users: {interactions['userId'].nunique():,}")
    print(f"   Songs: {interactions['song'].nunique():,}")
    print(f"   Interactions: {len(interactions):,}")
    print(f"   Sparsity: {1 - len(interactions) / (interactions['userId'].nunique() * interactions['song'].nunique()):.4f}")

    return interactions


def encode_ids(interactions: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Encode userId/song thành consecutive integers.
    Trả về: (encoded_df, user_map, song_map)

    # Tận dụng tính năng Categorical của pandas để ánh xạ nhằm tránh việc cài đặt thêm thư viện không cần thiết.
    """
    print("🔢 [Data Prep] Encode IDs → integers...")

    users = sorted(interactions["userId"].unique())
    songs = sorted(interactions["song"].unique())

    user_to_idx = {u: i for i, u in enumerate(users)}
    song_to_idx = {s: i for i, s in enumerate(songs)}

    interactions = interactions.copy()
    interactions["user_idx"] = interactions["userId"].map(user_to_idx).astype("int32")
    interactions["song_idx"] = interactions["song"].map(song_to_idx).astype("int32")

    # Gán nhãn phản hồi ngầm định (Implicit feedback):
    # NCF cần nhãn nhị phân (nghe từ 1 lần trở lên coi như thích và gán nhãn 1.0).
    # Cột play_count vẫn giữ lại phòng khi sau này muốn làm trọng số mẫu (sample weight).
    interactions["label"] = 1.0

    print(f"   n_users: {len(user_to_idx):,} | n_songs: {len(song_to_idx):,}")
    return interactions, user_to_idx, song_to_idx


def generate_negative_samples(
    interactions: pd.DataFrame,
    n_users: int,
    n_songs: int,
    neg_ratio: int = 4,
) -> pd.DataFrame:
    """
    Tạo các mẫu âm tính (Negative samples) - nghĩa là user chưa từng nghe bài hát đó.

    Đây là bước bắt buộc đối với phản hồi ngầm định (Implicit Feedback):
    Dataset gốc chỉ chứa các tương tác dương tính (đã nghe), mô hình cần học cả dữ liệu đối chứng 
    để biết user không nghe những bài nào. Ta sẽ chọn ngẫu nhiên các cặp (user, song) chưa có trong dữ liệu nghe thực tế.

    Dùng tỷ lệ 4:1 (neg_ratio=4) - đây là con số tối ưu được chứng minh trong bài báo gốc NCF (He et al., 2017).
    """
    print(f"🎲 [Data Prep] Tạo negative samples (ratio={neg_ratio})...")

    # Set các (user, song) đã có
    positive_set = set(zip(interactions["user_idx"], interactions["song_idx"]))

    rng = np.random.default_rng(seed=42)
    neg_users, neg_songs = [], []

    for user_idx in interactions["user_idx"].unique():
        # Với mỗi user, sample neg_ratio * số positive của user đó
        n_pos = (interactions["user_idx"] == user_idx).sum()
        n_neg = n_pos * neg_ratio

        sampled = 0
        attempts = 0
        while sampled < n_neg and attempts < n_neg * 10:
            song_idx = rng.integers(0, n_songs)
            if (user_idx, song_idx) not in positive_set:
                neg_users.append(user_idx)
                neg_songs.append(song_idx)
                sampled += 1
            attempts += 1

    neg_df = pd.DataFrame({
        "user_idx":   np.array(neg_users, dtype="int32"),
        "song_idx":   np.array(neg_songs, dtype="int32"),
        "play_count": 0,
        "label":      0.0,
    })

    full_df = pd.concat(
        [
            interactions[["user_idx", "song_idx", "play_count", "label"]],
            neg_df
        ],
        ignore_index=True
    ).sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    print(f"   Positives: {len(interactions):,} | Negatives: {len(neg_df):,}")
    print(f"   Total: {len(full_df):,}")
    return full_df


def train_val_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    # Phân chia dữ liệu theo tỷ lệ 70% Train, 15% Val và 15% Test.
    # Dùng stratify để giữ nguyên tỷ lệ nhãn dương/âm (1 và 0) đồng đều ở cả 3 tập.
    """
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    val, test   = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["label"])

    print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    client = get_minio_client()

    # 1. Load Silver
    silver_df = load_silver(client)

    # 2. Build interaction matrix
    interactions = build_interaction_matrix(silver_df)

    # 3. Encode IDs
    encoded, user_map, song_map = encode_ids(interactions)

    n_users = len(user_map)
    n_songs = len(song_map)

    # 4. Negative sampling
    full_df = generate_negative_samples(encoded, n_users, n_songs, neg_ratio=4)

    # 5. Split
    print("✂️  [Data Prep] Train/Val/Test split...")
    train_df, val_df, test_df = train_val_test_split(full_df)

    # 6. Upload lên MinIO
    print("💾 [Data Prep] Upload lên MinIO...")
    upload_parquet_to_minio(client, GOLD_BUCKET, f"{OUTPUT_PREFIX}/train.parquet",    train_df)
    upload_parquet_to_minio(client, GOLD_BUCKET, f"{OUTPUT_PREFIX}/val.parquet",      val_df)
    upload_parquet_to_minio(client, GOLD_BUCKET, f"{OUTPUT_PREFIX}/test.parquet",     test_df)

    # Lưu metadata (n_users, n_songs) cùng với mapping để NCF load đúng
    metadata = {"n_users": n_users, "n_songs": n_songs}
    upload_json_to_minio(client, GOLD_BUCKET, f"{OUTPUT_PREFIX}/metadata.json",   metadata)
    upload_json_to_minio(client, GOLD_BUCKET, f"{OUTPUT_PREFIX}/user_mapping.json", user_map)
    upload_json_to_minio(client, GOLD_BUCKET, f"{OUTPUT_PREFIX}/song_mapping.json", song_map)

    print(f"""
╔══════════════════════════════════════════╗
║  ✅ DATA PREP HOÀN TẤT!                  ║
║  n_users : {n_users:<8,}                    ║
║  n_songs : {n_songs:<8,}                    ║
║  Output  : gold-zone/{OUTPUT_PREFIX[:18]}  ║
╚══════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
