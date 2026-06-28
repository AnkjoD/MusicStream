"""
ML Monitoring — Data Drift + Model Performance
===============================================
Tự implement drift detection bằng scipy KS-test — không dùng Evidently
để tránh version conflict issues.

Output:
    - JSON summary → MinIO (gold-zone/monitoring/summary/YYYY-MM-DD.json)
    - HTML report  → MinIO (gold-zone/monitoring/reports/YYYY-MM-DD.html)
    - MLflow metrics để track theo thời gian
"""

import os
import io
import json
import re
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from minio import Minio
from scipy import stats
import mlflow
from dotenv import load_dotenv
from pathlib import Path

# Load cấu hình môi trường từ file .env
root = Path(__file__).resolve().parents[3] / "containers"
env_local = root / ".env.local"
# Ưu tiên load file .env.local khi dev ở máy cá nhân, nếu không có thì dùng file .env gốc
load_dotenv(env_local if env_local.exists() else root / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9090").replace("http://", "")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioadmin")
SILVER_BUCKET  = "silver-zone"
SILVER_PREFIX  = "datalake/silver/eventsim"
GOLD_BUCKET    = "gold-zone"
BASELINE_KEY   = "monitoring/baseline/baseline.parquet"
REPORT_PREFIX  = "monitoring/reports"
SUMMARY_PREFIX = "monitoring/summary"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DRIFT_THRESHOLD     = float(os.getenv("DRIFT_THRESHOLD", "0.05"))  # KS p-value threshold

MONITOR_FEATURES = [
    "total_songs", "thumbs_down", "thumbs_up",
    "settings_visits", "help_visits", "total_sessions",
    "days_active", "dislike_ratio", "avg_sessions_per_day",
]


# ── MinIO helpers ─────────────────────────────────────────────────────────────

def get_minio_client():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)


def read_parquet_minio(client, bucket, key) -> pd.DataFrame:
    resp = client.get_object(bucket, key)
    data = resp.read(); resp.close(); resp.release_conn()
    return pq.read_table(io.BytesIO(data)).to_pandas()


def upload_bytes(client, bucket, key, data: bytes, content_type="application/octet-stream"):
    client.put_object(bucket, key, io.BytesIO(data), len(data), content_type=content_type)
    logger.info(f"Uploaded: s3a://{bucket}/{key}")


def object_exists(client, bucket, key) -> bool:
    try:
        client.stat_object(bucket, key)
        return True
    except Exception:
        return False


# ── Load Silver ───────────────────────────────────────────────────────────────

def load_silver(client: Minio) -> pd.DataFrame:
    logger.info("📦 Load Silver layer...")
    objects = client.list_objects(SILVER_BUCKET, prefix=SILVER_PREFIX, recursive=True)
    files   = [o.object_name for o in objects if o.object_name.endswith(".parquet")]

    if not files:
        raise FileNotFoundError("Silver layer trống!")

    dfs = []
    for f in files:
        resp = client.get_object(SILVER_BUCKET, f)
        data = resp.read(); resp.close(); resp.release_conn()
        df = pq.read_table(io.BytesIO(data)).to_pandas()

        # Restore partition column
        match = re.search(r'page=([^/]+)/', f)
        if match and 'page' not in df.columns:
            df['page'] = match.group(1)

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    logger.info(f"   {len(df):,} events | {df['userId'].nunique():,} users")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def build_user_features(silver_df: pd.DataFrame) -> pd.DataFrame:
    silver_df = silver_df.dropna(subset=["userId", "event_time"])
    max_date  = silver_df["event_time"].max()

    agg = silver_df.groupby("userId").agg(
        total_songs    =("page", lambda x: (x == "NextSong").sum()),
        thumbs_down    =("page", lambda x: (x == "Thumbs Down").sum()),
        thumbs_up      =("page", lambda x: (x == "Thumbs Up").sum()),
        settings_visits=("page", lambda x: (x == "Settings").sum()),
        help_visits    =("page", lambda x: (x == "Help").sum()),
        total_sessions =("sessionId", "nunique"),
        first_event    =("event_time", "min"),
    ).reset_index()

    agg["days_active"] = (max_date - agg["first_event"]).dt.days.clip(lower=1)
    agg["dislike_ratio"] = np.where(
        agg["total_songs"] > 0,
        agg["thumbs_down"] / agg["total_songs"], 0.0
    )
    agg["avg_sessions_per_day"] = agg["total_sessions"] / agg["days_active"]
    return agg[["userId"] + MONITOR_FEATURES].fillna(0)


# ── Baseline ──────────────────────────────────────────────────────────────────

def load_or_create_baseline(client: Minio, current: pd.DataFrame) -> pd.DataFrame:
    if object_exists(client, GOLD_BUCKET, BASELINE_KEY):
        logger.info("📦 Load existing baseline...")
        return read_parquet_minio(client, GOLD_BUCKET, BASELINE_KEY)
    else:
        logger.info("🆕 Tạo baseline mới từ current data...")
        table = pa.Table.from_pandas(current)
        buf   = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        client.put_object(GOLD_BUCKET, BASELINE_KEY, buf, buf.getbuffer().nbytes)
        return current


# ── Drift Detection (KS-test) ─────────────────────────────────────────────────

def run_drift_detection(baseline: pd.DataFrame, current: pd.DataFrame, today: str) -> tuple[dict, bytes]:
    """
    Chạy kiểm định Kolmogorov-Smirnov (KS-test) cho từng đặc trưng.
    KS test giúp so sánh sự tương đồng giữa 2 phân phối dữ liệu mà không cần giả định phân phối chuẩn (Gaussian).
    Nếu p-value bé hơn ngưỡng (drift threshold) -> phân phối lệch nhau nhiều -> phát hiện data drift.
    """
    logger.info("🔍 Chạy KS-test drift detection...")

    ref = baseline[MONITOR_FEATURES]
    cur = current[MONITOR_FEATURES]

    column_results = {}
    n_drifted = 0

    for col in MONITOR_FEATURES:
        ks_stat, p_value = stats.ks_2samp(
            ref[col].dropna().values,
            cur[col].dropna().values,
        )
        drifted = p_value < DRIFT_THRESHOLD
        if drifted:
            n_drifted += 1

        column_results[col] = {
            "ks_statistic": float(ks_stat),
            "p_value":      float(p_value),
            "drift":        drifted,
            "ref_mean":     float(ref[col].mean()),
            "cur_mean":     float(cur[col].mean()),
            "ref_std":      float(ref[col].std()),
            "cur_std":      float(cur[col].std()),
        }

    # Điểm drift tổng hợp = tỷ lệ các đặc trưng bị lệch (drifted features)
    drift_score = n_drifted / len(MONITOR_FEATURES)
    has_drift   = n_drifted > len(MONITOR_FEATURES) // 3  # Nếu lệch quá 1/3 tổng số đặc trưng (>33%) thì coi như toàn bộ dataset bị lệch

    summary = {
        "date":             today,
        "n_baseline_users": len(baseline),
        "n_current_users":  len(current),
        "dataset_drift":    has_drift,
        "drift_score":      drift_score,
        "n_drifted_cols":   n_drifted,
        "n_total_cols":     len(MONITOR_FEATURES),
        "drifted_columns":  [c for c, v in column_results.items() if v["drift"]],
        "column_details":   column_results,
        "alert":            drift_score > 0.3,
    }

    # Tự render file báo cáo HTML trực quan (khỏi cần cài đặt Evidently làm nặng venv)
    html = _build_html_report(summary, column_results, today)

    return summary, html.encode("utf-8")

def _json_serial(obj):
    """Chuyển đổi mấy kiểu dữ liệu của numpy sang kiểu chuẩn Python để ghi được file JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

def _build_html_report(summary: dict, column_results: dict, today: str) -> str:
    """Tự ráp chuỗi HTML thô để làm file báo cáo, không phụ thuộc thư viện ngoài."""
    rows = ""
    for col, v in column_results.items():
        status = "🔴 DRIFT" if v["drift"] else "🟢 OK"
        rows += f"""
        <tr>
            <td>{col}</td>
            <td>{v['ks_statistic']:.4f}</td>
            <td>{v['p_value']:.4f}</td>
            <td>{v['ref_mean']:.2f} ± {v['ref_std']:.2f}</td>
            <td>{v['cur_mean']:.2f} ± {v['cur_std']:.2f}</td>
            <td>{status}</td>
        </tr>"""

    alert_html = ""
    if summary["alert"]:
        alert_html = f"""
        <div style="background:#ff4444;color:white;padding:12px;border-radius:6px;margin:16px 0">
            ⚠️ DATA DRIFT DETECTED — Score: {summary['drift_score']:.2f}
            — Drifted: {summary['drifted_columns']}
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ML Monitoring Report — {today}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 32px; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .metric {{ display: inline-block; margin: 8px 16px; text-align: center; }}
  .metric .value {{ font-size: 2em; font-weight: bold; color: #4C9BE8; }}
  .metric .label {{ color: #666; font-size: 0.9em; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #4C9BE8; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h1>🔍 ML Monitoring Report — {today}</h1>
{alert_html}
<div class="card">
  <h2>Summary</h2>
  <div class="metric"><div class="value">{summary['drift_score']:.2f}</div><div class="label">Drift Score</div></div>
  <div class="metric"><div class="value">{summary['n_drifted_cols']}/{summary['n_total_cols']}</div><div class="label">Drifted Features</div></div>
  <div class="metric"><div class="value">{summary['n_baseline_users']:,}</div><div class="label">Baseline Users</div></div>
  <div class="metric"><div class="value">{summary['n_current_users']:,}</div><div class="label">Current Users</div></div>
</div>
<div class="card">
  <h2>Feature Drift Details (KS-test, threshold p &lt; {0.05})</h2>
  <table>
    <tr>
      <th>Feature</th><th>KS Statistic</th><th>P-value</th>
      <th>Baseline (mean±std)</th><th>Current (mean±std)</th><th>Status</th>
    </tr>
    {rows}
  </table>
</div>
<div class="card" style="color:#888;font-size:0.85em">
  Generated by Streamlify ML Monitoring — scipy KS-test (no Evidently dependency)
</div>
</body>
</html>"""


# ── MLflow Performance ────────────────────────────────────────────────────────

def get_latest_metric(experiment_name: str, metric_name: str) -> float | None:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return None
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
        if not runs:
            return None
        return float(runs[0].data.metrics.get(metric_name, 0))
    except Exception as e:
        logger.warning(f"MLflow unavailable: {e}")
        return None




# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today  = datetime.now().strftime("%Y-%m-%d")
    client = get_minio_client()
    logger.info(f"🚀 ML Monitoring run: {today}")

    # 1. Load + feature engineering
    silver_df = load_silver(client)
    current   = build_user_features(silver_df)

    # 2. Baseline
    baseline = load_or_create_baseline(client, current)

    # 3. Drift detection
    drift_summary, html_bytes = run_drift_detection(baseline, current, today)

    # 4. Model performance từ MLflow
    ncf_hr10  = get_latest_metric("streamlify_ncf",     "best_val_hr10")
    churn_auc = get_latest_metric("streamlify_churn_xgb", "auc")

    perf = {
        "ncf_hr10":  ncf_hr10,
        "churn_auc": churn_auc,
    }

    # 5. Full summary
    full_summary = {
        "date":        today,
        "drift":       drift_summary,
        "performance": perf,
        "alerts":      [],
    }

    if drift_summary["alert"]:
        full_summary["alerts"].append({
            "type":     "DATA_DRIFT",
            "severity": "HIGH",
            "message":  f"Drift score={drift_summary['drift_score']:.2f} | Drifted: {drift_summary['drifted_columns']}",
        })

    # 6. Upload
    upload_bytes(client, GOLD_BUCKET, f"{REPORT_PREFIX}/{today}.html",  html_bytes,  "text/html")
    upload_bytes(client, GOLD_BUCKET, f"{SUMMARY_PREFIX}/{today}.json",
                 json.dumps(full_summary, indent=2, default=_json_serial).encode(), "application/json")

    # 7. MLflow log
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("streamlify_monitoring")
        with mlflow.start_run(run_name=f"monitor_{today}"):
            mlflow.log_metrics({
                "drift_score":       drift_summary["drift_score"],
                "n_drifted_columns": drift_summary["n_drifted_cols"],
                "n_current_users":   drift_summary["n_current_users"],
            })
            if ncf_hr10:  mlflow.log_metric("ncf_hr10",  ncf_hr10)
            if churn_auc: mlflow.log_metric("churn_auc", churn_auc)
    except Exception as e:
        logger.warning(f"MLflow log failed (non-critical): {e}")

    # 8. Print
    print(f"""
╔══════════════════════════════════════════════════╗
║  ML MONITORING — {today}              ║
╠══════════════════════════════════════════════════╣
║  Drift Score  : {drift_summary['drift_score']:.4f}                         ║
║  Has Drift    : {str(drift_summary['dataset_drift']):<5}                        ║
║  Drifted cols : {drift_summary['n_drifted_cols']}/{drift_summary['n_total_cols']}                           ║
║  NCF HR@10    : {str(ncf_hr10 or 'N/A'):<8}                    ║
║  Churn AUC    : {str(churn_auc or 'N/A'):<8}                    ║
║  Alerts       : {len(full_summary['alerts'])}                             ║
╚══════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()