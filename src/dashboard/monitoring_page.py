"""
Streamlit Monitoring Page
=========================
Thêm file này vào src/dashboard/ rồi import vào app.py chính.

Hiển thị:
    - Drift score timeline (chart)
    - Latest alerts
    - HTML report từ Evidently (embedded iframe)
    - Model performance timeline (NCF HR@10, Churn AUC)

Usage trong app.py:
    from dashboard.monitoring_page import render_monitoring_page
    
    # Trong sidebar:
    page = st.sidebar.selectbox("Page", ["Overview", "ML Monitoring", ...])
    if page == "ML Monitoring":
        render_monitoring_page()
"""

import io
import json
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from minio import Minio

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000").replace("http://", "")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "minioadmin")
GOLD_BUCKET    = "gold-zone"
SUMMARY_PREFIX = "monitoring/summary"
REPORT_PREFIX  = "monitoring/reports"


@st.cache_resource
def get_minio_client():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)


def read_bytes(client, bucket, key) -> bytes:
    resp = client.get_object(bucket, key)
    data = resp.read(); resp.close(); resp.release_conn()
    return data


def load_summaries(client: Minio, days: int = 30) -> list[dict]:
    """Load tất cả JSON summaries trong N ngày gần nhất."""
    summaries = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        key  = f"{SUMMARY_PREFIX}/{date}.json"
        try:
            data = read_bytes(client, GOLD_BUCKET, key)
            summaries.append(json.loads(data))
        except Exception:
            continue
    return summaries


def load_latest_html_report(client: Minio) -> str | None:
    """Load HTML report mới nhất."""
    for i in range(7):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        key  = f"{REPORT_PREFIX}/{date}.html"
        try:
            data = read_bytes(client, GOLD_BUCKET, key)
            return data.decode("utf-8")
        except Exception:
            continue
    return None


def render_monitoring_page():
    st.title("🔍 ML Monitoring Dashboard")
    st.caption("Data Drift Detection + Model Performance Tracking — powered by Evidently")

    client = get_minio_client()

    # ── Load data ──
    with st.spinner("Loading monitoring data..."):
        summaries = load_summaries(client, days=30)

    if not summaries:
        st.warning(
            "⚠️ Chưa có monitoring data. "
            "Chạy `ml_monitoring.py` hoặc đợi Airflow ETL pipeline trigger."
        )
        st.code(
            "python src/jobs/monitoring/ml_monitoring.py",
            language="bash"
        )
        return

    # Sort theo date
    summaries.sort(key=lambda x: x["date"])
    latest = summaries[-1]

    # ── Header metrics ──
    col1, col2, col3, col4 = st.columns(4)

    drift_score = latest["drift"]["drift_score"]
    n_drifted   = latest["drift"]["n_drifted_cols"]
    n_total     = latest["drift"]["n_total_cols"]
    n_alerts    = len(latest.get("alerts", []))

    col1.metric(
        "Drift Score",
        f"{drift_score:.3f}",
        delta=None,
        help="Dataset drift score (0=no drift, 1=full drift). Alert nếu > 0.3",
    )
    col2.metric(
        "Drifted Features",
        f"{n_drifted}/{n_total}",
        help="Số features bị drift so với baseline",
    )

    ncf_val   = latest["performance"]["ncf"].get("value")
    churn_val = latest["performance"]["churn"].get("value")
    col3.metric(
        "NCF HR@10",
        f"{ncf_val:.4f}" if ncf_val else "N/A",
        help="Hit Rate @10 của NCF model (cao hơn = tốt hơn)",
    )
    col4.metric(
        "Churn AUC",
        f"{churn_val:.4f}" if churn_val else "N/A",
        help="AUC của Churn Prediction model",
    )

    # ── Alerts ──
    if latest.get("alerts"):
        st.markdown("---")
        st.subheader("🚨 Active Alerts")
        for alert in latest["alerts"]:
            severity = alert.get("severity", "LOW")
            icon = "🔴" if severity == "HIGH" else ("🟡" if severity == "MEDIUM" else "🟢")
            st.error(f"{icon} **[{severity}]** {alert['message']}")
    else:
        st.success("✅ No alerts — data distribution ổn định")

    # ── Drift timeline chart ──
    st.markdown("---")
    st.subheader("📈 Drift Score Timeline (30 ngày)")

    timeline_data = pd.DataFrame([
        {
            "date":        s["date"],
            "drift_score": s["drift"]["drift_score"],
            "n_drifted":   s["drift"]["n_drifted_cols"],
            "has_drift":   s["drift"]["dataset_drift"],
        }
        for s in summaries
    ])
    timeline_data["date"] = pd.to_datetime(timeline_data["date"])

    # Vẽ chart
    import altair as alt

    threshold_line = alt.Chart(
        pd.DataFrame({"threshold": [0.3]})
    ).mark_rule(color="red", strokeDash=[5, 5]).encode(
        y="threshold:Q"
    )

    drift_chart = alt.Chart(timeline_data).mark_line(
        point=True, color="#4C9BE8"
    ).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("drift_score:Q", title="Drift Score", scale=alt.Scale(domain=[0, 1])),
        tooltip=["date:T", "drift_score:Q", "n_drifted:Q", "has_drift:N"],
    )

    st.altair_chart(drift_chart + threshold_line, use_container_width=True)
    st.caption("🔴 Đường đứt nét = threshold 0.3. Vượt qua = cần retrain model.")

    # ── Drifted columns ──
    drifted_cols = latest["drift"].get("drifted_columns", [])
    if drifted_cols:
        st.markdown("---")
        st.subheader("⚠️ Features bị drift")
        col_details = latest["drift"].get("column_details", {})
        drift_df = pd.DataFrame([
            {
                "Feature":   col,
                "Stat Test": col_details.get(col, {}).get("stattest", "N/A"),
                "P-value":   col_details.get(col, {}).get("p_value", None),
                "Drifted":   "✅" if col_details.get(col, {}).get("drift") else "❌",
            }
            for col in col_details
        ])
        st.dataframe(drift_df, use_container_width=True)

    # ── Evidently HTML Report ──
    st.markdown("---")
    st.subheader("📊 Full Evidently Report")

    with st.spinner("Loading Evidently report..."):
        html_report = load_latest_html_report(client)

    if html_report:
        # Embed HTML report trong iframe
        st.components.v1.html(html_report, height=800, scrolling=True)
    else:
        st.info("Chưa có HTML report cho 7 ngày gần nhất.")

    # ── Monitoring info ──
    st.markdown("---")
    with st.expander("ℹ️ Về ML Monitoring"):
        st.markdown("""
        **Data Drift Detection** sử dụng [Evidently AI](https://www.evidentlyai.com/):
        
        - **Baseline**: Distribution của data lúc train model lần đầu (lưu tại `gold-zone/monitoring/baseline/`)
        - **Current**: Distribution của Silver data 7 ngày gần nhất
        - **Statistical test**: Kolmogorov-Smirnov test cho numerical features
        - **Drift threshold**: p-value < 0.05 → feature bị drift
        
        **Khi nào nên retrain?**
        - Drift score > 0.3 (nhiều hơn 30% features thay đổi phân phối)
        - NCF HR@10 giảm > 5% so với lần train trước
        - Churn AUC giảm > 5%
        
        **Monitoring chạy khi nào?**  
        Tự động 1 lần/giờ qua Airflow `etl_fast_pipeline` DAG.
        """)


# Standalone test
if __name__ == "__main__":
    st.set_page_config(page_title="ML Monitoring", layout="wide")
    render_monitoring_page()
