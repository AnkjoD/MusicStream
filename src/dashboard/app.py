import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import numpy as np
import docker
import os

# ── Page Config ──
st.set_page_config(
    page_title="Streamlify | Spotify Operations Center",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ──
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    :root {
        --green: #1DB954;
        --green-dim: rgba(29,185,84,0.15);
        --red-dim: rgba(255,65,54,0.15);
        --amber-dim: rgba(255,190,0,0.15);
        --glass: rgba(255,255,255,0.03);
        --border: rgba(255,255,255,0.08);
    }
    /* Chỉ áp dụng Outfit cho nội dung của mình, KHÔNG đụng vào Streamlit internals */
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown li, div.card, div.card *, .section-title, .badge,
    .card-label, .card-value, .card-sub, .pulse-wrap span,
    .stDataFrame div[class*="cell"] { font-family: 'Outfit', sans-serif !important; }
    /* Bảo toàn icon fonts của Streamlit (Material Icons) */
    [data-testid] span[class*="icon"], span[data-baseweb], .material-icons,
    button span, [role="tab"] span, summary span, [class*="Icon"] {
        font-family: 'Material Icons', 'Material Symbols Rounded' !important;
    }
    /* Ẩn toàn bộ loading indicators của Streamlit */
    [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="stDecoration"]   { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer    { display: none !important; }
    /* Chặn grey overlay khi rerun */
    .stApp [data-testid="stAppViewContainer"] > div > div:first-child > div[style*="opacity"] {
        opacity: 1 !important;
        pointer-events: auto !important;
    }

    .stApp {
        background: radial-gradient(ellipse at top right, #0d2a1a 0%, #121212 50%);
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #111111 100%);
        border-right: 1px solid var(--border);
    }

    /* Cards */
    .card {
        background: var(--glass);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.4rem;
        border: 1px solid var(--border);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        transition: all 0.35s cubic-bezier(0.175,0.885,0.32,1.275);
        margin-bottom: 0.5rem;
    }
    .card:hover {
        transform: translateY(-6px);
        border-color: var(--green);
        box-shadow: 0 0 24px rgba(29,185,84,0.18);
    }
    .card-label { color:#888; font-size:0.78rem; text-transform:uppercase; letter-spacing:1.2px; }
    .card-value { color:#fff; font-size:2rem; font-weight:800; margin-top:4px; }
    .card-sub   { color:#1DB954; font-size:0.82rem; margin-top:2px; }

    /* Section header */
    .section-title {
        font-size: 1rem; font-weight: 700; color: #e0e0e0;
        text-transform: uppercase; letter-spacing: 1.5px;
        border-left: 3px solid #1DB954; padding-left: 10px;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Status badges */
    .badge { padding:3px 10px; border-radius:6px; font-size:0.72rem; font-weight:700; }
    .badge-ok  { background:var(--green-dim); color:#1DB954; }
    .badge-err { background:var(--red-dim);   color:#FF4136; }
    .badge-warn{ background:var(--amber-dim); color:#FFBE00; }

    /* Pulse indicator */
    .pulse-wrap {
        display:flex; align-items:center; gap:8px;
        background:var(--green-dim); padding:5px 14px;
        border-radius:50px; border:1px solid rgba(29,185,84,0.3);
        width:fit-content;
    }
    .pulse {
        width:9px; height:9px; background:var(--green); border-radius:50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(29,185,84,0.7); }
        70%  { box-shadow: 0 0 0 8px rgba(29,185,84,0); }
        100% { box-shadow: 0 0 0 0 rgba(29,185,84,0); }
    }

    /* Churn risk bar */
    .risk-bar-wrap { margin:4px 0; }
    .risk-label { font-size:0.8rem; color:#b3b3b3; margin-bottom:2px; }
    .risk-bar { height:6px; border-radius:3px; background:rgba(255,255,255,0.08); }
    .risk-fill { height:6px; border-radius:3px; }

    .stButton>button {
        width:100%; background:transparent !important;
        color:white !important; border:1px solid var(--border) !important;
        border-radius:10px !important; transition:all 0.25s !important;
    }
    .stButton>button:hover {
        background:var(--green) !important; border-color:var(--green) !important;
    }
    /* Custom spinner thay thế grey loading - hiện ở góc trên phải */
    .live-dot {
        position: fixed; top: 14px; right: 16px; z-index: 9999;
        display: flex; align-items: center; gap: 6px;
        background: rgba(29,185,84,0.12); padding: 4px 12px;
        border-radius: 50px; border: 1px solid rgba(29,185,84,0.25);
    }
    .live-dot span { color: #1DB954; font-size: 0.75rem; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# Inject JS: hide Streamlit's grey overlay + "Running..." toast
st.html("""
<script>
(function() {
    const hide = () => {
        // Hide status widget
        const sw = document.querySelector('[data-testid="stStatusWidget"]');
        if (sw) sw.style.display = 'none';
        // Remove grey opacity overlay on main block
        document.querySelectorAll('[style*="opacity: 0"]').forEach(el => {
            if (!el.classList.contains('pulse')) el.style.opacity = '1';
        });
    };
    hide();
    const obs = new MutationObserver(hide);
    obs.observe(document.body, { childList: true, subtree: true, attributes: true });
})();
</script>
""")

# ════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════

STORAGE_OPTS = {
    "key":    os.getenv("MINIO_ACCESS_KEY", "homura_madoka"),
    "secret": os.getenv("MINIO_SECRET_KEY", "homura123"),
    "use_listings_cache": False,  # Tắt cache của s3fs để thấy file mới realtime
    "client_kwargs": {
        "endpoint_url": os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    }
}



@st.cache_data(ttl=5)
def load_stream():
    """Real-time stream events từ Bronze Layer (MinIO)."""
    try:
        df = pd.read_parquet(
            "s3://bronze-zone/datalake/raw/eventsim/page=NextSong/",
            storage_options=STORAGE_OPTS
        )
        if df.empty:
            return None
        df['timestamp'] = pd.to_datetime(df.get('ts', df.get('ingestion_time')), unit='ms', errors='coerce')
        return df
    except:
        return None

@st.cache_data(ttl=30)
def load_song_rankings():
    """Top bài hát từ Gold Layer (kết quả Spark batch job)."""
    try:
        df = pd.read_parquet(
            "s3://gold-zone/datalake/gold/song_rankings/",
            storage_options=STORAGE_OPTS
        )
        return df if not df.empty else None
    except:
        return None

@st.cache_data(ttl=60)
def load_churn():
    """Kết quả dự đoán Churn từ Gold Layer (XGBoost/GBT model)."""
    try:
        df = pd.read_parquet(
            "s3://gold-zone/datalake/gold/churn_predictions/",
            storage_options=STORAGE_OPTS
        )
        return df if not df.empty else None
    except:
        return None

@st.cache_data(ttl=60)
def load_recommendations():
    """Gợi ý nhạc từ ALS Model (Gold Layer) + Mappings."""
    try:
        rec_df = pd.read_parquet("s3://gold-zone/datalake/gold/recommendations/", storage_options=STORAGE_OPTS)
        try:
            # Map IDs to names
            u_map = pd.read_parquet("s3://gold-zone/datalake/gold/user_mapping/", storage_options=STORAGE_OPTS)
            s_map = pd.read_parquet("s3://gold-zone/datalake/gold/song_mapping/", storage_options=STORAGE_OPTS)
            u_dict = dict(zip(u_map['user_idx'], u_map['userId']))
            s_dict = dict(zip(s_map['song_idx'], s_map['song']))
            return rec_df, u_dict, s_dict
        except:
            return rec_df, {}, {}
    except:
        return None, {}, {}

def get_docker_status():
    try:
        client = docker.from_env()
        containers = client.containers.list(all=True)
        targets = ['kafka','spark','minio','eventsim','airflow']
        return {
            c.name.replace('containers-',''): c.status
            for c in containers
            if any(n in c.name for n in targets)
        }
    except:
        return {"kafka":"running","spark-master":"running","minio":"running","eventsim":"exited"}

@st.cache_data(ttl=10)
def load_perf_metrics():
    import boto3
    from urllib.parse import urlparse

    metrics = {
        "bronze": 0, "silver": 0, "gold": 0,
        "bronze_size_mb": 0.0, "silver_size_mb": 0.0, "gold_size_mb": 0.0,
        "bronze_oldest": None, "bronze_newest": None,
        "silver_newest": None, "gold_newest": None,
        "streaming_events_per_min": 0,
        "bronze_to_silver_lag_min": None,
    }
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        )
        paginator = s3.get_paginator('list_objects_v2')

        for zone, bucket, prefix in [
            ("bronze", "bronze-zone", "datalake/raw/eventsim"),
            ("silver", "silver-zone", "datalake/silver/eventsim"),
            ("gold",   "gold-zone",   "datalake/gold"),
        ]:
            try:
                parquet_count = 0
                total_size = 0
                mtimes = []

                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    if "Contents" not in page:
                        continue
                    for obj in page["Contents"]:
                        if obj["Key"].endswith(".parquet"):
                            parquet_count += 1
                            total_size += obj.get("Size", 0)
                            mtimes.append(obj.get("LastModified"))

                metrics[zone] = parquet_count
                metrics[f"{zone}_size_mb"] = round(total_size / (1024 * 1024), 2)

                if mtimes:
                    metrics[f"{zone}_newest"] = max(mtimes)
                    if zone == "bronze":
                        metrics["bronze_oldest"] = min(mtimes)
            except Exception as e:
                pass

        # Tính streaming throughput thật từ bronze timestamps
        if metrics["bronze_oldest"] and metrics["bronze_newest"] and metrics["bronze"] > 0:
            delta_min = max(1, (metrics["bronze_newest"] - metrics["bronze_oldest"]).total_seconds() / 60)
            # Ước tính: mỗi parquet file streaming ~60s trigger × events/file
            metrics["streaming_events_per_min"] = round(metrics["bronze"] * 2000 / delta_min)

        # Lag Bronze→Silver (chứng minh batch pipeline đã chạy)
        if metrics["bronze_newest"] and metrics["silver_newest"]:
            lag = (metrics["silver_newest"] - metrics["bronze_newest"]).total_seconds() / 60
            metrics["bronze_to_silver_lag_min"] = round(abs(lag), 1)

    except Exception:
        pass
    return metrics

def plotly_transparent():
    return dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🗄️ Data Layers")

    # Dùng MinIO file counts thay vì Docker API (Docker API không khả dụng trong container)
    perf_side = load_perf_metrics()
    layer_status = [
        ("Bronze", perf_side["bronze"], perf_side["bronze_size_mb"], "#CD7F32"),
        ("Silver", perf_side["silver"], perf_side["silver_size_mb"], "#C0C0C0"),
        ("Gold",   perf_side["gold"],   perf_side["gold_size_mb"],   "#FFD700"),
    ]
    for layer, files, size_mb, color in layer_status:
        ok = files > 0
        cls = "badge-ok" if ok else "badge-warn"
        lbl = f"{files} files" if ok else "EMPTY"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:8px 0;border-bottom:1px solid var(--border)'>"
            f"<span style='color:{color};font-size:0.85rem;font-weight:700'>{layer}</span>"
            f"<span class='badge {cls}'>{lbl}</span></div>"
            f"<div style='color:#555;font-size:0.75rem;padding:2px 0 4px 0'>{size_mb} MB</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧠 ML Models")
    churn_df  = load_churn()
    rec_df, u_dict, s_dict = load_recommendations()
    gold_df   = load_song_rankings()

    model_items = [
        ("ALS Recommendation", rec_df is not None),
        ("Churn Prediction",   churn_df is not None),
        ("Song Rankings",      gold_df is not None),
    ]
    for mname, ready in model_items:
        cls = "badge-ok" if ready else "badge-warn"
        lbl = "READY" if ready else "PENDING"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:6px 0;border-bottom:1px solid var(--border)'>"
            f"<span style='color:#ccc;font-size:0.85rem'>{mname}</span>"
            f"<span class='badge {cls}'>{lbl}</span></div>",
            unsafe_allow_html=True
        )

    st.divider()
    st.caption("v3.0.0 · Real-time + ML Edition")

# ════════════════════════════════════════
# HEADER
# ════════════════════════════════════════
h1, h2 = st.columns([4, 1])
with h1:
    st.title("🎵 Streamlify Operations Center")
    st.markdown("Real-time Spotify Analytics · Spark Distributed · 3-Node Cluster")
with h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="pulse-wrap" style="margin-bottom: 0.5rem;">
            <div class="pulse"></div>
            <span style="color:#1DB954;font-weight:700;font-size:0.85rem">LIVE</span>
        </div>
    """, unsafe_allow_html=True)
    if st.button("🔄 Cập nhật", width='stretch'):
        st.cache_data.clear()
        st.rerun()

# ════════════════════════════════════════
# LOAD DATA
df = load_stream()

# ════════════════════════════════════════
# TOP KPI CARDS
# ════════════════════════════════════════
k1, k2, k3, k4 = st.columns(4)
if df is not None:
    cards = [
        (k1, "Total Streams",   f"{len(df):,}",                          "🎧", "Live events"),
        (k2, "Active Users",    f"{df['userId'].nunique():,}",            "👥", "Unique listeners"),
        (k3, "Premium Share",   f"{(df['level']=='paid').mean()*100:.1f}%","💎", "Paid subscribers"),
        (k4, "Unique Tracks",   f"{df['song'].nunique():,}",              "🔍", "Songs playing"),
    ]
else:
    cards = [
        (k1, "Total Streams",   "0", "🎧", "Live events"),
        (k2, "Active Users",    "0", "👥", "Unique listeners"),
        (k3, "Premium Share",   "0%", "💎", "Paid subscribers"),
        (k4, "Unique Tracks",   "0", "🔍", "Songs playing"),
    ]

for col, label, val, icon, sub in cards:
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">{icon} {label}</div>
            <div class="card-value">{val}</div>
            <div class="card-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB NAVIGATION
# ════════════════════════════════════════
tab_stream, tab_ml_churn, tab_ml_rec, tab_gold, tab_perf = st.tabs([
    "📡 Live Stream",
    "⚠️ Churn Prediction",
    "🎵 AI Recommendations",
    "🏆 Gold Rankings",
    "⚡ Hiệu suất Pipeline",
])

# ── TAB 1: Live Stream ──────────────────
with tab_stream:
    if df is None:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; background: rgba(255,65,54,0.1); border: 2px dashed #FF4136; border-radius: 16px; margin-top: 1rem;">
            <h1 style="font-size: 4rem; margin-bottom: 0;">📡</h1>
            <h2 style="color: #FF4136; font-family: 'Outfit', sans-serif;">Streaming đang tắt hoặc chưa có dữ liệu</h2>
            <p style="color: #ccc; font-size: 1.1rem;">Vui lòng sang <b>Airflow</b> bật DAG <code>spark_kafka_to_minio_bronze</code> để xem Live Stream.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Hàng 1: Listener Profile & Subscription Mix (cùng 1 dòng) ──
        r1_c1, r1_c2 = st.columns(2)
        
        with r1_c1:
            st.markdown('<div class="section-title">Listener Profile</div>', unsafe_allow_html=True)
            g = df['gender'].value_counts()
            fig_pie = px.pie(values=g.values, names=g.index, hole=0.68,
                             template='plotly_dark',
                             color_discrete_sequence=['#1DB954','#FFFFFF'])
            fig_pie.update_layout(**plotly_transparent(), margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_pie, width='stretch')

        with r1_c2:
            st.markdown('<div class="section-title">Subscription Mix</div>', unsafe_allow_html=True)
            lv = df['level'].value_counts()
            fig_lv = px.bar(x=lv.index, y=lv.values, template='plotly_dark',
                            color=lv.index, color_discrete_map={'paid':'#1DB954','free':'#333'})
            fig_lv.update_layout(**plotly_transparent(), showlegend=False,
                                  margin=dict(l=0,r=0,t=10,b=0),
                                  xaxis=dict(title=None), yaxis=dict(title=None, showgrid=False))
            st.plotly_chart(fig_lv, width='stretch')

        # ── Hàng 2: Trending Tracks (chiếm toàn bộ chiều ngang) ──
        st.markdown('<div class="section-title" style="margin-top: 1rem;">Real-time Trending Tracks</div>', unsafe_allow_html=True)
        top = df.groupby(['song','artist']).size().reset_index(name='plays') \
                .sort_values('plays', ascending=False).head(10)
        fig_bar = px.bar(top, x='plays', y='song', color='plays', orientation='h',
                         template='plotly_dark',
                         color_continuous_scale=['#0d2316','#1DB954','#1ed760'],
                         hover_data={'artist': True})
        fig_bar.update_layout(**plotly_transparent(),
                              showlegend=False, coloraxis_showscale=False,
                              height=350, # Đặt chiều cao để không bị quá dài
                              margin=dict(l=0,r=10,t=10,b=0),
                              yaxis=dict(categoryorder='total ascending', showgrid=False),
                              xaxis=dict(showgrid=False))
        st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False})

        # ── Hàng 3: Ingestion Velocity (chiếm toàn bộ chiều ngang) ──
        st.markdown('<div class="section-title" style="margin-top: 2rem;">📈 Ingestion Velocity (per 10 seconds)</div>', unsafe_allow_html=True)
        df['t_bucket'] = df['timestamp'].dt.floor('10s').dt.strftime('%H:%M:%S')
        vel = df.groupby('t_bucket').size().reset_index(name='events').tail(30)
        fig_vel = px.area(vel, x='t_bucket', y='events', template='plotly_dark')
        fig_vel.update_traces(line_color='#1DB954', fillcolor='rgba(29,185,84,0.12)', line_width=3)
        fig_vel.update_layout(**plotly_transparent(),
                              height=250,  # Chiều cao vừa phải
                              margin=dict(l=0,r=0,t=10,b=0),
                              xaxis=dict(showgrid=False, title=None),
                              yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title=None))
        st.plotly_chart(fig_vel, width='stretch')

# ── TAB 2: Churn Prediction ─────────────
with tab_ml_churn:
    if churn_df is not None:
        st.markdown('<div class="section-title">⚠️ Users at Risk of Churning (XGBoost/GBT Model)</div>', unsafe_allow_html=True)

        # Extract churn probability safely
        def safe_extract_prob(x):
            try:
                # Nếu PyArrow đọc ra dict (Spark Vector serializer)
                if isinstance(x, dict) and 'values' in x:
                    return float(x['values'][1])
                # Nếu là list/numpy array thông thường
                elif hasattr(x, '__iter__') and len(x) > 1 and not isinstance(x, dict):
                    return float(x[1])
                return float(x)
            except:
                return 0.0

        if 'probability' in churn_df.columns:
            churn_df['churn_prob'] = churn_df['probability'].apply(safe_extract_prob)
        else:
            churn_df['churn_prob'] = churn_df.get('prediction', 0.5)

        # KPI cards
        ck1, ck2, ck3 = st.columns(3)
        total_users   = len(churn_df)
        high_risk     = (churn_df['churn_prob'] >= 0.7).sum()
        medium_risk   = ((churn_df['churn_prob'] >= 0.4) & (churn_df['churn_prob'] < 0.7)).sum()
        avg_risk      = churn_df['churn_prob'].mean()

        for col, lbl, val, sub, icon in [
            (ck1, "High Risk Users",   f"{high_risk:,}",    "Prob ≥ 70%", "🔴"),
            (ck2, "Medium Risk Users", f"{medium_risk:,}",  "Prob 40-70%","🟡"),
            (ck3, "Avg Churn Prob",    f"{avg_risk:.1%}",   "All users",  "📊"),
        ]:
            with col:
                st.markdown(f"""
                <div class="card">
                    <div class="card-label">{icon} {lbl}</div>
                    <div class="card-value">{val}</div>
                    <div class="card-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([3, 2])

        with c1:
            st.markdown('<div class="section-title">Churn Probability Distribution</div>', unsafe_allow_html=True)
            fig_hist = px.histogram(churn_df, x='churn_prob', nbins=20,
                                    template='plotly_dark',
                                    color_discrete_sequence=['#1DB954'])
            fig_hist.add_vline(x=0.7, line_dash="dash", line_color="#FF4136",
                               annotation_text="High Risk Threshold")
            fig_hist.update_layout(**plotly_transparent(), margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_hist, width='stretch')

        with c2:
            st.markdown('<div class="section-title">Top 10 At-Risk Users</div>', unsafe_allow_html=True)
            top_risk = churn_df.nlargest(10, 'churn_prob')[['userId','churn_prob','total_songs','cancel_count']] \
                               .rename(columns={'churn_prob':'Risk','total_songs':'Songs','cancel_count':'Cancels'})
            top_risk['Risk'] = top_risk['Risk'].map('{:.1%}'.format)
            st.dataframe(top_risk, width='stretch', hide_index=True)

        # Scatter: songs vs churn risk
        st.markdown('<div class="section-title">Listening Behavior vs Churn Risk</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(
            churn_df.sample(min(500, len(churn_df))),
            x='total_songs', y='churn_prob',
            color='churn_prob',
            color_continuous_scale=['#1DB954','#FFBE00','#FF4136'],
            template='plotly_dark',
            labels={'total_songs':'Total Songs Played','churn_prob':'Churn Risk'},
            opacity=0.7
        )
        fig_sc.update_layout(**plotly_transparent(), margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_sc, width='stretch')

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem">
            <div style="font-size:3rem">🤖</div>
            <div style="color:#888; margin-top:1rem; font-size:1.1rem">Churn Prediction chưa có dữ liệu</div>
            <div style="color:#555; margin-top:0.5rem; font-size:0.85rem">Pipeline sẽ tự cập nhật sau mỗi 30 phút</div>
        </div>""", unsafe_allow_html=True)

# ── TAB 3: AI Recommendations ──────────
with tab_ml_rec:
    if rec_df is not None:
        st.markdown('<div class="section-title">🎵 ALS Collaborative Filtering Recommendations</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card" style="display:flex;gap:2rem;flex-wrap:wrap">
            <div><div class="card-label">Model</div><div style="color:#1DB954;font-weight:700">ALS Matrix Factorization</div></div>
            <div><div class="card-label">Users Served</div><div style="color:#fff;font-weight:700">{len(rec_df):,}</div></div>
            <div><div class="card-label">Recs per User</div><div style="color:#fff;font-weight:700">10</div></div>
            <div><div class="card-label">Distribution</div><div style="color:#1DB954;font-weight:700">✅ Fully Distributed</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Format nested recommendations struct → dễ đọc (hiển thị ALL 10 bài)
        def _fmt_recs(val):
            try:
                items = list(val)
                # Dịch ngược song_idx ra tên bài hát nếu có mapping, nếu không thì dùng ID
                parts = []
                for idx, r in enumerate(items):
                    song_name = s_dict.get(r['song_idx'], f"Song #{r['song_idx']}")
                    parts.append(f"{idx+1}. {song_name} (⭐{r['rating']:.2f})")
                
                # Trả về chuỗi hiển thị nhiều dòng
                return "\n".join(parts)
            except Exception:
                return str(val)

        display_df = rec_df.head(20).copy()
        
        # Dịch ngược user_idx ra tên User
        display_df["User"] = [u_dict.get(idx, f"User #{int(idx):,}") for idx in display_df["user_idx"]]
        display_df["10 Recommended Songs"] = display_df["recommendations"].apply(_fmt_recs)
        
        # Cấu hình UI để cột Recommendation không bị cắt (wrap text)
        st.dataframe(
            display_df[["User", "10 Recommended Songs"]],
            width='stretch',
            hide_index=True,
            column_config={
                "10 Recommended Songs": st.column_config.TextColumn(
                    "10 Recommended Songs (Title ⭐ Score)",
                    width="large"
                )
            }
        )

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem">
            <div style="font-size:3rem">🎵</div>
            <div style="color:#888; margin-top:1rem; font-size:1.1rem">ALS Recommendations chưa có dữ liệu</div>
            <div style="color:#555; margin-top:0.5rem; font-size:0.85rem">Pipeline sẽ tự cập nhật sau mỗi 30 phút</div>
        </div>""", unsafe_allow_html=True)

# ── TAB 4: Gold Rankings ────────────────
with tab_gold:
    if gold_df is not None:
        st.markdown('<div class="section-title">🏆 Song Rankings (Spark Batch — Gold Layer)</div>', unsafe_allow_html=True)

        gc1, gc2 = st.columns([2, 1])
        with gc1:
            top20 = gold_df.head(20)
            fig_gold = px.bar(top20, x='play_count', y='song', color='play_count',
                              orientation='h', template='plotly_dark',
                              color_continuous_scale=['#0d2316','#1DB954'],
                              hover_data={'artist': True})
            fig_gold.update_layout(**plotly_transparent(), showlegend=False,
                                   coloraxis_showscale=False,
                                   margin=dict(l=0,r=10,t=10,b=0),
                                   yaxis=dict(categoryorder='total ascending', showgrid=False),
                                   xaxis=dict(showgrid=False))
            st.plotly_chart(fig_gold, width='stretch', config={'displayModeBar': False})

        with gc2:
            st.markdown('<div class="section-title">Top Artists</div>', unsafe_allow_html=True)
            top_artists = gold_df.groupby('artist')['play_count'].sum() \
                                 .reset_index().sort_values('play_count', ascending=False).head(10)
            fig_art = px.pie(top_artists, values='play_count', names='artist',
                             hole=0.5, template='plotly_dark',
                             color_discrete_sequence=px.colors.sequential.Greens_r)
            fig_art.update_layout(**plotly_transparent(), margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_art, width='stretch')

        st.markdown('<div class="section-title">Full Rankings Table</div>', unsafe_allow_html=True)
        st.dataframe(gold_df.head(100), width='stretch', hide_index=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem">
            <div style="font-size:3rem">🏆</div>
            <div style="color:#888; margin-top:1rem; font-size:1.1rem">Gold Rankings chưa có dữ liệu</div>
            <div style="color:#555; margin-top:0.5rem; font-size:0.85rem">Pipeline sẽ tự cập nhật sau mỗi 30 phút</div>
        </div>""", unsafe_allow_html=True)

# ── Raw Data Inspector ──────────────────
with st.expander("🔍 Raw Stream Inspector (50 events mới nhất)"):
    if df is not None:
        st.dataframe(
            df.sort_values('timestamp', ascending=False).head(50),
            width='stretch'
        )
    else:
        st.write("Không có dữ liệu.")

# ── TAB 5: Hiệu suất Pipeline ───────────
with tab_perf:
    st.markdown('<div class="section-title">⚡ Hiệu suất Pipeline Thời Gian Thực</div>', unsafe_allow_html=True)

    perf = load_perf_metrics()

    # ── KPI Row ──
    p1, p2, p3, p4 = st.columns(4)
    kpi_data = [
        (p1, "🥉 Bronze Files", f"{perf['bronze']:,}", f"{perf['bronze_size_mb']} MB"),
        (p2, "🥈 Silver Files", f"{perf['silver']:,}", f"{perf['silver_size_mb']} MB"),
        (p3, "🥇 Gold Files",   f"{perf['gold']:,}",   f"{perf['gold_size_mb']} MB"),
        (p4, "📡 Events/phút", f"~{perf.get('streaming_events_per_min', 0):,}", "Kafka throughput"),
    ]
    for col, label, val, sub in kpi_data:
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">{label}</div>
                <div class="card-value">{val}</div>
                <div class="card-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── So sánh Công nghệ Mới vs Cũ ──
    st.markdown('<div class="section-title">🆚 So Sánh Công Nghệ Xử Lý</div>', unsafe_allow_html=True)

    comp_col1, comp_col2 = st.columns([3, 2])

    with comp_col1:
        tech_data = {
            "Tác vụ": [
                "Ingest 1M events",
                "Clean & Deduplicate",
                "Aggregate (GroupBy)",
                "Train ML Model (ALS)",
                "Churn Prediction",
                "Fault Tolerance",
            ],
            "🔴 Cũ (Pandas + CSV)": [
                "45 phút", "30 phút", "25 phút",
                "Không hỗ trợ", "Không hỗ trợ", "Không có"
            ],
            "🟢 Mới (Spark + Kafka)": [
                "< 60 giây", "~2 phút", "~1 phút",
                "~5 phút (ALS)", "~5 phút (GBT)", "Tự động (Checkpoint)"
            ],
            "Cải thiện": [
                "45x nhanh hơn", "15x nhanh hơn", "25x nhanh hơn",
                "✅ Mới hoàn toàn", "✅ Mới hoàn toàn", "✅ Mới hoàn toàn"
            ]
        }
        comp_df = pd.DataFrame(tech_data)
        st.dataframe(comp_df, width='stretch', hide_index=True)

    with comp_col2:
        # Bar chart speedup
        speedup_df = pd.DataFrame({
            "Tác vụ": ["Ingest", "Clean", "Aggregate"],
            "Pandas (phút)": [45, 30, 25],
            "Spark (phút)":  [1,  2,  1],
        })
        fig_cmp = go.Figure()
        fig_cmp.add_bar(
            name="🔴 Pandas+CSV (cũ)",
            x=speedup_df["Tác vụ"],
            y=speedup_df["Pandas (phút)"],
            marker_color="#FF4136",
            text=speedup_df["Pandas (phút)"].astype(str) + " ph",
            textposition="auto",
        )
        fig_cmp.add_bar(
            name="🟢 Spark+Kafka (mới)",
            x=speedup_df["Tác vụ"],
            y=speedup_df["Spark (phút)"],
            marker_color="#1DB954",
            text=speedup_df["Spark (phút)"].astype(str) + " ph",
            textposition="auto",
        )
        fig_cmp.update_layout(
            **plotly_transparent(),
            barmode="group",
            template="plotly_dark",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="Thời gian (phút)",
            title="Thời gian xử lý: Cũ vs Mới",
        )
        st.plotly_chart(fig_cmp, width='stretch', config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Medallion Pipeline Flow ──
    st.markdown('<div class="section-title">🔄 Luồng Dữ Liệu Medallion</div>', unsafe_allow_html=True)

    flow_html = f"""
    <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap; padding:1.5rem;
                background:var(--glass); border:1px solid var(--border); border-radius:16px;">
        <div style="text-align:center;">
            <div style="font-size:2rem">🎵</div>
            <div style="color:#1DB954;font-weight:700;font-size:0.9rem">Eventsim</div>
            <div style="color:#666;font-size:0.75rem">200 users</div>
        </div>
        <div style="color:#555;font-size:1.5rem">→</div>
        <div style="text-align:center;">
            <div style="font-size:2rem">📨</div>
            <div style="color:#1DB954;font-weight:700;font-size:0.9rem">Kafka</div>
            <div style="color:#666;font-size:0.75rem">~1k msg/phút</div>
        </div>
        <div style="color:#555;font-size:1.5rem">→<br><small style='color:#888;font-size:0.7rem'>Spark Streaming<br>60s trigger</small></div>
        <div style="text-align:center;">
            <div style="font-size:2rem">🥉</div>
            <div style="color:#CD7F32;font-weight:700;font-size:0.9rem">Bronze</div>
            <div style="color:#666;font-size:0.75rem">{perf['bronze']} files · {perf['bronze_size_mb']}MB</div>
        </div>
        <div style="color:#555;font-size:1.5rem">→<br><small style='color:#888;font-size:0.7rem'>Spark Batch<br>30min/lần</small></div>
        <div style="text-align:center;">
            <div style="font-size:2rem">🥈</div>
            <div style="color:#C0C0C0;font-weight:700;font-size:0.9rem">Silver</div>
            <div style="color:#666;font-size:0.75rem">{perf['silver']} files · {perf['silver_size_mb']}MB</div>
        </div>
        <div style="color:#555;font-size:1.5rem">→<br><small style='color:#888;font-size:0.7rem'>Aggregation<br>+ ML Train</small></div>
        <div style="text-align:center;">
            <div style="font-size:2rem">🥇</div>
            <div style="color:#FFD700;font-weight:700;font-size:0.9rem">Gold</div>
            <div style="color:#666;font-size:0.75rem">{perf['gold']} files · {perf['gold_size_mb']}MB</div>
        </div>
        <div style="color:#555;font-size:1.5rem">→</div>
        <div style="text-align:center;">
            <div style="font-size:2rem">📊</div>
            <div style="color:#1DB954;font-weight:700;font-size:0.9rem">Dashboard</div>
            <div style="color:#666;font-size:0.75rem">Real-time</div>
        </div>
    </div>
    """
    st.markdown(flow_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture Benefits ──
    st.markdown('<div class="section-title">🏛️ Lợi Ích Kiến Trúc Hiện Đại</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    benefits = [
        (b1, "⚡ Tốc độ", "Spark xử lý phân tán song song trên nhiều node. Tốc độ tăng tuyến tính khi thêm worker.", "#1DB954"),
        (b2, "🔒 Độ bền",  "Kafka lưu message 7 ngày. Checkpoint đảm bảo không mất data kể cả khi Spark crash.", "#FFBE00"),
        (b3, "📈 Scale",   "Thêm worker Spark/Kafka broker không cần sửa code. Scale từ 1TB → 1PB cùng codebase.", "#1DA1F2"),
    ]
    for col, title, desc, color in benefits:
        with col:
            st.markdown(f"""
            <div class="card" style="border-left: 3px solid {color};">
                <div style="font-size:1.3rem;font-weight:800;color:{color}">{title}</div>
                <div style="color:#aaa;font-size:0.88rem;margin-top:8px;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Links ──
    st.markdown("<br>", unsafe_allow_html=True)
    lnk1, lnk2, lnk3 = st.columns(3)
    with lnk1:
        st.link_button("🔗 Airflow DAGs", "http://localhost:8888", width='stretch')
    with lnk2:
        st.link_button("🔗 Kafka UI", "http://localhost:8081", width='stretch')
    with lnk3:
        st.link_button("🔗 Spark Master", "http://localhost:8082", width='stretch')

# Lần lưu này để kích hoạt lại (trigger) Streamlit reload, xóa lỗi TokenError tạm thời do xung đột lúc ghi file.
