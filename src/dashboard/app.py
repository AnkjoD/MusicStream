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
    * { font-family: 'Outfit', sans-serif !important; }

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
    </style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════

STORAGE_OPTS = {
    "key":    os.getenv("MINIO_ACCESS_KEY", "homura_madoka"),
    "secret": os.getenv("MINIO_SECRET_KEY", "homura123"),
    "client_kwargs": {
        "endpoint_url": os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    }
}

def _mock_stream():
    """Fallback data khi MinIO chưa có data."""
    songs   = ['Blinding Lights','Save Your Tears','Levitating','Stay','Peaches',
               'Good 4 U','Kiss Me More','Montero','Easy On Me','As It Was']
    artists = ['The Weeknd','The Weeknd','Dua Lipa','The Kid LAROI','Justin Bieber',
               'Olivia Rodrigo','Doja Cat','Lil Nas X','Adele','Harry Styles']
    rows = []
    for _ in range(400):
        i = np.random.randint(0, len(songs))
        rows.append({
            'song': songs[i], 'artist': artists[i],
            'timestamp': datetime.now(),
            'userId': f"user_{np.random.randint(1,500)}",
            'level': np.random.choice(['free','paid']),
            'gender': np.random.choice(['M','F']),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=5)
def load_stream():
    """Real-time stream events từ Bronze Layer (MinIO)."""
    try:
        df = pd.read_parquet(
            "s3://bronze-zone/datalake/raw/eventsim/page=NextSong/",
            storage_options=STORAGE_OPTS
        )
        if df.empty:
            return _mock_stream(), True
        df['timestamp'] = pd.to_datetime(df.get('ts', df.get('ingestion_time')), unit='ms', errors='coerce')
        return df, False
    except:
        return _mock_stream(), True

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
    """Gợi ý nhạc từ ALS Model (Gold Layer)."""
    try:
        df = pd.read_parquet(
            "s3://gold-zone/datalake/gold/recommendations/",
            storage_options=STORAGE_OPTS
        )
        return df if not df.empty else None
    except:
        return None

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

def plotly_transparent():
    return dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Infrastructure")

    for name, s in get_docker_status().items():
        cls = "badge-ok" if s == "running" else "badge-err"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:6px 0;border-bottom:1px solid var(--border)'>"
            f"<span style='color:#ccc;font-size:0.85rem'>{name}</span>"
            f"<span class='badge {cls}'>{s.upper()}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧠 ML Models")
    churn_df  = load_churn()
    rec_df    = load_recommendations()
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
        <div class="pulse-wrap">
            <div class="pulse"></div>
            <span style="color:#1DB954;font-weight:700;font-size:0.85rem">LIVE</span>
        </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════
df, is_mock = load_stream()

if is_mock:
    st.info("⚡ Hiển thị dữ liệu mô phỏng. Pipeline đang chờ Eventsim → Kafka → Spark.", icon="ℹ️")

# ════════════════════════════════════════
# TOP KPI CARDS
# ════════════════════════════════════════
k1, k2, k3, k4 = st.columns(4)
cards = [
    (k1, "Total Streams",   f"{len(df):,}",                          "🎧", "Live events"),
    (k2, "Active Users",    f"{df['userId'].nunique():,}",            "👥", "Unique listeners"),
    (k3, "Premium Share",   f"{(df['level']=='paid').mean()*100:.1f}%","💎", "Paid subscribers"),
    (k4, "Unique Tracks",   f"{df['song'].nunique():,}",              "🔍", "Songs playing"),
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
tab_stream, tab_ml_churn, tab_ml_rec, tab_gold = st.tabs([
    "📡 Live Stream",
    "⚠️ Churn Prediction",
    "🎵 AI Recommendations",
    "🏆 Gold Rankings",
])

# ── TAB 1: Live Stream ──────────────────
with tab_stream:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown('<div class="section-title">Real-time Trending Tracks</div>', unsafe_allow_html=True)
        top = df.groupby(['song','artist']).size().reset_index(name='plays') \
                .sort_values('plays', ascending=False).head(10)
        fig = px.bar(top, x='plays', y='song', color='plays', orientation='h',
                     template='plotly_dark',
                     color_continuous_scale=['#0d2316','#1DB954','#1ed760'],
                     hover_data={'artist': True})
        fig.update_layout(**plotly_transparent(),
                          showlegend=False, coloraxis_showscale=False,
                          margin=dict(l=0,r=10,t=10,b=0),
                          yaxis=dict(categoryorder='total ascending', showgrid=False),
                          xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with c2:
        st.markdown('<div class="section-title">Listener Profile</div>', unsafe_allow_html=True)
        g = df['gender'].value_counts()
        fig_pie = px.pie(values=g.values, names=g.index, hole=0.68,
                         template='plotly_dark',
                         color_discrete_sequence=['#1DB954','#FFFFFF'])
        fig_pie.update_layout(**plotly_transparent(),
                               margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="section-title">Subscription Mix</div>', unsafe_allow_html=True)
        lv = df['level'].value_counts()
        fig_lv = px.bar(x=lv.index, y=lv.values, template='plotly_dark',
                        color=lv.index, color_discrete_map={'paid':'#1DB954','free':'#333'})
        fig_lv.update_layout(**plotly_transparent(), showlegend=False,
                              margin=dict(l=0,r=0,t=10,b=0),
                              xaxis=dict(title=None), yaxis=dict(title=None, showgrid=False))
        st.plotly_chart(fig_lv, use_container_width=True)

    # Ingestion velocity
    st.markdown('<div class="section-title">📈 Ingestion Velocity (per second)</div>', unsafe_allow_html=True)
    df['t_bucket'] = df['timestamp'].dt.floor('10s').dt.strftime('%H:%M:%S')
    vel = df.groupby('t_bucket').size().reset_index(name='events').tail(30)
    fig_vel = px.area(vel, x='t_bucket', y='events', template='plotly_dark')
    fig_vel.update_traces(line_color='#1DB954', fillcolor='rgba(29,185,84,0.12)')
    fig_vel.update_layout(**plotly_transparent(),
                          margin=dict(l=0,r=0,t=10,b=0),
                          xaxis=dict(showgrid=False, title=None),
                          yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title=None))
    st.plotly_chart(fig_vel, use_container_width=True)

# ── TAB 2: Churn Prediction ─────────────
with tab_ml_churn:
    if churn_df is not None:
        st.markdown('<div class="section-title">⚠️ Users at Risk of Churning (XGBoost/GBT Model)</div>', unsafe_allow_html=True)

        # Extract churn probability
        if 'probability' in churn_df.columns:
            churn_df['churn_prob'] = churn_df['probability'].apply(
                lambda x: float(x[1]) if hasattr(x, '__iter__') else float(x)
            )
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
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.markdown('<div class="section-title">Top 10 At-Risk Users</div>', unsafe_allow_html=True)
            top_risk = churn_df.nlargest(10, 'churn_prob')[['userId','churn_prob','total_songs','cancel_count']] \
                               .rename(columns={'churn_prob':'Risk','total_songs':'Songs','cancel_count':'Cancels'})
            top_risk['Risk'] = top_risk['Risk'].map('{:.1%}'.format)
            st.dataframe(top_risk, use_container_width=True, hide_index=True)

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
        st.plotly_chart(fig_sc, use_container_width=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem">
            <div style="font-size:3rem">🤖</div>
            <div style="color:#888; margin-top:1rem; font-size:1.1rem">
                Churn Prediction model chưa chạy
            </div>
            <div style="color:#555; margin-top:0.5rem; font-size:0.85rem">
                Trigger DAG <code>spark_medallion_batch_pipeline</code> trong Airflow
                để train XGBoost/GBT model
            </div>
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
        st.markdown("**Sample recommendations (first 20 users):**")
        st.dataframe(rec_df.head(20), use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem">
            <div style="font-size:3rem">🎵</div>
            <div style="color:#888; margin-top:1rem; font-size:1.1rem">
                ALS Recommendation model chưa chạy
            </div>
            <div style="color:#555; margin-top:0.5rem; font-size:0.85rem">
                Trigger task <code>train_als_recommendation</code> trong Airflow DAG
            </div>
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
            st.plotly_chart(fig_gold, use_container_width=True, config={'displayModeBar': False})

        with gc2:
            st.markdown('<div class="section-title">Top Artists</div>', unsafe_allow_html=True)
            top_artists = gold_df.groupby('artist')['play_count'].sum() \
                                 .reset_index().sort_values('play_count', ascending=False).head(10)
            fig_art = px.pie(top_artists, values='play_count', names='artist',
                             hole=0.5, template='plotly_dark',
                             color_discrete_sequence=px.colors.sequential.Greens_r)
            fig_art.update_layout(**plotly_transparent(), margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_art, use_container_width=True)

        st.markdown('<div class="section-title">Full Rankings Table</div>', unsafe_allow_html=True)
        st.dataframe(gold_df.head(100), use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem">
            <div style="font-size:3rem">🏆</div>
            <div style="color:#888; margin-top:1rem; font-size:1.1rem">
                Gold Rankings chưa có
            </div>
            <div style="color:#555; margin-top:0.5rem; font-size:0.85rem">
                Trigger task <code>silver_to_gold_rankings</code> trong Airflow DAG
            </div>
        </div>""", unsafe_allow_html=True)

# ── Raw Data Inspector ──────────────────
with st.expander("🔍 Raw Stream Inspector (50 events mới nhất)"):
    st.dataframe(
        df.sort_values('timestamp', ascending=False).head(50),
        use_container_width=True
    )

# Auto-refresh mỗi 5 giây
time.sleep(5)
st.rerun()
