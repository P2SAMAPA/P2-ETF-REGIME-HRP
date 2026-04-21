"""
Streamlit Dashboard for REGIME-HRP Allocator.
Displays regime-blended HRP weights.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, hf_hub_download
import json
import numpy as np
import scipy.cluster.hierarchy as sch
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Regime HRP", page_icon="🌐", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def display_allocation(weights: dict, title: str):
    if not weights:
        st.info("No weights available.")
        return
    df = pd.DataFrame(weights.items(), columns=['Ticker', 'Weight']).sort_values('Weight', ascending=False)
    fig = go.Figure(go.Pie(labels=df['Ticker'], values=df['Weight'], hole=0.4))
    fig.update_layout(title=title, height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.style.format({'Weight': '{:.2%}'}), use_container_width=True, hide_index=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
    tail = data['config'].get('tail_warning_active', False)
    if tail:
        st.sidebar.warning("⚠️ Tail warning active — defensive allocation applied")
st.sidebar.divider()
st.sidebar.markdown("### 🌐 Regime HRP Parameters")
st.sidebar.markdown(f"- Windows: {config.COV_WINDOWS}")
st.sidebar.markdown(f"- Linkage: **{config.LINKAGE_METHOD}**")

st.markdown('<div class="main-header">🌐 P2Quant Regime HRP</div>', unsafe_allow_html=True)
st.markdown('<div>Regime‑Aware Hierarchical Risk Parity – Blended Covariance Allocation</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        weights = daily['top5_weights'].get(key, {})
        display_allocation(weights, f"{key} – Top 5 Allocation")
