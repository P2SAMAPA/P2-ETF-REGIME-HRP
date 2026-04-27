"""
Streamlit Dashboard for REGIME-HRP Engine.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Regime HRP", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .stress-badge { background: #dc3545; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.9rem; }
    .neutral-badge { background: #ffc107; color: black; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.9rem; }
    .calm-badge { background: #28a745; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.9rem; }
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

def regime_badge(regime):
    if regime.get('stress', 0) > 0.5:
        return '<span class="stress-badge">STRESS</span>'
    elif regime.get('calm', 0) > 0.5:
        return '<span class="calm-badge">CALM</span>'
    else:
        return '<span class="neutral-badge">NEUTRAL</span>'

def display_allocation(weights, title):
    if not weights:
        st.info("No weights available.")
        return
    df = pd.DataFrame(weights.items(), columns=['Ticker', 'Weight'])
    df = df.sort_values('Weight', ascending=False)
    
    fig = go.Figure(go.Pie(labels=df['Ticker'], values=df['Weight'], hole=0.4))
    fig.update_layout(title=title, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Format weights as percentages for display
    df_display = df.copy()
    df_display['Weight'] = df_display['Weight'].apply(lambda x: f"{x:.2%}")
    st.dataframe(df_display, use_container_width=True, hide_index=True)

def render_mode_tab(mode_data, mode_name):
    if not mode_data:
        st.warning(f"No {mode_name} data.")
        return
    regime = mode_data.get('regime', {})
    defensive = mode_data.get('defensive_mode', False)

    st.markdown(f"### {mode_name}")
    st.markdown(f"Regime: {regime_badge(regime)} {'🛡️ Defensive' if defensive else ''}", unsafe_allow_html=True)

    top5 = mode_data.get('top5_weights', {})
    if top5:
        ticker = list(top5.keys())[0]
        weight = top5[ticker]
        st.markdown(f"""
        <div class="hero-card">
            <div style="font-size: 1.2rem; opacity: 0.8;">📊 TOP ALLOCATION</div>
            <div class="hero-ticker">{ticker}</div>
            <div>Weight: {weight:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    display_allocation(top5, f"{mode_name} – Top 5 Allocation")

def render_shrinking_tab(shrinking_data):
    if not shrinking_data:
        st.warning("No shrinking data.")
        return
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">🔄 SHRINKING CONSENSUS</div>
        <div class="hero-ticker">{shrinking_data['ticker']}</div>
        <div>{shrinking_data['conviction']:.0f}% conviction · {shrinking_data['num_windows']} windows</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📋 All Windows"):
        rows = []
        for w in shrinking_data.get('windows', []):
            rows.append({
                'Window': f"{w['window_start']}-{w['window_end']}",
                'Top Pick': w['top_ticker'],
                'Regime': w['regime'].get('stress', 0) > 0.5 and 'Stress' or
                          w['regime'].get('calm', 0) > 0.5 and 'Calm' or 'Neutral'
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">📊 P2Quant Regime HRP</div>', unsafe_allow_html=True)
st.markdown('<div>Regime‑Aware Hierarchical Risk Parity – VIX‑Based Regime Blending</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

universes_data = data.get('universes', {})
tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, keys):
    uni = universes_data.get(key, {})
    if not uni:
        with tab:
            st.info(f"No data for {key}.")
        continue
    with tab:
        d, g, s = st.tabs(["📅 Daily (504d)", "🌍 Global (2008‑YTD)", "🔄 Shrinking Consensus"])
        with d:
            render_mode_tab(uni.get('daily'), "Daily")
        with g:
            render_mode_tab(uni.get('global'), "Global")
        with s:
            render_shrinking_tab(uni.get('shrinking'))
