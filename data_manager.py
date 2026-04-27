"""
Data loading and preprocessing for REGIME-HRP engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Prepare wide‑format log returns."""
    available_tickers = [t for t in tickers if t in df_wide.columns]
    df_long = pd.melt(
        df_wide, id_vars=['Date'], value_vars=available_tickers,
        var_name='ticker', value_name='price'
    )
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    return df_long.pivot(index='Date', columns='ticker', values='log_return')[available_tickers].dropna()

def get_vix_based_regime(returns: pd.DataFrame, vix_col: str = 'VIX') -> dict:
    """
    Classify the current market regime based on VIX level.
    Returns blending probabilities [stress, neutral, calm].
    """
    if vix_col not in returns.columns:
        return {'stress': 0.2, 'neutral': 0.6, 'calm': 0.2}

    latest_vix = returns[vix_col].iloc[-1]
    if latest_vix >= config.VIX_STRESS_THRESHOLD:
        probs = {'stress': 0.7, 'neutral': 0.2, 'calm': 0.1}
    elif latest_vix <= config.VIX_CALM_THRESHOLD:
        probs = {'stress': 0.1, 'neutral': 0.2, 'calm': 0.7}
    else:
        probs = {'stress': 0.2, 'neutral': 0.6, 'calm': 0.2}

    probs['stress'] = float(probs['stress'])
    probs['neutral'] = float(probs['neutral'])
    probs['calm'] = float(probs['calm'])
    return probs
