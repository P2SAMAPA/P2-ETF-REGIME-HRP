"""
Main training script – Daily, Global, and Shrinking modes.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from regime_hrp_model import RegimeHRPAllocator
import push_results

def get_top_n_weights(weights, n=5):
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:n]
    total = sum(w for _, w in sorted_items)
    return {t: w / total for t, w in sorted_items}

def run_allocation(returns, macro_df=None):
    """Run regime HRP allocation on a data slice."""
    if len(returns) < config.MIN_OBSERVATIONS:
        return None

    allocator = RegimeHRPAllocator(linkage_method=config.LINKAGE_METHOD)

    # Get regime classification
    if macro_df is not None and 'VIX' in macro_df.columns:
        regime = data_manager.get_vix_based_regime(macro_df)
    elif 'VIX' in returns.columns:
        regime = data_manager.get_vix_based_regime(returns)
    else:
        regime = {'stress': 0.2, 'neutral': 0.6, 'calm': 0.2}

    # Defensive gating based on regime
    defensive_only = regime['stress'] > 0.5 and config.USE_REGIME_GATING

    # Filter out macro columns for HRP
    etf_cols = [c for c in returns.columns if c in config.ALL_TICKERS]
    returns_etf = returns[etf_cols]

    weights = allocator.allocate(returns_etf, regime, defensive_only=defensive_only)
    top5 = get_top_n_weights(weights, n=5)

    return {
        'full_weights': weights,
        'top5_weights': top5,
        'regime': regime,
        'defensive_mode': defensive_only,
        'training_start': str(returns.index[0].date()),
        'training_end': str(returns.index[-1].date())
    }


def main():
    import os
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return

    df_master = data_manager.load_master_data()
    df_master['Date'] = pd.to_datetime(df_master['Date'])

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        returns_all = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns_all) < config.MIN_OBSERVATIONS:
            continue

        # Add VIX column for regime detection (if available)
        if 'VIX' in df_master.columns:
            returns_all = returns_all.join(df_master[['Date','VIX']].set_index('Date').loc[returns_all.index])

        universe_out = {}

        # Daily (504d)
        daily_ret = returns_all.iloc[-config.DAILY_LOOKBACK:]
        daily_out = run_allocation(daily_ret, daily_ret)
        if daily_out:
            universe_out['daily'] = daily_out
            print(f"  Daily top: {list(daily_out['top5_weights'].keys())[0]}")

        # Global (2008‑YTD)
        global_mask = df_master['Date'] >= config.GLOBAL_TRAIN_START
        global_ret = returns_all.loc[returns_all.index.isin(df_master[global_mask]['Date'])]
        global_out = run_allocation(global_ret, global_ret)
        if global_out:
            universe_out['global'] = global_out
            print(f"  Global top: {list(global_out['top5_weights'].keys())[0]}")

        # Shrinking Windows
        shrinking_windows = []
        for start_year in config.SHRINKING_WINDOW_START_YEARS:
            sd = pd.Timestamp(f"{start_year}-01-01")
            ed = sd + pd.Timedelta(days=config.DAILY_LOOKBACK * 2)
            mask = (df_master['Date'] >= sd) & (df_master['Date'] <= ed)
            ret_win = returns_all.loc[returns_all.index.isin(df_master[mask]['Date'])]
            if len(ret_win) < config.MIN_OBSERVATIONS:
                continue
            out = run_allocation(ret_win.iloc[:config.DAILY_LOOKBACK], ret_win.iloc[:config.DAILY_LOOKBACK])
            if out:
                shrinking_windows.append({
                    'window_start': start_year,
                    'window_end': start_year + 2,
                    'top_ticker': list(out['top5_weights'].keys())[0],
                    'regime': out['regime']
                })

        if shrinking_windows:
            # Consensus across windows
            vote = {}
            for w in shrinking_windows:
                vote[w['top_ticker']] = vote.get(w['top_ticker'], 0) + 1
            pick = max(vote, key=vote.get)
            conviction = vote[pick] / len(shrinking_windows) * 100
            universe_out['shrinking'] = {
                'ticker': pick,
                'conviction': conviction,
                'num_windows': len(shrinking_windows),
                'windows': shrinking_windows
            }

        all_results[universe_name] = universe_out

    push_results.push_daily_result({"run_date": config.TODAY, "universes": all_results})
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
