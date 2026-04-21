"""
Main training script for REGIME-HRP engine.
Blends covariances using HHMM regime probabilities and computes HRP weights.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from regime_hrp_model import RegimeHRPAllocator
import push_results

def get_top_n_weights(weights_dict: dict, n: int = 5) -> dict:
    sorted_items = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    top = dict(sorted_items)
    total = sum(top.values())
    return {k: v / total for k, v in top.items()}

def run_regime_hrp():
    print(f"=== P2-ETF-REGIME-HRP Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master['Date'] = pd.to_datetime(df_master['Date'])
    df_master = df_master.sort_values('Date')

    # Load external regime signals
    hhmm_regime = data_manager.load_hhmm_regime_probs()
    tail_warning = data_manager.load_evt_tail_warning() if config.USE_TAIL_WARNING_GATE else False
    print(f"HHMM regime: {hhmm_regime}")
    print(f"Tail warning active: {tail_warning}")

    allocator = RegimeHRPAllocator(linkage_method=config.LINKAGE_METHOD)

    all_weights = {}
    top5_weights = {}
    cluster_info = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        weights = allocator.allocate(returns, hhmm_regime, defensive_only=tail_warning)
        all_weights[universe_name] = weights
        top5 = get_top_n_weights(weights, n=5)
        top5_weights[universe_name] = top5

        linkage, original_tickers = allocator.base_allocator.get_linkage_and_labels()
        if linkage is not None and original_tickers is not None:
            cluster_info[universe_name] = {
                "linkage": linkage.tolist(),
                "original_tickers": original_tickers
            }

        print(f"  Top 5: {list(top5.keys())}")

    # Shrinking windows (simplified)
    shrinking_results = {}
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{start_year+2}"
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            continue

        window_top = {}
        for universe_name, tickers in config.UNIVERSES.items():
            returns_win = data_manager.prepare_returns_matrix(df_window, tickers)
            if len(returns_win) < config.MIN_OBSERVATIONS:
                continue
            weights_win = allocator.allocate(returns_win, hhmm_regime, defensive_only=False)
            top = get_top_n_weights(weights_win, n=1)
            if top:
                window_top[universe_name] = list(top.keys())[0]

        shrinking_results[window_label] = {
            'start_year': start_year,
            'top_pick': window_top
        }

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "cov_windows": config.COV_WINDOWS,
            "linkage_method": config.LINKAGE_METHOD,
            "tail_warning_active": tail_warning
        },
        "daily_trading": {
            "full_weights": all_weights,
            "top5_weights": top5_weights,
            "cluster_info": cluster_info
        },
        "shrinking_windows": shrinking_results
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_regime_hrp()
