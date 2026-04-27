"""
Configuration for P2-ETF-REGIME-HRP engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-regime-hrp-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# Defensive ETFs (only when stress gating is active)
DEFENSIVE_TICKERS = ["TLT", "LQD", "XLP", "XLU", "XLV"]

# --- HRP Parameters ---
LINKAGE_METHOD = "ward"
MIN_OBSERVATIONS = 252
COV_WINDOWS = {"short": 63, "medium": 252, "long": 504}

# --- Regime Classification (self‑contained VIX‑based) ---
VIX_STRESS_THRESHOLD = 28.0
VIX_CALM_THRESHOLD = 16.0
USE_REGIME_GATING = True              # restrict to defensive ETFs in stress

# --- Return‑Tilt Parameters ---
USE_RETURN_TILT = True                # enable Sortino‑based weighting
RETURN_TILT_METRIC = "sortino"        # "sortino" or "sharpe"
TILT_LOOKBACK = 126                   # days for computing the metric
RISK_FREE_RATE_ANNUAL = 0.02          # for Sortino/Sharpe

# --- Training Modes ---
DAILY_LOOKBACK = 504
GLOBAL_TRAIN_START = "2008-01-01"
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
