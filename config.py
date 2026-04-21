"""
Configuration for P2-ETF-REGIME-HRP engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"

HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-regime-hrp-results"

# HHMM and EVT repositories (for regime signals)
HF_HHMM_REPO = "P2SAMAPA/p2-etf-hhmm-regime-results"
HF_EVT_REPO = "P2SAMAPA/p2-etf-evt-tailrisk-results"

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

# Defensive ETF list (used when tail warning active)
DEFENSIVE_TICKERS = ["TLT", "LQD", "XLP", "XLU", "XLV"]

# --- HRP Parameters ---
LINKAGE_METHOD = "ward"
MIN_OBSERVATIONS = 252

# Covariance blending windows (trading days)
COV_WINDOWS = {
    "short": 63,    # 3 months
    "medium": 252,  # 1 year
    "long": 504     # 2 years
}

# Default regime probabilities if HHMM unavailable
DEFAULT_REGIME_PROBS = [0.2, 0.6, 0.2]  # [stress, neutral, calm]

# Tail warning gate
USE_TAIL_WARNING_GATE = True

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
