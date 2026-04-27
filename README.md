# P2-ETF-REGIME-HRP

**Regime‑Aware Hierarchical Risk Parity – VIX‑Based Covariance Blending**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-REGIME-HRP/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-REGIME-HRP/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--regime--hrp--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-regime-hrp-results)

## Overview

`P2-ETF-REGIME-HRP` extends Hierarchical Risk Parity by blending covariance matrices from three lookback windows (short 63d, medium 252d, long 504d) weighted by **VIX‑derived regime probabilities**. During stress regimes (VIX ≥ 28), the universe is restricted to defensive ETFs. The engine is fully self‑contained—no external HHMM or EVT dependencies.

## Methodology

1. **Regime Classification** – VIX level determines blending weights:
   - Stress (VIX ≥ 28): 70% short, 20% medium, 10% long
   - Neutral (16 < VIX < 28): 20% short, 60% medium, 20% long
   - Calm (VIX ≤ 16): 10% short, 20% medium, 70% long
2. **Covariance Blending** – Weighted average of three covariance matrices.
3. **HRP Allocation** – Standard hierarchical risk parity on the blended covariance.
4. **Defensive Gating** – During stress, allocation restricted to TLT, LQD, XLP, XLU, XLV.
5. **Three Training Modes** – Daily, Global, Shrinking Windows Consensus.

## Universe

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
Dashboard
Three sub‑tabs per universe: Daily, Global, Shrinking Consensus.

Hero card with top allocation and regime badge (Stress/Neutral/Calm).

Pie chart and weights table.
