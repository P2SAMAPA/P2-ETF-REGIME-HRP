# P2-ETF-REGIME-HRP

**Regime‑Aware Hierarchical Risk Parity – Dynamic Covariance Blending & Tail‑Risk Filtering**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-REGIME-HRP/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-REGIME-HRP/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--regime--hrp--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-regime-hrp-results)

## Overview

`P2-ETF-REGIME-HRP` extends standard Hierarchical Risk Parity by blending covariance matrices from multiple lookback windows (short, medium, long) weighted by **HHMM regime probabilities**. When EVT tail warnings are active, the universe is restricted to defensive ETFs. The result is a robust, forward‑looking allocation that adapts to changing market conditions.

## Methodology

1. **Regime Probabilities**: Fetched daily from `P2-ETF-HHMM-REGIME`.
2. **Tail Warning**: Fetched from `P2-ETF-EVT-TAILRISK`; if active, universe filtered to defensive tickers.
3. **Covariance Blending**: `Cov_blended = p_stress * Cov_63d + p_neutral * Cov_252d + p_calm * Cov_504d`
4. **HRP Allocation**: Recursive bisection on blended covariance.
5. **Top 5 Weights**: Highest‑weighted ETFs displayed.

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
