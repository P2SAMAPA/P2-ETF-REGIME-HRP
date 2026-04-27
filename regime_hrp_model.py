"""
Regime-aware HRP – blends covariances based on VIX‑derived regime probabilities.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from hrp_model import HRPAllocator
import config

class RegimeHRPAllocator:
    def __init__(self, linkage_method='ward'):
        self.linkage_method = linkage_method
        self.base_allocator = HRPAllocator(linkage_method=linkage_method)

    def compute_blended_covariance(self, returns: pd.DataFrame, regime_probs: dict) -> pd.DataFrame:
        """Blend covariances using regime probabilities."""
        n = len(returns)
        short_n = min(config.COV_WINDOWS['short'], n)
        medium_n = min(config.COV_WINDOWS['medium'], n)
        long_n = min(config.COV_WINDOWS['long'], n)

        short_cov = returns.iloc[-short_n:].cov().values
        medium_cov = returns.iloc[-medium_n:].cov().values
        long_cov = returns.iloc[-long_n:].cov().values

        # Map regime dict to probabilities
        p_stress = regime_probs.get('stress', 0.2)
        p_neutral = regime_probs.get('neutral', 0.6)
        p_calm = regime_probs.get('calm', 0.2)

        blended = p_stress * short_cov + p_neutral * medium_cov + p_calm * long_cov
        blended = (blended + blended.T) / 2
        np.fill_diagonal(blended, blended.diagonal() + 1e-8)
        return pd.DataFrame(blended, index=returns.columns, columns=returns.columns)

    def allocate(self, returns: pd.DataFrame, regime_probs: dict, defensive_only: bool = False) -> dict:
        """Compute HRP weights using regime‑blended covariance."""
        tickers = returns.columns.tolist()

        if defensive_only and config.USE_REGIME_GATING:
            defensive = [t for t in config.DEFENSIVE_TICKERS if t in tickers]
            if defensive:
                returns = returns[defensive]
                tickers = defensive

        if len(tickers) < 2:
            return {tickers[0]: 1.0} if tickers else {}

        blended_cov = self.compute_blended_covariance(returns, regime_probs)

        # Convert to correlation
        std = np.sqrt(np.diag(blended_cov.values))
        corr_vals = blended_cov.values / np.outer(std, std)
        corr_vals = np.clip(corr_vals, -1.0, 1.0)
        corr_vals = (corr_vals + corr_vals.T) / 2
        np.fill_diagonal(corr_vals, 1.0)

        # Distance matrix
        dist_vals = np.sqrt((1 - corr_vals) / 2)
        np.fill_diagonal(dist_vals, 0.0)
        dist_vals = (dist_vals + dist_vals.T) / 2

        # Condensed form
        n = len(tickers)
        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(dist_vals[i, j])
        condensed = np.array(condensed)

        linkage = sch.linkage(condensed, method=self.linkage_method)
        ordered_indices = sch.leaves_list(linkage)
        ordered_tickers = [tickers[i] for i in ordered_indices]

        cov_ordered = blended_cov.loc[ordered_tickers, ordered_tickers]
        weights = self.base_allocator._recursive_bisection(cov_ordered)
        return dict(zip(ordered_tickers, weights))
