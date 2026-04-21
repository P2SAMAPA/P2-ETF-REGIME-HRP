"""
Regime-aware HRP: blends covariances based on HHMM regime probabilities.
"""

import numpy as np
import pandas as pd
from hrp_model import HRPAllocator
import config

class RegimeHRPAllocator:
    def __init__(self, linkage_method='ward'):
        self.linkage_method = linkage_method
        self.base_allocator = HRPAllocator(linkage_method=linkage_method)

    def compute_blended_covariance(self, returns: pd.DataFrame, regime_probs: dict) -> pd.DataFrame:
        """
        Blend covariances from short, medium, and long windows.
        regime_probs: {'macro_prob': float, 'sector_prob': float, ...}
        We map the first three HHMM probabilities to [stress, neutral, calm].
        """
        windows = config.COV_WINDOWS
        short_cov = returns.iloc[-windows['short']:].cov()
        medium_cov = returns.iloc[-windows['medium']:].cov()
        long_cov = returns.iloc[-windows['long']:].cov()

        # Extract probabilities (default if missing)
        probs = [
            regime_probs.get('macro_prob', config.DEFAULT_REGIME_PROBS[0]),
            regime_probs.get('sector_prob', config.DEFAULT_REGIME_PROBS[1]),
            regime_probs.get('etf_prob', config.DEFAULT_REGIME_PROBS[2])
        ]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1/3, 1/3, 1/3]

        blended = probs[0] * short_cov + probs[1] * medium_cov + probs[2] * long_cov
        return blended

    def allocate(self, returns: pd.DataFrame, regime_probs: dict,
                 defensive_only: bool = False) -> dict:
        """
        Compute HRP weights using regime-blended covariance.
        If defensive_only, restrict to defensive tickers.
        """
        tickers = returns.columns.tolist()
        if defensive_only:
            defensive = [t for t in config.DEFENSIVE_TICKERS if t in tickers]
            if defensive:
                returns = returns[defensive]
                tickers = defensive

        if returns.shape[1] < 2:
            return {returns.columns[0]: 1.0} if returns.shape[1] == 1 else {}

        blended_cov = self.compute_blended_covariance(returns, regime_probs)
        # Convert covariance back to returns-like data for HRP (it only needs covariance)
        # HRP uses returns to compute cov; we bypass by overriding allocate
        # Simpler: pass returns but use blended_cov inside a custom allocator
        allocator = HRPAllocator(linkage_method=self.linkage_method)
        allocator.original_tickers = tickers

        corr = blended_cov.corr()
        dist = scipy.spatial.distance.squareform(((1 - corr) / 2) ** 0.5)
        allocator.linkage = scipy.cluster.hierarchy.linkage(dist, method=self.linkage_method)

        ordered_indices = scipy.cluster.hierarchy.leaves_list(allocator.linkage)
        ordered_tickers = [tickers[i] for i in ordered_indices]

        cov_ordered = blended_cov.loc[ordered_tickers, ordered_tickers]
        weights = allocator._recursive_bisection(cov_ordered)
        return dict(zip(ordered_tickers, weights))
