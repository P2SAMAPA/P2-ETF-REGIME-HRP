"""
Regime-aware HRP: blends covariances based on HHMM regime probabilities.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from hrp_model import HRPAllocator
import config

class RegimeHRPAllocator:
    def __init__(self, linkage_method='ward'):
        self.linkage_method = linkage_method
        self.base_allocator = HRPAllocator(linkage_method=linkage_method)

    def compute_blended_covariance(self, returns: pd.DataFrame, regime_probs: dict) -> np.ndarray:
        """
        Blend covariances from short, medium, and long windows.
        Returns a plain NumPy array (symmetric, positive semi-definite).
        """
        windows = config.COV_WINDOWS
        
        n = len(returns)
        short_n = min(windows['short'], n)
        medium_n = min(windows['medium'], n)
        long_n = min(windows['long'], n)
        
        short_cov = returns.iloc[-short_n:].cov().values
        medium_cov = returns.iloc[-medium_n:].cov().values
        long_cov = returns.iloc[-long_n:].cov().values

        probs = [
            regime_probs.get('macro_prob', config.DEFAULT_REGIME_PROBS[0]),
            regime_probs.get('sector_prob', config.DEFAULT_REGIME_PROBS[1]),
            regime_probs.get('etf_prob', config.DEFAULT_REGIME_PROBS[2])
        ]
        total = sum(probs) if probs else 0
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1/3, 1/3, 1/3]

        blended = probs[0] * short_cov + probs[1] * medium_cov + probs[2] * long_cov
        
        # Symmetrize
        blended = (blended + blended.T) / 2
        
        # Ridge for numerical stability
        np.fill_diagonal(blended, blended.diagonal() + 1e-8)
        
        return blended

    def allocate(self, returns: pd.DataFrame, regime_probs: dict,
                 defensive_only: bool = False) -> dict:
        """
        Compute HRP weights using regime-blended covariance.
        """
        tickers = returns.columns.tolist()
        
        if defensive_only:
            defensive = [t for t in config.DEFENSIVE_TICKERS if t in tickers]
            if defensive:
                returns = returns[defensive]
                tickers = defensive

        n_assets = returns.shape[1]
        if n_assets < 2:
            if n_assets == 1:
                return {returns.columns[0]: 1.0}
            return {}

        # Blended covariance as numpy array
        cov = self.compute_blended_covariance(returns, regime_probs)
        
        # Convert to correlation
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

        # Distance matrix
        dist = np.sqrt((1 - corr) / 2)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2

        # Condensed distance for linkage
        condensed = []
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                condensed.append(dist[i, j])
        condensed = np.array(condensed)

        linkage = sch.linkage(condensed, method=self.linkage_method)
        ordered_indices = sch.leaves_list(linkage)
        ordered_tickers = [tickers[i] for i in ordered_indices]

        # Build ordered covariance DataFrame for HRP (needed by base allocator)
        cov_ordered = pd.DataFrame(cov, index=tickers, columns=tickers).loc[ordered_tickers, ordered_tickers]
        
        allocator = HRPAllocator(linkage_method=self.linkage_method)
        allocator.original_tickers = ordered_tickers
        allocator.linkage = linkage
        weights = allocator._recursive_bisection(cov_ordered)
        
        return dict(zip(ordered_tickers, weights))
