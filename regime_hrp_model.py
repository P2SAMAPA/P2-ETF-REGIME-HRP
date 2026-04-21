"""
Regime-aware HRP: blends covariances based on HHMM regime probabilities.
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
        """
        Blend covariances from short, medium, and long windows.
        """
        windows = config.COV_WINDOWS
        
        # Ensure we have enough data for each window
        n = len(returns)
        short_n = min(windows['short'], n)
        medium_n = min(windows['medium'], n)
        long_n = min(windows['long'], n)
        
        short_cov = returns.iloc[-short_n:].cov()
        medium_cov = returns.iloc[-medium_n:].cov()
        long_cov = returns.iloc[-long_n:].cov()

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
        
        # Ensure symmetry (fix floating-point asymmetries)
        blended = (blended + blended.T) / 2
        
        # Add small ridge for numerical stability
        np.fill_diagonal(blended.values, blended.values.diagonal() + 1e-8)
        
        return blended

    def allocate(self, returns: pd.DataFrame, regime_probs: dict,
                 defensive_only: bool = False) -> dict:
        """
        Compute HRP weights using regime-blended covariance.
        """
        tickers = returns.columns.tolist()
        original_tickers = tickers.copy()
        
        if defensive_only:
            defensive = [t for t in config.DEFENSIVE_TICKERS if t in tickers]
            if defensive:
                returns = returns[defensive]
                tickers = defensive

        if returns.shape[1] < 2:
            if returns.shape[1] == 1:
                return {returns.columns[0]: 1.0}
            return {}

        blended_cov = self.compute_blended_covariance(returns, regime_probs)
        allocator = HRPAllocator(linkage_method=self.linkage_method)
        allocator.original_tickers = tickers

        # Convert covariance to correlation
        std = np.sqrt(np.diag(blended_cov))
        corr = blended_cov / np.outer(std, std)
        
        # Clip to valid range and ensure symmetry
        corr = np.clip(corr, -1.0, 1.0)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

        # Distance matrix
        dist = np.sqrt((1 - corr) / 2)
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2  # ensure symmetry
        
        # Convert to condensed form
        try:
            condensed_dist = ssd.squareform(dist, checks=False)
        except:
            # Fallback: manually compute condensed
            n = len(tickers)
            condensed_dist = []
            for i in range(n):
                for j in range(i+1, n):
                    condensed_dist.append(dist[i, j])
            condensed_dist = np.array(condensed_dist)

        allocator.linkage = sch.linkage(condensed_dist, method=self.linkage_method)

        ordered_indices = sch.leaves_list(allocator.linkage)
        ordered_tickers = [tickers[i] for i in ordered_indices]

        cov_ordered = blended_cov.loc[ordered_tickers, ordered_tickers]
        weights = allocator._recursive_bisection(cov_ordered)
        
        # Map back to original ticker list (if defensive filtered, only those appear)
        return dict(zip(ordered_tickers, weights))
