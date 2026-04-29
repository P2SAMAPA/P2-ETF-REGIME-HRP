"""
Regime-aware HRP – blends covariances based on VIX‑derived regime probabilities.
Optionally returns‑tilted (Sortino) within clusters.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from hrp_model import HRPAllocator
import config

def compute_sortino(returns: np.ndarray, rf: float = 0.02) -> float:
    """Annualised Sortino ratio for a series of daily returns."""
    if len(returns) < 20:
        return 0.0
    excess = returns - (rf / 252)
    downside = excess[excess < 0].std()
    if downside == 0 or np.isnan(downside):
        return 0.0
    annual_excess = excess.mean() * 252
    annual_downside = downside * np.sqrt(252)
    return annual_excess / annual_downside


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

        p_stress = regime_probs.get('stress', 0.2)
        p_neutral = regime_probs.get('neutral', 0.6)
        p_calm = regime_probs.get('calm', 0.2)

        blended = p_stress * short_cov + p_neutral * medium_cov + p_calm * long_cov
        blended = (blended + blended.T) / 2
        np.fill_diagonal(blended, blended.diagonal() + 1e-8)
        return pd.DataFrame(blended, index=returns.columns, columns=returns.columns)

    def allocate(self, returns: pd.DataFrame, regime_probs: dict,
                 defensive_only: bool = False) -> dict:
        """Compute HRP weights using regime‑blended covariance, with optional return tilt."""
        tickers = returns.columns.tolist()

        if defensive_only and config.USE_REGIME_GATING:
            defensive = [t for t in config.DEFENSIVE_TICKERS if t in tickers]
            if defensive:
                returns = returns[defensive]
                tickers = defensive

        if len(tickers) < 2:
            return {tickers[0]: 1.0} if tickers else {}

        blended_cov = self.compute_blended_covariance(returns, regime_probs)

        # ----- if return‑tilt is enabled, compute per‑asset scores -----
        if config.USE_RETURN_TILT:
            lookback = min(config.TILT_LOOKBACK, len(returns))
            recent = returns.iloc[-lookback:]
            scores = {}
            for t in tickers:
                if config.RETURN_TILT_METRIC == "sortino":
                    scores[t] = compute_sortino(recent[t].values, config.RISK_FREE_RATE_ANNUAL)
                else:   # Sharpe
                    ret = recent[t].values
                    excess = ret - (config.RISK_FREE_RATE_ANNUAL / 252)
                    scores[t] = (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(252)

            # Shift scores to be non‑negative
            min_score = min(scores.values())
            if min_score < 0:
                for t in scores:
                    scores[t] += (-min_score + 1e-6)

            return self._allocate_with_tilt(returns, blended_cov, scores, tickers)

        # ----- standard HRP allocation (no tilt) -----
        corr_vals = blended_cov.values / np.outer(
            np.sqrt(np.diag(blended_cov)), np.sqrt(np.diag(blended_cov))
        )
        corr_vals = np.clip(corr_vals, -1.0, 1.0)
        corr_vals = (corr_vals + corr_vals.T) / 2
        np.fill_diagonal(corr_vals, 1.0)

        dist_vals = np.sqrt((1 - corr_vals) / 2)
        np.fill_diagonal(dist_vals, 0.0)
        dist_vals = (dist_vals + dist_vals.T) / 2

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

    def _allocate_with_tilt(self, returns, blended_cov, scores, tickers):
        """Custom HRP allocation that uses score/variance for leaf weights."""
        # Get correlation matrix from blended covariance
        corr = blended_cov.corr().values

        # --- FIX: enforce symmetry and clamp to [-1, 1] to avoid sqrt(negative) ---
        corr = (corr + corr.T) / 2
        corr = np.clip(corr, -1.0, 1.0)

        # Now compute distance matrix safely
        dist = np.sqrt((1 - corr) / 2)
        np.fill_diagonal(dist, 0)
        condensed = ssd.squareform(dist)
        linkage = sch.linkage(condensed, method=self.linkage_method)
        ordered_idx = sch.leaves_list(linkage)
        ordered_tickers = [tickers[i] for i in ordered_idx]

        cov_ordered = blended_cov.values[np.ix_(ordered_idx, ordered_idx)]
        score_ordered = np.array([scores[t] for t in ordered_tickers])

        weights = self._recursive_bisection_tilted(cov_ordered, score_ordered)
        return dict(zip(ordered_tickers, weights))

    def _recursive_bisection_tilted(self, cov, scores):
        n = cov.shape[0]
        if n == 1:
            return np.array([1.0])
        mid = n // 2
        left_idx = list(range(mid))
        right_idx = list(range(mid, n))
        cov_left = cov[np.ix_(left_idx, left_idx)]
        cov_right = cov[np.ix_(right_idx, right_idx)]
        scores_left = scores[left_idx]
        scores_right = scores[right_idx]

        w_left = self._tilted_leaf_weights(cov_left, scores_left)
        w_right = self._tilted_leaf_weights(cov_right, scores_right)

        var_left = np.linalg.multi_dot((w_left, cov_left, w_left))
        var_right = np.linalg.multi_dot((w_right, cov_right, w_right))
        alpha = var_right / (var_left + var_right)

        weights = np.zeros(n)
        weights[left_idx] = alpha * self._recursive_bisection_tilted(cov_left, scores_left)
        weights[right_idx] = (1 - alpha) * self._recursive_bisection_tilted(cov_right, scores_right)
        return weights

    def _tilted_leaf_weights(self, cov, scores):
        """Leaf weights proportional to score / variance."""
        diag = np.diag(cov)
        diag = np.where(diag < 1e-10, 1e-10, diag)
        raw = scores / diag
        return raw / raw.sum()
