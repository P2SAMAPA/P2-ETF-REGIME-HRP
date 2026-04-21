"""
Hierarchical Risk Parity (HRP) allocation model.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from typing import Dict, List, Tuple

class HRPAllocator:
    def __init__(self, linkage_method: str = 'ward'):
        self.linkage_method = linkage_method
        self.linkage = None
        self.original_tickers = None

    def allocate(self, returns: pd.DataFrame) -> Dict[str, float]:
        if returns.shape[1] < 2:
            return {returns.columns[0]: 1.0}

        self.original_tickers = returns.columns.tolist()
        cov = returns.cov()
        corr = returns.corr()
        dist = ssd.squareform(((1 - corr) / 2) ** 0.5)
        self.linkage = sch.linkage(dist, method=self.linkage_method)

        ordered_indices = sch.leaves_list(self.linkage)
        ordered_tickers = [self.original_tickers[i] for i in ordered_indices]

        cov_ordered = cov.loc[ordered_tickers, ordered_tickers]
        weights = self._recursive_bisection(cov_ordered)
        return dict(zip(ordered_tickers, weights))

    def _recursive_bisection(self, cov: pd.DataFrame) -> np.ndarray:
        n = cov.shape[0]
        if n == 1:
            return np.array([1.0])
        mid = n // 2
        left_idx, right_idx = list(range(mid)), list(range(mid, n))
        cov_left, cov_right = cov.iloc[left_idx, left_idx], cov.iloc[right_idx, right_idx]
        w_left = self._inverse_variance_weights(cov_left)
        w_right = self._inverse_variance_weights(cov_right)
        var_left = np.linalg.multi_dot((w_left, cov_left.values, w_left))
        var_right = np.linalg.multi_dot((w_right, cov_right.values, w_right))
        alpha = var_right / (var_left + var_right)
        weights = np.zeros(n)
        weights[left_idx] = alpha * self._recursive_bisection(cov_left)
        weights[right_idx] = (1 - alpha) * self._recursive_bisection(cov_right)
        return weights

    def _inverse_variance_weights(self, cov: pd.DataFrame) -> np.ndarray:
        diag = np.diag(cov)
        diag = np.where(diag < 1e-10, 1e-10, diag)
        ivp = 1.0 / diag
        return ivp / ivp.sum()

    def get_linkage_and_labels(self) -> Tuple[np.ndarray, List[str]]:
        return self.linkage, self.original_tickers
