from __future__ import annotations
import math
import numpy as np
from numba import cuda, float32, int32, njit, prange
import warnings
from numba.core.errors import NumbaPerformanceWarning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from fast_relief.MultiSURF import _compute_ranges

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


TPB = 64              # Threads Per Block
MAX_F_TILE = 1024     # Features loaded per shared-memory tile


@njit(parallel=True, fastmath=True)
def _compute_surf_threshold(X, recip_full):
    n_samples, n_features = X.shape
    total = 0.0
    for i in prange(n_samples-1):
        for j in range(i+1, n_samples):
            d = 0.0
            for f in range(n_features):
                d += abs(X[i,f] - X[j,f]) * recip_full[f]
            d /= n_features
            total += d
    num_pairs = n_samples*(n_samples-1)/2
    return total / num_pairs if num_pairs else 0.0


@cuda.jit
def _surf_score_kernel_idx(X, y, recip_full, thresh, feat_idx, n_kept, scores_out):
    """
    SURF scoring for an _arbitrary subset_ of features, using a pre-calculated
    global threshold.
    """
    n_samples = X.shape[0]
    i = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    hits_tile = cuda.shared.array(shape=MAX_F_TILE, dtype=float32)
    miss_tile = cuda.shared.array(shape=MAX_F_TILE, dtype=float32)
    sh_red_i32 = cuda.shared.array(shape=TPB, dtype=int32)

    for f0 in range(0, n_kept, MAX_F_TILE):
        tile_len = min(MAX_F_TILE, n_kept - f0)
        if tid < tile_len:
            hits_tile[tid] = 0.0
            miss_tile[tid] = 0.0
        cuda.syncthreads()

        n_hit_local = 0
        n_miss_local = 0
        for j in range(tid, n_samples, TPB):
            if j == i:
                continue
            dist = 0.0
            for f_tile_idx in range(tile_len):
                full_idx = feat_idx[f0 + f_tile_idx]
                diff = X[i, full_idx] - X[j, full_idx]
                dist += abs(diff) * recip_full[full_idx]

            dist /= n_kept
            if dist >= thresh:
                continue

            is_hit = y[i] == y[j]
            for f_tile_idx in range(tile_len):
                full_idx = feat_idx[f0 + f_tile_idx]
                diff = abs(X[i, full_idx] - X[j, full_idx]) * recip_full[full_idx]
                if is_hit:
                    cuda.atomic.add(hits_tile, f_tile_idx, diff)
                else:
                    cuda.atomic.add(miss_tile, f_tile_idx, diff)
            if is_hit:
                n_hit_local += 1
            else:
                n_miss_local += 1
        cuda.syncthreads()

        sh_red_i32[tid] = n_hit_local
        cuda.syncthreads()
        off = TPB // 2
        while off > 0:
            if tid < off:
                sh_red_i32[tid] += sh_red_i32[tid + off]
            cuda.syncthreads()
            off //= 2
        total_hits = sh_red_i32[0]

        sh_red_i32[tid] = n_miss_local
        cuda.syncthreads()
        off = TPB // 2
        while off > 0:
            if tid < off:
                sh_red_i32[tid] += sh_red_i32[tid + off]
            cuda.syncthreads()
            off //= 2
        total_miss = sh_red_i32[0]

        # Update final scores
        if tid < tile_len:
            local_idx = f0 + tid
            term = 0.0
            if total_miss > 0:
                term += miss_tile[tid] / total_miss
            if total_hits > 0:
                term -= hits_tile[tid] / total_hits
            cuda.atomic.add(scores_out, local_idx, term)
        cuda.syncthreads()


@njit(parallel=True, fastmath=True)
def _surf_cpu_kernel(X, y, recip_full, thresh, feat_idx, n_kept, scores_out):
    """Optimized SURF scoring for CPU using a global threshold."""
    n_samples = X.shape[0]
    temp_scores = np.zeros((n_samples, n_kept), dtype=np.float64)

    for i in prange(n_samples):
        hit_diffs = np.zeros(n_kept, dtype=np.float64)
        miss_diffs = np.zeros(n_kept, dtype=np.float64)
        n_hits = 0
        n_miss = 0

        for j in range(n_samples):
            if i == j:
                continue
            dist = 0.0
            for k in range(n_kept):
                f = feat_idx[k]
                diff = X[i, f] - X[j, f]
                dist += abs(diff) * recip_full[f]

            dist /= n_kept
            if dist >= thresh:
                continue

            is_hit = (y[i] == y[j])
            if is_hit:
                n_hits += 1
            else:
                n_miss += 1
            for k in range(n_kept):
                f = feat_idx[k]
                diff = abs(X[i, f] - X[j, f]) * recip_full[f]
                if is_hit:
                    hit_diffs[k] += diff
                else:
                    miss_diffs[k] += diff

        if n_hits > 0:
            hit_diffs /= n_hits
        if n_miss > 0:
            miss_diffs /= n_miss
        for k in range(n_kept):
            temp_scores[i, k] = miss_diffs[k] - hit_diffs[k]

    for k in prange(n_kept):
        scores_out[k] = temp_scores[:, k].sum()


def _surf_gpu_host_caller(X_d, y_d, recip_full_d, thresh, feat_idx):
    """Host helper function that launches the SURF GPU kernel."""
    n_samples, _ = X_d.shape
    n_kept = feat_idx.size
    feat_idx_d = cuda.to_device(feat_idx.astype(np.int64, copy=False))
    scores_d = cuda.device_array(n_kept, dtype=np.float32)
    scores_d[:] = 0.0

    _surf_score_kernel_idx[n_samples, TPB](
        X_d, y_d, recip_full_d, thresh, feat_idx_d, n_kept, scores_d
    )
    cuda.synchronize()
    return scores_d.copy_to_host() / n_samples


def _surf_cpu_host_caller(X, y, recip_full, thresh, feat_idx):
    """Host caller for the SURF CPU kernel."""
    n_kept = feat_idx.size
    scores = np.zeros(n_kept, dtype=np.float32)
    _surf_cpu_kernel(X, y, recip_full, thresh, feat_idx, n_kept, scores)
    return scores / X.shape[0]


class SURF(BaseEstimator, TransformerMixin):
    """GPU and CPU-accelerated feature selection using the SURF algorithm."""
    def __init__(self, n_features_to_select: int = 10, backend: str = 'auto'):
        self.n_features_to_select = n_features_to_select
        self.backend = backend

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y, dtype=np.float32, ensure_2d=True)
        self.n_features_in_ = X.shape[1]

        if self.backend == 'auto':
            self.effective_backend_ = 'gpu' if cuda.is_available() else 'cpu'
        elif self.backend == 'gpu':
            if not cuda.is_available():
                raise RuntimeError("backend='gpu' selected, but no compatible GPU found.")
            self.effective_backend_ = 'gpu'
        else:
            self.effective_backend_ = 'cpu'

        feature_ranges = _compute_ranges(X)
        feature_ranges[feature_ranges == 0] = 1
        recip_full = (1.0 / feature_ranges).astype(np.float32)
        all_feature_indices = np.arange(self.n_features_in_, dtype=np.int64)

        # --- Key SURF difference: Calculate global threshold first ---
        global_thresh = _compute_surf_threshold(X, recip_full)

        if self.effective_backend_ == 'gpu':
            X_d = cuda.to_device(X)
            y_d = cuda.to_device(y.astype(np.int32))
            recip_full_d = cuda.to_device(recip_full)
            scores = _surf_gpu_host_caller(
                X_d, y_d, recip_full_d, global_thresh, all_feature_indices
            )
        else:
            scores = _surf_cpu_host_caller(
                X, y, recip_full, global_thresh, all_feature_indices
            )

        self.feature_importances_ = scores
        self.top_features_ = np.argsort(scores)[::-1][:self.n_features_to_select]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but was trained with {self.n_features_in_}.")
        return X[:, self.top_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)