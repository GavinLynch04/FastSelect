import time
import numpy as np
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed
import numba

def _get_attribute_info(X, discrete_threshold):
    attr_info = {}
    for i in range(X.shape[1]):
        unique_vals = np.unique(X[:, i][~np.isnan(X[:, i])])
        if len(unique_vals) <= discrete_threshold:
            attr_info[i] = ('discrete', 0, 0, 0, 0)
        else:
            min_val, max_val = np.min(unique_vals), np.max(unique_vals)
            attr_info[i] = ('continuous', max_val, min_val, max_val - min_val, np.std(unique_vals))
    return attr_info


@numba.jit(nopython=True, parallel=True)
def _dist_no_missing(X, c_indices, d_indices, c_diffs, num_attributes):
    n_samples = X.shape[0]
    dist_array = np.zeros((n_samples, n_samples))
    num_d_features = len(d_indices)
    num_c_features = len(c_indices)

    for i in numba.prange(n_samples):
        for j in range(i + 1, n_samples):
            c_dist = 0.0
            if num_c_features > 0:
                for k_idx, k in enumerate(c_indices):
                    val1 = X[i, k]
                    val2 = X[j, k]
                    if c_diffs[k_idx] > 0:
                        c_dist += np.abs(val1 - val2) / c_diffs[k_idx]

            d_dist_raw = 0.0
            if num_d_features > 0:
                for k in d_indices:
                    if X[i, k] != X[j, k]:
                        d_dist_raw += 1
                normalized_d_dist = d_dist_raw / num_d_features
                d_dist = normalized_d_dist * num_attributes
            else:
                d_dist = 0.0

            final_dist = c_dist + d_dist
            dist_array[i, j] = final_dist
            dist_array[j, i] = final_dist

    return dist_array


@numba.jit(nopython=True)
def _find_surf_neighbors(inst, n_samples, distance_array, avg_dist):
    neighbors = []
    for i in range(n_samples):
        if inst != i:
            if distance_array[inst, i] < avg_dist:
                neighbors.append(i)
    return np.array(neighbors, dtype=np.int32)


@numba.jit(nopython=True)
def _surf_compute_scores(instance_num, X, y, attr_info_arrays, nan_entries, mcmap, neighbors, class_type, labels_std,
                         data_type):
    scores = np.zeros(X.shape[1])
    instance = X[instance_num]
    n_neighbors = len(neighbors)

    if n_neighbors == 0:
        return scores

    diff_miss = np.zeros(X.shape[1])
    diff_hit = np.zeros(X.shape[1])
    n_hits = 0

    is_hit_map = np.zeros(n_neighbors, dtype=numba.boolean)
    for i, neighbor_idx in enumerate(neighbors):
        is_hit = False
        if class_type == 'binary' or class_type == 'multiclass':
            if y[instance_num] == y[neighbor_idx]:
                is_hit = True
        else:
            if abs(y[instance_num] - y[neighbor_idx]) < labels_std:
                is_hit = True
        if is_hit:
            n_hits += 1
            is_hit_map[i] = True

    n_misses = n_neighbors - n_hits

    for attr_idx in range(X.shape[1]):
        if nan_entries[instance_num, attr_idx]:
            continue

        attr_type = attr_info_arrays[0][attr_idx]

        for i, neighbor_idx in enumerate(neighbors):
            if nan_entries[neighbor_idx, attr_idx]:
                continue

            diff = 0.0
            if attr_type == 0:
                if instance[attr_idx] != X[neighbor_idx, attr_idx]:
                    diff = 1.0
            else:
                attr_max_min_diff = attr_info_arrays[1][attr_idx]
                if attr_max_min_diff > 0:
                    diff = abs(instance[attr_idx] - X[neighbor_idx, attr_idx]) / attr_max_min_diff

            if is_hit_map[i]:
                diff_hit[attr_idx] += diff
            else:
                if class_type == 'multiclass':
                    diff_miss[attr_idx] += diff * mcmap[int(y[neighbor_idx])]
                else:
                    diff_miss[attr_idx] += diff

    if n_hits > 0:
        scores -= diff_hit / n_hits
    if n_misses > 0:
        scores += diff_miss / n_misses

    return scores


class SURF(BaseEstimator):
    def __init__(self, n_features_to_select=10, discrete_threshold=10, verbose=False, n_jobs=-1):
        self.n_features_to_select = n_features_to_select
        self.discrete_threshold = discrete_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self._datalen, self._num_attributes = self._X.shape

        self._label_list = np.unique(self._y)
        if len(self._label_list) <= self.discrete_threshold:
            self._class_type = 'binary' if len(self._label_list) == 2 else 'multiclass'
        else:
            self._class_type = 'continuous'

        self._labels_std = np.std(self._y, ddof=1) if self._class_type == 'continuous' else 0.0

        self.attr_ = _get_attribute_info(self._X, self.discrete_threshold)

        c_indices = [i for i, info in self.attr_.items() if info[0] == 'continuous']
        d_indices = [i for i, info in self.attr_.items() if info[0] == 'discrete']
        c_diffs = np.array([self.attr_[i][3] for i in c_indices], dtype=np.float64)

        if self.verbose:
            start_time = time.time()
            print("Calculating distance array...")

        self._distance_array = _dist_no_missing(self._X,
                                                np.array(c_indices, dtype=np.int64),
                                                np.array(d_indices, dtype=np.int64),
                                                c_diffs,
                                                self._num_attributes)

        if self.verbose:
            print(f"Distance array calculated in {time.time() - start_time:.2f} seconds.")

        avg_dist = np.mean(self._distance_array)

        nan_entries = np.isnan(self._X)

        mcmap = np.zeros(int(np.max(self._y)) + 1) if self._class_type == 'multiclass' else np.array([0.0])
        if self._class_type == 'multiclass':
            class_counts = np.bincount(self._y.astype(int))
            class_probs = class_counts / len(self._y)
            for i, prob in enumerate(class_probs):
                mcmap[i] = prob

        attr_types = np.array([0 if info[0] == 'discrete' else 1 for info in self.attr_.values()], dtype=np.int64)
        attr_max_min_diffs = np.array([info[3] for info in self.attr_.values()], dtype=np.float64)
        attr_info_arrays = (attr_types, attr_max_min_diffs)

        if self.verbose:
            start_time = time.time()
            print("Scoring features...")

        scores = Parallel(n_jobs=self.n_jobs)(delayed(_surf_compute_scores)(
            i, self._X, self._y, attr_info_arrays, nan_entries, mcmap,
            _find_surf_neighbors(i, self._datalen, self._distance_array, avg_dist),
            self._class_type, self._labels_std, 'mixed'
        ) for i in range(self._datalen))

        self.feature_importances_ = np.sum(scores, axis=0) / self._datalen
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]

        if self.verbose:
            print(f"Feature scoring completed in {time.time() - start_time:.2f} seconds.")

        return self

    def transform(self, X):
        return X[:, self.top_features_[:self.n_features_to_select]]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

