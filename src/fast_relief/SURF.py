import time

import numba
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

TPB = 64  # Threads Per Block


def _get_attribute_info(x, discrete_threshold):
    attr_info = {}
    for i in range(x.shape[1]):
        unique_vals = np.unique(x[:, i][~np.isnan(x[:, i])])
        if len(unique_vals) <= discrete_threshold:
            attr_info[i] = ("discrete", 0, 0, 0, 0)
        else:
            min_val, max_val = np.min(unique_vals), np.max(unique_vals)
            attr_info[i] = (
                "continuous",
                max_val,
                min_val,
                max_val - min_val,
                np.std(unique_vals),
            )
    return attr_info


@numba.jit(nopython=True, parallel=True)
def _dist_no_missing(x, c_indices, d_indices, c_diffs, num_attributes):
    n_samples = x.shape[0]
    dist_array = np.zeros((n_samples, n_samples))
    num_d_features = len(d_indices)
    num_c_features = len(c_indices)

    for i in numba.prange(n_samples):
        for j in range(i + 1, n_samples):
            c_dist = 0.0
            if num_c_features > 0:
                for k_idx, k in enumerate(c_indices):
                    val1 = x[i, k]
                    val2 = x[j, k]
                    if c_diffs[k_idx] > 0:
                        c_dist += np.abs(val1 - val2) / c_diffs[k_idx]

            d_dist_raw = 0.0
            if num_d_features > 0:
                for k in d_indices:
                    if x[i, k] != x[j, k]:
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
def _surf_compute_scores(
    instance_num,
    x,
    y,
    attr_info_arrays,
    nan_entries,
    mcmap,
    neighbors,
    class_type,
    labels_std,
    data_type,
):
    scores = np.zeros(x.shape[1])
    instance = x[instance_num]
    n_neighbors = len(neighbors)

    if n_neighbors == 0:
        return scores

    diff_miss = np.zeros(x.shape[1])
    diff_hit = np.zeros(x.shape[1])
    n_hits = 0

    is_hit_map = np.zeros(n_neighbors, dtype=numba.boolean)
    for i, neighbor_idx in enumerate(neighbors):
        is_hit = False
        if class_type == "binary" or class_type == "multiclass":
            if y[instance_num] == y[neighbor_idx]:
                is_hit = True
        else:
            if abs(y[instance_num] - y[neighbor_idx]) < labels_std:
                is_hit = True
        if is_hit:
            n_hits += 1
            is_hit_map[i] = True

    n_misses = n_neighbors - n_hits

    for attr_idx in range(x.shape[1]):
        if nan_entries[instance_num, attr_idx]:
            continue

        attr_type = attr_info_arrays[0][attr_idx]

        for i, neighbor_idx in enumerate(neighbors):
            if nan_entries[neighbor_idx, attr_idx]:
                continue

            diff = 0.0
            if attr_type == 0:
                if instance[attr_idx] != x[neighbor_idx, attr_idx]:
                    diff = 1.0
            else:
                attr_max_min_diff = attr_info_arrays[1][attr_idx]
                if attr_max_min_diff > 0:
                    diff = (
                        abs(instance[attr_idx] - x[neighbor_idx, attr_idx])
                        / attr_max_min_diff
                    )

            if is_hit_map[i]:
                diff_hit[attr_idx] += diff
            else:
                if class_type == "multiclass":
                    diff_miss[attr_idx] += diff * mcmap[int(y[neighbor_idx])]
                else:
                    diff_miss[attr_idx] += diff

    if n_hits > 0:
        scores -= diff_hit / n_hits
    if n_misses > 0:
        scores += diff_miss / n_misses

    return scores


@numba.jit(nopython=True)
def _surf_star_compute_scores(
    instance_num,
    x,
    y,
    attr_info_arrays,
    nan_entries,
    mcmap,
    distance_array,
    avg_dist,
    class_type,
    labels_std,
    data_type,
    use_star,
):
    scores = np.zeros(x.shape[1])
    instance = x[instance_num]
    datalen = x.shape[0]

    # --- Near Neighbor Scoring ---
    diff_miss_near = np.zeros(x.shape[1])
    diff_hit_near = np.zeros(x.shape[1])
    n_hits_near = 0
    n_miss_near = 0

    # --- Far Neighbor Scoring (for SURF*) ---
    diff_miss_far = np.zeros(x.shape[1])
    diff_hit_far = np.zeros(x.shape[1])
    n_hits_far = 0
    n_miss_far = 0

    for j in range(datalen):
        if instance_num == j:
            continue

        # Determine if the instance is a Hit or Miss
        is_hit = False
        if class_type == "binary" or class_type == "multiclass":
            if y[instance_num] == y[j]:
                is_hit = True
        else:  # Continuous
            if abs(y[instance_num] - y[j]) < labels_std:
                is_hit = True

        # Calculate difference for each attribute
        for attr_idx in range(x.shape[1]):
            if nan_entries[instance_num, attr_idx] or nan_entries[j, attr_idx]:
                continue

            diff = 0.0
            attr_type = attr_info_arrays[0][attr_idx]
            if attr_type == 0:  # Discrete
                if instance[attr_idx] != x[j, attr_idx]:
                    diff = 1.0
            else:  # Continuous
                attr_max_min_diff = attr_info_arrays[1][attr_idx]
                if attr_max_min_diff > 0:
                    diff = abs(instance[attr_idx] - x[j, attr_idx]) / attr_max_min_diff

            # Weighted diff for multiclass misses
            mc_weight = (
                mcmap[int(y[j])] if class_type == "multiclass" and not is_hit else 1.0
            )

            # Check if neighbor is NEAR or FAR
            is_near = distance_array[instance_num, j] < avg_dist

            if is_near:
                if is_hit:
                    diff_hit_near[attr_idx] += diff
                else:
                    diff_miss_near[attr_idx] += diff * mc_weight
            elif use_star:  # Only process far neighbors if use_star is True
                if is_hit:
                    diff_hit_far[attr_idx] += diff
                else:
                    diff_miss_far[attr_idx] += diff * mc_weight

        # Increment hit/miss counts
        is_near = distance_array[instance_num, j] < avg_dist
        if is_near:
            if is_hit:
                n_hits_near += 1
            else:
                n_miss_near += 1
        elif use_star:
            if is_hit:
                n_hits_far += 1
            else:
                n_miss_far += 1

    # --- Final Score Calculation ---
    # Standard SURF update for near neighbors
    if n_hits_near > 0:
        scores -= diff_hit_near / n_hits_near
    if n_miss_near > 0:
        scores += diff_miss_near / n_miss_near

    # Inverted SURF* update for far neighbors
    if use_star:
        if n_hits_far > 0:
            scores += diff_hit_far / n_hits_far
        if n_miss_far > 0:
            scores -= diff_miss_far / n_miss_far

    return scores


class SURF(BaseEstimator):
    def __init__(
        self,
        n_features_to_select=10,
        discrete_threshold=10,
        verbose=False,
        n_jobs=-1,
        use_star=False,
    ):
        self.n_features_to_select = n_features_to_select
        self.discrete_threshold = discrete_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.use_star = use_star  # Run SURF* adaptation of the SURF algorithm

    def fit(self, x, y):
        self._x = np.asarray(x, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self._datalen, self._num_attributes = self._x.shape

        self._label_list = np.unique(self._y)
        if len(self._label_list) <= self.discrete_threshold:
            self._class_type = "binary" if len(self._label_list) == 2 else "multiclass"
        else:
            self._class_type = "continuous"

        self._labels_std = (
            np.std(self._y, ddof=1) if self._class_type == "continuous" else 0.0
        )

        self.attr_ = _get_attribute_info(self._x, self.discrete_threshold)

        c_indices = [i for i, info in self.attr_.items() if info[0] == "continuous"]
        d_indices = [i for i, info in self.attr_.items() if info[0] == "discrete"]
        c_diffs = np.array([self.attr_[i][3] for i in c_indices], dtype=np.float64)

        if self.verbose:
            start_time = time.time()
            print("Calculating distance array...")

        self._distance_array = _dist_no_missing(
            self._x,
            np.array(c_indices, dtype=np.int64),
            np.array(d_indices, dtype=np.int64),
            c_diffs,
            self._num_attributes,
        )

        if self.verbose:
            print(
                f"Distance array calculated in {time.time() - start_time:.2f} seconds."
            )

        avg_dist = np.mean(self._distance_array)

        nan_entries = np.isnan(self._x)

        mcmap = (
            np.zeros(int(np.max(self._y)) + 1)
            if self._class_type == "multiclass"
            else np.array([0.0])
        )
        if self._class_type == "multiclass":
            class_counts = np.bincount(self._y.astype(int))
            class_probs = class_counts / len(self._y)
            for i, prob in enumerate(class_probs):
                mcmap[i] = prob

        attr_types = np.array(
            [0 if info[0] == "discrete" else 1 for info in self.attr_.values()],
            dtype=np.int64,
        )
        attr_max_min_diffs = np.array(
            [info[3] for info in self.attr_.values()], dtype=np.float64
        )
        attr_info_arrays = (attr_types, attr_max_min_diffs)

        if self.verbose:
            start_time = time.time()
            print("Scoring features...")

        scores = Parallel(n_jobs=self.n_jobs)(
            delayed(_surf_star_compute_scores)(
                i,
                self._x,
                self._y,
                attr_info_arrays,
                nan_entries,
                mcmap,
                self._distance_array,
                avg_dist,
                self._class_type,
                self._labels_std,
                "mixed",
                self.use_star,
            )
            for i in range(self._datalen)
        )

        self.feature_importances_ = np.sum(scores, axis=0) / self._datalen
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]

        if self.verbose:
            print(
                f"Feature scoring completed in {time.time() - start_time:.2f} seconds."
            )

        return self

    def transform(self, x):
        return x[:, self.top_features_[: self.n_features_to_select]]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)
