# User Guide

This guide provides a high-level overview of the different feature selection algorithms available in this library. The goal is to help you understand the core ideas behind each method so you can choose the one that best fits your data and your research question.

For detailed instructions on class parameters and usage, please refer to the [API Reference](./api.md).

## Backend selection

`backend="auto"` prefers a supported NVIDIA/CUDA backend, then optional
Apple Silicon Metal, then CPU. Install Metal support with
`pip install "fast-select[metal]"` (Python 3.10+, arm64 macOS). Use
`backend="metal"` or `backend="cpu"` to force either; `backend="gpu"`
explicitly means CUDA. TuRF inherits the preference from its wrapped selector
on every refit.

Metal is available for Relief-family methods, discrete MI, mRMR, CFS, MDR and
Chi2 (which has no CUDA path). Chi2 and CFS run on Metal only when requested
with `backend="metal"`: their Metal paths were slower than CPU in sustained
benchmarks, so `auto` keeps them on CPU. ReliefF supports up to 64 neighbors on
Metal; information/CFS kernels support up to 64 encoded state slots.
Unsupported inputs fall through to CPU under `auto`; explicit Metal requests
raise. CPU is faster for small inputs and for continuous-feature MultiSURF.

JMI, CMIM, FCBF and LaplacianScore currently retain only their parallel Numba
CPU backends: `auto` selects CPU and explicit GPU requests raise. The dense
continuous MI function `mutual_info_classif` is CPU-only as well. Their Metal
experiments did not consistently improve on these new CPU implementations.

## Additional selectors admitted against external baselines

`JMI(n_features_to_select=10)` balances relevance, redundancy and within-class
dependence. `CMIM(n_features_to_select=10)` maximizes minimum conditional MI.
Both take categorical features and classification targets, use empirical
entropies in bits, and select a fixed number of features. Their categorical
encoding preserves symbol identity; it does not discretize continuous data.
The general categorical CMIM formulation is an extension of the original
binary algorithm. CMIM follows Fleuret's Eq. (2) (as do Brown et al.), the
minimum over selected features only; his Algorithm 1 pseudocode also caps each
score at the candidate's marginal MI, which can choose differently when the
candidate is synergistic with every selected feature. `feature_importances_` is marginal target MI;
`selection_scores_` records the greedy criterion and `top_features_` its order.

`FCBF(delta=0.0)` keeps target SU strictly above `delta`, then removes redundant
peers with pairwise SU greater than or equal to their target SU. Its selected
count is determined by the filter and may be zero. Importances and selection
scores are SU values.

`LaplacianScore(n_features_to_select=0.2, n_neighbors=5)` ranks unsupervised
local structure. It builds an undirected Euclidean binary kNN graph including
self edges; use `weight_mode="heat_kernel"` with `bandwidth`, or supply a
symmetric nonnegative graph through `fit(X, W=W)`. Its complete fit includes
graph construction. Raw `laplacian_scores_` are lower-is-better and importances
are their negatives. Undefined zero weighted-variance features score infinity
and rank last. Graph construction still uses quadratic memory in sample count.

`mutual_info_classif(X, y, random_state=0)` estimates continuous/discrete MI in
nats using Ross's nearest-neighbor estimator. It accepts dense numeric inputs, optional
discrete column masks/indices, and can be passed to `SelectKBest`. It preserves
scikit-learn's public scaling, jitter and negative-estimate clipping conventions;
set `random_state` for repeatability. Exact float64 sorted distances avoid the
small-class distance-cancellation issue recorded in the benchmark audit.

All five passed independent paper-equation oracles and complete-call external
baseline comparisons before entering the package. Workload-specific timings
and limitations are in the repository's new-algorithm benchmark report.

## Second batch of admitted selectors

`GiniIndex(n_features_to_select=10)` ranks numeric classification features by
the minimum weighted child Gini impurity over every distinct-value split,
including the no-split parent. It sorts each feature once and sweeps exact
class counts. Binary and multiclass targets are supported; arbitrary class
labels are encoded by identity. Raw `gini_scores_` is lower-is-better and
`feature_importances_` negates it. This is a univariate split criterion, not
whole-tree normalized feature importance. Unlike scikit-learn CART input
conversion, the implementation preserves float64 near-ties and uint64 ordering.

`DISR`, `CIFE` and `MIFS` take categorical classification inputs and select a
fixed `n_features_to_select` (default 10), with no implicit discretization.
They initialize with maximum marginal target MI. DISR sums joint MI divided
by joint entropy, with zero-entropy contributions explicitly defined as zero.
CIFE subtracts summed redundancy and adds summed within-class dependence.
MIFS subtracts `beta` times summed redundancy, default 0.5. It has no
conditional-dependence term or subset-size averaging. These are different
criteria from JMI/CMIM/mRMR; they can choose different subsets. Importances are
marginal MI in bits, while `selection_scores_` and `top_features_` record the
greedy criterion and order. Negative MIFS/CIFE objective values are retained.

`mutual_info_regression(X, y, random_state=0)` uses KSG estimator 1 with exact
Chebyshev neighbors for continuous columns; explicit discrete columns use
Ross's estimator. It accepts dense numeric inputs, returns nats, preserves the
public scikit-learn scaling/jitter convention and clips negative estimates to
zero. Working feature and target values are float64, including float32 inputs;
this preserves jitter that an in-place float32 target would round away. It
supports `SelectKBest` and requires `n_neighbors < n_samples`.
Singleton categories are excluded for discrete columns; all-singleton columns
raise. The sorted lower-bound search is exact, but can approach quadratic
work on adversarial data or large neighbor counts. Set `random_state` for
repeatability and use a regression estimator downstream.

All five additions use Numba CPU kernels; the estimators accept CPU/auto and
reject explicit GPU backends. No GPU performance claim is made for this batch.
The ANOVA F prototype failed the external performance comparison and was left
outside the installable package. Baselines were tuned across serial, threaded
and process settings; Gini was compared to the faster compiled scikit-learn
CART alternative as well as original scikit-feature. See the repository's
second algorithm-admission report for raw timings and scope.

---

## 1. Relief-Based Algorithms

The Relief family of algorithms are powerful, instance-based feature selectors that excel at detecting feature interactions without exhaustively testing all combinations. They are considered "multivariate" in the sense that their feature scoring depends on the context provided by other features.

### Core Idea: The "Nearest Neighbors" Intuition

The logic behind Relief is simple and intuitive. For a randomly chosen sample, it finds its nearest neighbors:
*   **Near Hit:** The closest neighbor from the **same class**.
*   **Near Miss:** The closest neighbor from a **different class**.

The score for each feature is then updated based on a simple principle:
- A good feature should have **similar values** for instances of the same class (the near hit).
- A good feature should have **different values** for instances of different classes (the near miss).

If a feature's values are closer for the near hit, its score increases. If they are closer for the near miss, its score decreases. This process is repeated for many random samples to get a robust score for each feature.

### Available Algorithms

The library includes several variations of the Relief algorithm, each improving upon the last.

#### ReliefF
This is the most well-known extension of the original Relief. Instead of one near miss, it finds **k** nearest neighbors for both the "hit" and "miss" categories, making its scores more robust, especially for noisy data and multi-class problems.

ReliefF also accepts missing values (NaN) in discrete features, using the papers' rule: a missing value counts as the probability, given the instance's class, that the two values differ. NaN in continuous features, and in the SURF family, is rejected because no paper defines it. Missing-value fits run on the CPU.

#### RReliefF
The regression form of ReliefF (Robnik-Šikonja and Kononenko). With a continuous target there are no hits or misses, so RReliefF uses the **k** nearest neighbors of every sample, whatever their targets, and weighs how often a feature differs when the target differs against how often it differs when the target does not. Nearer neighbors count more (`sigma` controls how quickly influence decays with rank), or all count equally with `weighting="constant"`. It runs on the CPU.

#### SURF
SURF improves upon ReliefF by removing the need to specify `k`. Instead, it defines neighbors as all other samples within a certain distance (radius) from the target sample. This avoids the problem of choosing an arbitrary `k` and is more robust to different feature densities.

#### MultiSURF
This is a powerful variant that uses the distance threshold idea from SURF but applies it to **all samples in the dataset**, not just the nearest ones. It defines a "near" and "far" neighborhood for each instance and updates scores based on neighbors that fall into the "near" category. This makes its scoring very stable.

### TuRF
This is a wrapper method that goes around any of the Relief based algorithms in this package, its purpose is to iteratively subselect features until a desired level is met. This has the advantage of improving performance in noisy environments, and datasets with huge numbers of features (such as biological datasets). It is recommended to shave off a small percent of features each run, say 20%, to improve stability.

### When to Use Relief-Based Algorithms

**Strengths:**
- **Handles Feature Interactions:** This is their main advantage. They can detect features that are only useful in the context of others.
- **Handles Mixed Data Types:** They naturally work with both continuous and discrete (categorical) features without extra pre-processing.
- **Theoretically Sound:** The logic is simple, powerful, and effective in a wide range of domains.

**Weaknesses:**
- **Computationally Expensive:** Because they operate on instance-to-instance distances, their complexity can be high for datasets with many samples (`O(n_samples^2 * n_features)` in the naive case), although running on a powerful GPU makes these algorithms much faster.
- **Sensitive to Noisy Features:** If many irrelevant features are present, the distance metric can be skewed, making it harder to find true nearest neighbors.

---

## 2. Minimum Redundancy Maximum Relevance (mRMR)

mRMR is a feature selection method based on information theory. It formalizes the common-sense goal of finding features that are individually informative about the target variable, but not redundant with each other.

### Core Idea: Balancing Relevance and Redundancy

mRMR uses Mutual Information (MI) to quantify both relevance and redundancy. It selects features iteratively using a greedy search. In each step, it chooses the feature that best optimizes the mRMR criterion:

- **Relevance:** The Mutual Information between a candidate feature and the target class: `I(f; y)`. A higher value is better.
- **Redundancy:** The average Mutual Information between a candidate feature and all features already selected (`S`): `mean(I(f; f_s))` for `f_s` in `S`. A lower value is better.

Two main criteria are used to combine these:
- **MID (Mutual Information Difference):** `Score = I(f; y) - mean(I(f; S))`
- **MIQ (Mutual Information Quotient):** `Score = I(f; y) / mean(I(f; S))`

### When to Use mRMR

**Strengths:**
- **Produces Non-Redundant Feature Sets:** Excellent at finding a compact set of features that are all independently powerful.
- **Strong Theoretical Foundation:** Based on well-understood principles from information theory.
- **Fast Greedy Search:** The selection process after pre-computation is very quick.

**Weaknesses:**
- **Requires Discrete Data:** Standard MI is defined for discrete variables. Continuous data must be discretized first.
- **Pre-computation is Expensive:** Calculating the full feature-feature redundancy matrix can be costly (`O(n_features^2 * n_samples)`).
- **Ignores Interactions (Partially):** While it handles redundancy, it doesn't explicitly model higher-order interactions like Relief does.

---

## 3. Correlation-based Feature Selection (CFS)

CFS is another theoretically elegant method that evaluates the merit of a **subset of features** rather than scoring individual features.

### Core Idea: Good Subsets, Not Just Good Individuals

The core hypothesis of CFS is:
> "A good feature subset contains features highly correlated with the target, yet uncorrelated with each other."

It uses a "merit" score to evaluate a subset, which is conceptually:

`Merit(S) = (Avg. Feature-Target Correlation) / sqrt(Avg. Feature-Feature Correlation)`

The algorithm searches for the subset with the highest merit. Since an exhaustive search is impossible, a heuristic search is used; `fast-select` implements Hall's **forward best-first** search with the canonical consecutive-non-improvement stopping rule (`max_backtracks`, default 5).

### When to Use CFS

**Strengths:**
- **Evaluates Subsets:** Its focus on subset quality is unique and can lead to better performing models.
- **Redundancy is a Core Component:** Like mRMR, it explicitly penalizes redundant features.

**Weaknesses:**
- **NP-Hard Problem:** Finding the optimal subset is computationally intractable. Its performance depends entirely on the quality of the heuristic search algorithm used.
- **Can Be Slow:** The search process can be much slower than the greedy search of mRMR, as it may evaluate many different subsets.

---

## 4. Univariate Methods (e.g., Chi-Squared)

Univariate methods are the simplest and fastest class of feature selectors. They evaluate each feature independently, without considering any other features.

### Core Idea: Testing for Independence

The Chi-Squared (χ²) test is a statistical test applied to contingency tables to evaluate the likelihood that an observed distribution is due to chance. In feature selection, it's used to test whether a feature is independent of the class label.

- **High Chi² Score:** Implies the feature is **dependent** on the class label (it is informative).
- **Low Chi² Score:** Implies the feature is **independent** of the class label (it is uninformative).

Each feature is scored, and the top `k` features are selected.

### When to Use Chi-Squared

**Strengths:**
- **Extremely Fast:** Since each feature is evaluated independently, it's the fastest method available.
- **Good Baseline:** Excellent for getting a quick first impression of feature importance.

**Weaknesses:**
- **Ignores Feature Redundancy:** It may select a group of highly correlated features because it doesn't know they are redundant.
- **Ignores Feature Interactions:** It cannot detect features that are only useful in combination with others.
- **Requires Non-Negative Data:** The standard implementation requires non-negative feature values.

---

## 5. Multifactor Dimensionality Reduction (MDR)

MDR is a highly specialized, non-parametric method explicitly designed to **detect and characterize non-linear feature interactions**, particularly statistical epistasis in genetics.

### Core Idea: From High-Dimensions to One Dimension

Unlike other methods that select original features, MDR is a **feature construction** method. For a given set of `n` features (e.g., two features, `f1` and `f2`):
1.  It creates a discrete space of all possible multi-locus genotypes (e.g., all combinations of values for `f1` and `f2`).
2.  For each combination, it calculates the ratio of cases (affected) to controls (unaffected).
3.  It labels each combination as "high-risk" if its case/control ratio **meets or exceeds** the dataset-wide case/control ratio, and "low-risk" otherwise. Empty cells are low-risk.
4.  This process collapses the `n`-dimensional space into a new single one-dimensional feature with two values: "high-risk" and "low-risk".
5.  The quality of this new feature is then evaluated using a classifier (e.g., cross-validated accuracy).

The algorithm exhaustively tests up to k=6 interactions of features to find the combination that produces the most predictive new feature.

### When to Use MDR

**Strengths:**
- **State-of-the-Art for Interaction Detection:** Specifically designed for and excels at finding epistasis.
- **Model-Free:** Makes very few assumptions about the data.

**Weaknesses:**
- **Extremely Computationally Intensive:** Exhaustively searching all pairs, triplets, etc., is combinatorially explosive. It is not feasible for large numbers of features.
- **Interpretation Can Be Difficult:** The output is a set of interacting features, but understanding the nature of that interaction requires further analysis.
- **Usually Not a General-Purpose Feature Selector:** Use this when your primary goal is to find interactions, not just to reduce dimensionality for a predictive model.

---
### Summary and Comparison
| Algorithm Family | `X` Data Type | `y` Data Type | Handles Interactions? | Handles Redundancy? | Computational Cost | Best For... |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ReliefF / SURF / MultiSURF** | Discrete & Continuous | Classification | ✅ **Yes** | ❌ No | High | Finding interacting features in mixed-type classification data. |
| **RReliefF** | Discrete & Continuous | Regression | ✅ **Yes** | ❌ No | High | The same neighbor-based scoring for a continuous target. |
| **mRMR** | Discrete | Classification | ❌ No | ✅ **Yes** | Medium-High | Creating a compact, non-redundant feature set for discrete classification data. |
| **CFS** | Discrete & Continuous | Classification | ❌ No | ✅ **Yes** | High | Finding a synergistically predictive subset (theory-driven). |
| **Chi-Squared** | Discrete (Non-Negative), dense or sparse | Classification | ❌ No | ❌ No | **Very Low** | A fast, simple baseline for non-negative discrete classification data. |
| **MDR** | Discrete | Classification | ✅ **Yes (Primary Goal)** | ❌ No | **Very High** | Detecting epistasis and complex non-linear interactions in case-control studies. |
