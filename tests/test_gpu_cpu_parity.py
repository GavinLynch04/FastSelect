import pytest
import numpy as np
from numpy.testing import assert_allclose
from fast_select import ReliefF, SURF, MultiSURF, CFS, MDR, mRMR
from fast_select.utils import is_cuda_ready

REQUIRES_CUDA = pytest.mark.skipif(
    not is_cuda_ready(), reason="NVIDIA GPU with CUDA environment required for GPU tests."
)


@pytest.fixture
def synthetic_datasets():
    np.random.seed(42)
    datasets = {}

    # 1. Standard Dataset
    X_std = np.random.randn(100, 20).astype(np.float32)
    y_std = np.random.choice([0, 1, 2], size=100)
    datasets['standard'] = (X_std, y_std)

    # 2. Single Feature Edge Case (p=1)
    X_single = np.random.randn(50, 1).astype(np.float32)
    y_single = np.random.choice([0, 1], size=50)
    datasets['single_feature'] = (X_single, y_single)

    # 3. All Discrete Features
    X_disc = np.random.randint(0, 4, size=(80, 15)).astype(np.float32)
    y_disc = np.random.choice([0, 1], size=80)
    datasets['all_discrete'] = (X_disc, y_disc)

    # 4. Large p >> n (p=500, n=40)
    X_p_dom = np.random.randn(40, 500).astype(np.float32)
    y_p_dom = np.random.choice([0, 1, 2], size=40)
    datasets['p_dominant'] = (X_p_dom, y_p_dom)

    # 5. Constant / Zero-Variance Features
    X_const = np.random.randn(60, 10).astype(np.float32)
    X_const[:, 3] = 5.0  # Constant column
    y_const = np.random.choice([0, 1], size=60)
    datasets['zero_variance'] = (X_const, y_const)

    return datasets


@REQUIRES_CUDA
@pytest.mark.parametrize("estimator_cls", [ReliefF, SURF, MultiSURF])
@pytest.mark.parametrize("ds_key", ['standard', 'single_feature', 'all_discrete', 'p_dominant', 'zero_variance'])
def test_cpu_gpu_numerical_parity_relief(synthetic_datasets, estimator_cls, ds_key):
    X, y = synthetic_datasets[ds_key]

    cpu_model = estimator_cls(n_features_to_select=max(1, X.shape[1] // 2), backend='cpu')
    cpu_model.fit(X, y)
    scores_cpu = cpu_model.feature_importances_

    gpu_model = estimator_cls(n_features_to_select=max(1, X.shape[1] // 2), backend='gpu')
    gpu_model.fit(X, y)
    scores_gpu = gpu_model.feature_importances_

    assert_allclose(
        scores_cpu,
        scores_gpu,
        rtol=1e-4,
        atol=1e-5,
        err_msg=f"{estimator_cls.__name__} CPU/GPU parity failed on {ds_key}"
    )
