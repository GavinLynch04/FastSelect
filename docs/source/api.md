# API Reference

This page provides a detailed API reference for the main classes and functions in `fast-select`.

The `set_*_request` methods that scikit-learn generates on estimators for metadata
routing are omitted: their docstrings link into the scikit-learn glossary, which is
not part of this documentation set.

## Feature Selection Algorithms

### ReliefF

```{eval-rst}
.. autoclass:: fast_select.ReliefF.ReliefF
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: set_fit_request, set_transform_request, set_score_request, set_predict_request
```

### SURF

```{eval-rst}
.. autoclass:: fast_select.SURF.SURF
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: set_fit_request, set_transform_request, set_score_request, set_predict_request
```

### MultiSURF

```{eval-rst}
.. autoclass:: fast_select.MultiSURF.MultiSURF
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: set_fit_request, set_transform_request, set_score_request, set_predict_request
```

### TuRF

```{eval-rst}
.. autoclass:: fast_select.TuRF.TuRF
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: set_fit_request, set_transform_request, set_score_request, set_predict_request
```

### Chi2

```{eval-rst}
.. autoclass:: fast_select.Chi2.chi2
   :members:
   :undoc-members:
   :show-inheritance:
```

### mRMR

```{eval-rst}
.. autoclass:: fast_select.mRMR.mRMR
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: set_fit_request, set_transform_request, set_score_request, set_predict_request
```

### CFS

```{eval-rst}
.. autoclass:: fast_select.CFS.CFS
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: set_fit_request, set_transform_request, set_score_request, set_predict_request
```

### MDR

```{eval-rst}
.. autoclass:: fast_select.MDR.MDR
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: set_fit_request, set_transform_request, set_score_request, set_predict_request
```

## Mutual Information

Discrete mutual information with the same CPU/CUDA backend selection as the
estimators. These back the mRMR criterion and are exported for direct use.

```{eval-rst}
.. autofunction:: fast_select.mutual_information.calculate_mi_single_pair
```

```{eval-rst}
.. autofunction:: fast_select.mutual_information.calculate_mi_matrices
```

## Additional CPU selectors

These selectors support `backend="auto"` and `backend="cpu"`; GPU experiments
were not admitted. See the user guide for categorical-input and score semantics.

```{eval-rst}
.. autoclass:: fast_select.InformationSelectors.JMI
   :members: fit, transform
   :undoc-members:
   :inherited-members:
   :exclude-members: set_fit_request, set_transform_request, get_metadata_routing, set_output

.. autoclass:: fast_select.InformationSelectors.CMIM
   :members: fit, transform
   :undoc-members:
   :inherited-members:
   :exclude-members: set_fit_request, set_transform_request, get_metadata_routing, set_output

.. autoclass:: fast_select.InformationSelectors.FCBF
   :members: fit, transform
   :undoc-members:
   :inherited-members:
   :exclude-members: set_fit_request, set_transform_request, get_metadata_routing, set_output

.. autoclass:: fast_select.LaplacianScore.LaplacianScore
   :members: fit, transform
   :undoc-members:
   :exclude-members: set_fit_request, set_transform_request

.. autofunction:: fast_select.ContinuousMI.mutual_info_classif
```

## Second admission batch: CPU selectors and regression MI

```{eval-rst}
.. autoclass:: fast_select.GiniIndex.GiniIndex
   :members: fit, transform
   :undoc-members:
   :exclude-members: set_fit_request, set_transform_request

.. autoclass:: fast_select.JointInformation.DISR
   :members: fit, transform
   :undoc-members:
   :inherited-members:
   :exclude-members: set_fit_request, set_transform_request, get_metadata_routing, set_output

.. autoclass:: fast_select.JointInformation.CIFE
   :members: fit, transform
   :undoc-members:
   :inherited-members:
   :exclude-members: set_fit_request, set_transform_request, get_metadata_routing, set_output

.. autoclass:: fast_select.JointInformation.MIFS
   :members: fit, transform
   :undoc-members:
   :inherited-members:
   :exclude-members: set_fit_request, set_transform_request, get_metadata_routing, set_output

.. autofunction:: fast_select.RegressionMI.mutual_info_regression
```
