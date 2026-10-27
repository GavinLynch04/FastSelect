# Contributing to fast-select

Thank you for helping improve fast-select. Bug reports, documentation fixes,
new algorithms, and performance work are all welcome. Everyone taking part is
expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting bugs and requesting features

Open an [issue](https://github.com/GavinLynch04/FastSelect/issues). For a bug,
please include:

- a minimal script that reproduces it, with a small dataset or the code that
  generates one;
- the expected and actual output (scores, selected features, or the traceback);
- `fast_select.__version__`, your Python and NumPy/Numba/scikit-learn versions,
  your OS, and the backend that ran (`effective_backend_` after fitting).

To request a new algorithm, cite the paper that defines it and say which
problem types (binary, multiclass, regression; discrete or continuous
features) you need.

## Development setup

```bash
git clone https://github.com/GavinLynch04/FastSelect.git
cd FastSelect
pip install -e ".[dev]"
```

On Apple Silicon, add the Metal backend with `pip install -e ".[dev,metal]"`.

## Making a change

1. Fork the repository and create a branch from `development`.
2. Make your change, with tests (see below).
3. Run the checks:

   ```bash
   ruff check src tests benchmarking
   black --check src tests benchmarking
   pytest
   ```

   If you touched CUDA kernels, also run them in the Numba CUDA simulator:

   ```bash
   NUMBA_ENABLE_CUDASIM=1 pytest tests/test_cuda_simulator.py tests/test_exact_boundaries.py --no-cov
   ```

   If you touched the documentation, build it with warnings as errors:

   ```bash
   sphinx-build -W --keep-going -b html docs/source docs/_build/html
   ```

4. Add a line under `## [Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md) for
   any user-visible change.
5. Open a pull request against `development` describing what changed and why.

## Algorithm changes

fast-select treats each algorithm's **defining paper** as the authority on what
it computes. The full rules, including the per-algorithm invariants and the test
standards, are in the
[algorithm policy](docs/development/algorithm-policy.md). In short:

- **Cite the source.** Name the paper and the equation or pseudocode rule you
  implement, the problem types supported, and any limitation. Author errata or
  a later correcting paper may override the original; say which and why.
- **Reference libraries are evidence, not authority.** scikit-rebate,
  scikit-learn and similar packages can corroborate a result, but never copy
  their behaviour where it conflicts with the paper. Existing fast-select
  behaviour is not an authority either.
- **Add an independent oracle test.** Compute the expected result directly from
  the published equations in plain NumPy/Python. The oracle must not call the
  production kernel or any helper that implements the same logic.
- **Cover the boundaries:** exact threshold equality, empty neighbour groups,
  constant features, mixed discrete/continuous inputs, imbalanced labels,
  multiclass targets where supported, and binary labels other than `0/1`.
  Boundary tests must not rely on floating-point luck; use integer data whose
  ties are exact in rational arithmetic (`fractions.Fraction`).
- **Test every backend.** Compare the CPU result to the oracle, and GPU results
  (CUDA, Metal) to both the oracle and the CPU.
- **Label extensions.** If you support more than the paper defines, document it
  as an extension, not as the original algorithm.

## Performance changes

A performance change may alter data layout, traversal order, parallel
reduction, allocation, precision within tolerance, or host/device placement.
It must **not** change distance or similarity definitions, neighbour
membership, boundary inequalities or tie rules, normalisation, update signs,
priors or thresholds, logarithm units, subset-search behaviour, or what fitted
scores mean.

A change is kept only when a sustained measurement shows a real runtime or
memory gain against the current fastest correct implementation:

- freeze that baseline first (an exact revision, or a preserved source
  snapshot plus its hash);
- run each version in fresh processes, warm up JIT compilation, alternate the
  order across rounds, and measure each case for at least five cumulative
  seconds;
- store the machine-readable results, with baseline identity, dataset shape,
  backend, thread count, iterations, duration and memory method.

`benchmarking/compare_head_vs_current.py --gate` automates the comparison and
fails unless every case keeps identical outputs and improves in every round.
See [`benchmarking/README.md`](benchmarking/README.md) for the full workflow. A
required correctness fix is kept even if it is slower; report the regression
rather than calling it a speedup.

## GPU backends

- The CUDA simulator checks kernel logic only. It is not evidence of
  correctness or speed on a physical NVIDIA GPU, so don't describe it as such.
- Metal tests run only on Apple Silicon with `mlx` installed. CI runs on Linux
  and cannot run them, so run `pytest tests/test_metal_algorithms.py
  tests/test_relieff_metal.py` locally if you change Metal code.
- `backend="auto"` may route an algorithm to a GPU only when that GPU path has
  passed the performance gate against the CPU.

## Licence

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
