# Algorithm Verification Standards

Follow the repository policy and keep three distinct test layers:

1. **Equation oracle:** a tiny NumPy/Python calculation derived directly from
   the paper. It must not import private production helpers.
2. **Backend conformance:** compare CPU and CUDA results to the oracle, then to
   each other with a tolerance justified by reduction order and precision.
3. **Behavior/API:** validate fitted attributes, transformations, errors, and
   scikit-learn compatibility separately from mathematical correctness.

Every neighbor algorithm needs a dataset for which global and per-target
thresholds differ, plus threshold-equality and empty-group cases. Star variants
must include examples that distinguish `diff` from `1 - diff` and exercise
near, dead-band, and far observations.

Statistical algorithms need hand-calculated distributions with known results:
perfect dependence, independence, constant variables, class imbalance, and
label remapping. Search algorithms need a case where greedy hill climbing and
best-first traversal choose different exploration paths.

Do not approve a semantic change solely because it matches scikit-rebate,
scikit-learn, CPU/GPU parity, or a previous FastSelect release. When using an
external implementation as a cross-check, pin its version and retain the
paper-derived oracle.

Performance tests are not correctness tests. Sustained benchmarks must exclude
JIT warm-up, measure each compared case for at least five cumulative seconds,
run versions in isolated processes, and retain raw JSON output.

