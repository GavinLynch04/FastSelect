# Algorithm Compliance and Performance Policy

This policy applies to the entire repository. More specific `CLAUDE.md` files
under `src/fast_select` and `tests` add implementation and verification rules.

## Authority order

1. The defining paper and its equations/pseudocode are the semantic authority.
2. Author-maintained errata or a later paper that explicitly corrects the
   defining paper may override it; record the reason and citation.
3. Reference libraries such as scikit-rebate and scikit-learn are corroborating
   evidence only. Never copy their behavior when it conflicts with a paper.
4. Existing FastSelect behavior is not an authority. Backward compatibility
   must not preserve a confirmed algorithm error.

Every algorithm change must name the paper, equation/pseudocode rule, supported
problem type, and any intentional limitation. Do not describe an extension as
the original algorithm.

## No-drift rule

Performance work may change data layout, traversal order, parallel reduction,
allocation strategy, precision where tolerances permit, or host/device
placement. It must not change:

- distance or similarity definitions;
- neighbor membership, boundary inequalities, or tie rules;
- hit/miss, class, fold, or subset normalization;
- update signs or the quantity being scored (`diff` versus `1 - diff`);
- class priors, risk thresholds, entropy/logarithm units, or degrees of freedom;
- subset-search state, stopping criteria, or tie-breaking;
- public interpretation of fitted scores and selected features.

When floating-point reordering is the only difference, demonstrate bounded
error on adversarial and ordinary inputs. CPU/GPU agreement alone is
insufficient because both paths can share the same error.

## Required change gate

Before merging an algorithm or kernel change:

1. Add or retain an independent, equation-driven oracle test. The oracle must
   not call the production kernel or a helper that implements the same logic.
2. Cover relevant boundaries: exact threshold equality, empty neighbor groups,
   constant features, mixed discrete/continuous inputs, imbalanced labels,
   multiclass behavior where supported, and non-`0/1` binary labels.
3. Test CPU against the oracle and GPU against both the oracle and CPU.
4. Run the complete test suite.
5. For a performance change, use the current fastest correct implementation
   available in this repository as the baseline, including retained local
   optimizations. Freeze that baseline before changing it, using an exact
   revision or a preserved source snapshot and source hash. CPU changes must
   beat that CPU baseline; new GPU backends must be compared against that CPU
   baseline on the same machine, with existing GPU comparisons reported
   separately when available. Run in fresh processes, warm JIT compilation
   first, alternate execution order when multiple rounds are used, and measure
   each version/backend/case for at least five cumulative seconds. Comparisons
   against older releases, including v0.2.1, are optional historical references
   and must not replace the current baseline or establish a new speedup claim.
6. Retain a performance change only when the sustained result demonstrates a
   real runtime or memory improvement. Keep a required correctness fix even
   when it is slower, and report that regression without calling it a speedup.
7. Store machine-readable benchmark results and report baseline identity/source
   hash, dataset shape, backend, thread count, iterations, measured duration,
   central runtime statistic, and memory method.

## Primary sources

- Relief family: Kira and Rendell (1992); Kononenko (1994); Moore et al. for
  SURF/SURF*; Urbanowicz et al., *Benchmarking Relief-Based Feature Selection
  Methods for Bioinformatics Data Mining* (2018), PMCID: PMC6299838; and
  Urbanowicz et al., *Relief-Based Feature Selection: Introduction and Review*
  (2018).
- CFS: Mark A. Hall, *Correlation-based Feature Selection for Machine
  Learning* (PhD thesis, 1999).
- mRMR: Peng, Long, and Ding, *Feature Selection Based on Mutual Information:
  Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy* (TPAMI 2005),
  DOI: 10.1109/TPAMI.2005.159.
- MDR: Ritchie et al., *Multifactor-Dimensionality Reduction Reveals
  High-Order Interactions among Estrogen-Metabolism Genes in Sporadic Breast
  Cancer* (2001), DOI: 10.1086/321276.
- Mutual information: Shannon, *A Mathematical Theory of Communication*
  (1948).
- Chi-squared independence scoring: Pearson's chi-squared statistic; the
  package's count-feature API intentionally follows scikit-learn's documented
  `feature_selection.chi2` formulation.
- Continuous/discrete MI: Ross, *Mutual Information between Discrete and
  Continuous Data Sets* (2014), DOI: 10.1371/journal.pone.0087357, Eq. (2).
- Laplacian Score: He, Cai and Niyogi, *Laplacian Score for Feature Selection*
  (NIPS 2005), degree-centered graph-Laplacian quadratic ratio.
- JMI: Yang and Moody (1999), normalized entropy identity in Brown et al.,
  *Conditional Likelihood Maximisation* (JMLR 2012).
- CMIM: Fleuret, *Fast Binary Feature Selection with Conditional Mutual
  Information* (JMLR 2004); categorical extension using Brown et al. (2012).
- FCBF: Yu and Liu (ICML 2003; JMLR 2004 extension, Figure 4).
- Regression MI: Kraskov, Stögbauer and Grassberger, *Estimating Mutual
  Information* (2004), estimator 1; Ross (2014) for explicit discrete columns.
- GiniIndex: Breiman et al., *Classification and Regression Trees* (1984),
  minimum weighted class impurity over distinct numeric splits.
- MIFS: Battiti, *Using Mutual Information for Selecting Features in
  Supervised Neural Net Learning* (1994), DOI: 10.1109/72.298224.
- CIFE: Lin and Tang, *Conditional Infomax Learning* (ECCV 2006), with the
  feature-selection entropy identity in Brown et al. (JMLR 2012).
- DISR: Meyer and Bontempi (2006); Meyer, Schretter and Bontempi (2008,
  corrected 2009), normalized joint-information criterion.

