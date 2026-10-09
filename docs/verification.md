# Actually exercised verification

Publication verification was performed on **2026-10-09**. These checks are
distinct; no synthetic test is represented as a real GPU experiment.

- A fresh Python 3.12.3 CPU environment installed the project and pinned numerical
  dependencies; `pip check` reported no broken requirements.
- Final CPU-only suite: **63 passed, 4 skipped**, plus 19 passing subtests.
  Skips require optional PyTorch CPU fixtures. Separately, all **25** generation
  tests passed with the existing CPU-capable PyTorch environment, without CUDA
  or model calls. Standard-library-only verification also passed with Python's
  site-package loading disabled.
- The lossless numerical export verified every original input hash before and
  after reading. All feature values, dtypes, missingness, labels, folds and
  ordering were preserved; only grouping/row identifiers were replaced.
- Saved-model verification restored all five models: maximum prediction error
  **0.0**. Recomputing all **39** metrics, paired effects/intervals and matched
  statistics yielded maximum absolute error **0.0** against the historical record.
- Full CPU reproduction performed **640** grouped TRAIN CV fits and **five**
  final TRAIN fits, then one frozen validation evaluation. All 39 statistics
  reproduced with maximum absolute error **0.0**. Every selected regularization
  was `0.001`; both activation models selected zero-based layer `15`. Measured
  wall time was **215.266 seconds**, including concurrent synthetic-test activity;
  this is not a cross-machine runtime guarantee.
- CPU-only source reconstruction authenticated all **10** original Parquet shard
  hashes/byte sizes before reading **103,022** source rows in original order.
  All **160** selected identities and question/reference/metadata/duplicate hashes,
  **96/32/32** problem splits and **96** stored TRAIN fold assignments matched.
  Production generation-plan trust checks passed: **3840** maximum operations,
  **zero TEST requests**, with no model calls or dataset downloads.

The release archive is 83,543,193 bytes (about 79.7 MiB):

```
archive SHA256  baea58d3a5156bc70bdf46f0c105cc00ba1bf92db0145051969475215d1d2383
manifest SHA256 fed6a224ef7b6576de47f27cc2d5ef42e171999df57ffc1aa66dfb2e3f431c7f
```

CPU numerical environment: NumPy 2.2.6, SciPy 1.14.0, scikit-learn 1.5.1,
joblib 1.4.2, threadpoolctl 3.5.0. Local Parquet identity audit used PyArrow
23.0.1 and authenticated pinned input bytes; it was not a run of the future
`datasets.load_dataset` download/cache adapter.

Future real-model generation, CUDA integration, model downloads and the
`datasets.load_dataset` preparation adapter remain **unexecuted**. BF16 matrix
and generation/extraction orchestration tests use synthetic CPU fixtures, never
a model or GPU. Original source evidence and conclusions were left unchanged;
no original TEST outcomes were loaded or collected, and no paid resources were
started.

## Public delivery checks

The initial public source commit is
`d71422e0f59fddcbff8960c876201301b8be4b4c`, tagged `v1.0.0`.
Both its [branch CI run](https://github.com/mangesh-ux/thinking-completion-probes/actions/runs/37889090814)
and [tag CI run](https://github.com/mangesh-ux/thinking-completion-probes/actions/runs/37889100658)
completed successfully.

The [public numerical release](https://github.com/mangesh-ux/thinking-completion-probes/releases/tag/v1.0.0)
uploaded all 83,543,193 archive bytes; GitHub's recorded digest matches the
pinned archive SHA256 above. A fresh clone built a distributable wheel, installed
it without numerical dependencies in a separate empty environment, and passed
the installed standard-library report check. That installed package downloaded
the public archive into a fresh directory and verified its archive/manifest
checksums. The cloned source then verified bundle integrity and restored models
using the pinned CPU environment: all predictions and 39 statistics again
matched exactly (maximum absolute error 0.0), with no historical private-workspace
dependency. This delivery check did not repeat model fitting or collect new data.

A final read-only check reconfirmed all five authoritative original input hashes
and unchanged historical Git HEAD/remote. The new public repository is separate;
the existing untracked machine-specific helper remains private and uncommitted.
