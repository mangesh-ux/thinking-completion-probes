# Reproducibility levels

The same five numerical release files are also available from
[Hugging Face](https://huggingface.co/datasets/mangesh-ux/thinking-completion-probes).
They retain their GitHub-release byte hashes; this is a mirror, not a new dataset,
split, result or experiment. The [dataset card and CPU example](huggingface/README.md#cpu-reproduction)
download an immutable Hub revision, authenticate the independently pinned manifest,
and run the existing saved-model verification. Full CPU refitting is an explicit
optional step. NPZ tensors are loaded with NumPy, not `datasets.load_dataset`.
The [Hugging Face publication receipt](huggingface-publication.json) pins the
uploaded revision and records the fresh anonymous download/verification checks.

| Command | What it exercises | What it does not establish |
|---|---|---|
| `tcp verify` | Standard-library consistency of the bundled historical report | Data integrity, refitting or GPU correctness |
| `tcp verify --bundle …` | Pinned numeric files, schemas/pairing, restored-model predictions, all 39 numerical statistics | Fresh fitting, new scientific outcomes |
| `tcp reproduce --bundle … --output …` | 640 TRAIN-only CV fits, five final fits, layer selection, one frozen validation evaluation | New held-out replication or GPU regeneration |
| `python -m pytest` | Synthetic boundary/leakage/serialization/loader/backend tests | Real-model inference success |
| Future `prepare`, `generate`, `features` | Opt-in, version-pinned new private TRAIN/validation regeneration | Bit-identical sampling or historical test evaluation |

The CPU path needs Python 3.12 and the pinned versions in `requirements/cpu.txt`.
An ordinary CPU with 4 GB free memory is a reasonable starting point; reserve
about 500 MB disk for the numerical assets and outputs, separately from the
environment. Full fitting time depends on CPU and BLAS; the maintainer's measured
time and error are recorded in [verification evidence](verification.md), not a
universal performance promise. Numerical comparisons use absolute tolerance
`1e-8`; different runtimes fail the historical reproduction contract rather than
silently accepting an approximate new procedure.

Release files: `train.npz`, `validation.npz`, `fitted-models.json`,
`heldout-predictions.npz`, `manifest.json`. Feature arrays contain no question,
answer, raw tokens or original problem IDs. Their dtype, row order, missingness,
labels and numerical values are preserved. Anonymous group order preserves the
seeded bootstrap stream; original TRAIN folds are copied, not recomputed. The
manifest records new hashes separately from original source hashes. This is an
authorized derivative, not the original immutable private archive.

There are no public raw trajectory/receipt archives and no TEST bundle. Therefore
numeric CPU reproduction can be complete while raw generation reproduction is a
separate opt-in exercise. Future users obtain source data and model weights from
their pinned upstreams under the applicable terms. Their generated raw outputs
must remain private; default `.gitignore` protections are deliberate.

Run outputs preserve source/procedure hashes, runtime, input fingerprints,
configuration scores, final serialized models, prediction identity and failures.
Never overwrite a failed directory or remove inconvenient examples. Repeating a
published CPU calculation in a fresh directory is numerical reproduction, not an
additional validation experiment, and must not be used to retune a scientific claim.
