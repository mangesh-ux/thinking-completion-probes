# Future GPU regeneration

This is an **opt-in, unexecuted real-GPU route** for a new stochastic experiment.
It is not required for the CPU reproduction of the completed study. Synthetic
tests establish software contracts, not successful model inference. No GPU
work, model download or dataset download was performed while preparing this
repository.

## Contract

`configs/regeneration_v1.json` pins the DeepMath and Qwen3-1.7B revisions, the
original thinking prompt, BF16, six original seed indices, the 4096-token
reasoning cap, and the native thinking chat template. Source row indices and
exact-question hashes determine identities. A shared normalized-question
duplicate guard precedes selection; both difficulty strata select 80 rows by
the original SHA256 ranking. Only then are selected answers checked for
nonempty strings. Invalid selected answers abort; no replacement is permitted.
Grouped, difficulty-stratified splitting retains 96 TRAIN, 32 VALIDATION and
32 untouched TEST identities. Raw TEST text is not stored in the manifest and
there is no TEST generation route.
Production preparation and generation also check independently pinned semantic
hashes of official membership, split assignments and TRAIN folds; a self-consistent
synthetic or altered-source cohort cannot authorize GPU work. Synthetic operation
tests explicitly mock only that official-cohort trust check.

Every eligible rollout has three symmetric greedy 64-token readouts at raw
generated-token coordinates 128, 256 and 512, using `</think>\n\n`. Answer
features use the last balanced box, exact prefix decoding for character/token
alignment, and only tokens strictly between the outer-brace-containing tokens.
An unavailable candidate probability remains missing. Agreement compares the
immediately preceding available checkpoint readout, with whitespace-only text
normalization; it is not semantic correctness.

The token-512 matrix is extracted separately from **prompt + exact saved
reasoning prefix, without a close cue**. One no-cache base-transformer forward
captures the final-position vector at all 28 post-block outputs before final
normalization: a finite CPU BF16 matrix of shape `(28, 2048)`. Serialization and
matrix-content hashes are recorded. Conversion to the numeric derivative's
float32 array is lossless for BF16 values.

The label is natural closure at or after the 512-token prefix and within the
original 4096-token cap, versus a genuine no-close 4096-token cap. Closure
before token 512 and EOS/other termination without closure are excluded and
retained. This endpoint does not establish correct answers, wrong-to-correct
recovery, intervention effects, or inference savings.

## APIs and ordering

## Hardware and installation

The historical collection used an A40-class 48 GB GPU. The new generation
backend requires native CUDA BF16 and the pinned CUDA-12.8 PyTorch wheel;
CPU-only reproduction does not. This repository has not established a smaller
minimum VRAM requirement. Use the historical hardware class as the conservative
reference and check actual peak memory on an explicitly partial future run.
Reserve at least 20 GB of disk for upstream assets, environment and private
outputs; actual environment/cache duplication can require more.

The original full-protocol synthetic qualification estimated about 29.3 GPU
hours. The reduced 512-checkpoint generation route below omits unnecessary
operations but has **not been timed on a real GPU**. A roughly 30-hour allocation
is only a conservative initial planning allowance, not a measured runtime or
an authorized spend. Establish a local cost/time cutoff before any future run.

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/gpu.txt
# Verify torch.__version__ is 2.11.0+cu128; another wheel fails the runtime gate.
tcp prepare --config configs/regeneration_v1.json --output artifacts/new-regeneration/manifest.json
tcp plan --config configs/regeneration_v1.json --manifest artifacts/new-regeneration/manifest.json
# Only a future user who reviewed compute costs should opt in:
tcp generate --config configs/regeneration_v1.json --manifest artifacts/new-regeneration/manifest.json --output artifacts/new-regeneration/generation --confirm-gpu
tcp features --generation artifacts/new-regeneration/generation --output artifacts/new-regeneration/numeric
```

### Python API

Run from this publication checkout's root. Its `pyproject.toml` must identify
`thinking-completion-probes`, and `.gitignore` must contain `artifacts/` without
ignore negations. All raw output paths are constrained to **this checkout's**
ignored `artifacts/` root, with resolved-path containment preventing symlink
escapes; another repository's identically named directory is forbidden. Do not
commit manifests, reasoning tokens, receipts, matrices or private original IDs.

```python
import json
from pathlib import Path
from thinking_completion.generation import (
    prepare_manifest, generation_plan, generate, extract_features,
)

config = json.loads(Path("configs/regeneration_v1.json").read_text())
manifest_path = Path("artifacts/new-regeneration/manifest.json")
# Future user: this may download the fixed dataset, but loads no model/GPU.
manifest = prepare_manifest(config, manifest_path)
plan = generation_plan(config, manifest)
print({key: value for key, value in plan.items() if key != "request_keys"})
# Deliberately opt in only after reviewing the plan and external costs.
summary = generate(config, manifest_path,
                   Path("artifacts/new-regeneration/generation"), confirm_gpu=True)
report = extract_features(Path("artifacts/new-regeneration/generation"),
                          Path("artifacts/new-regeneration/numeric"))
```

The full plan permits at most **3840 model operations**: 768 sampled reasoning
generations, 2304 greedy readouts and 768 activation forwards. Early/unavailable
prefixes reduce actual work without substitution. `max_rows` caps the number of
rollouts in canonical identity/index order for an explicitly partial diagnostic;
such a run is not a completed full-cohort study. The plan is written before CUDA
or model loading. No cloud resource or budget manager is included: users must
independently arrange compute, cost limits and shutdown.

Every model request has a write-once intent followed by a receipt. The first
operational failure stops collection; successful prior outputs and the failure
remain. An existing generation directory is never resumed or replayed, including
an uncertain intent with no receipt. A new paid scientific retry would require
new human and scientific authorization outside this software route.

The runtime fails closed before model loading if the pinned package versions or
native CUDA BF16 support differ. It hashes the resolved model snapshot assets,
chat template and full generation defaults, captures GPU/runtime provenance,
and checks exact control/EOS IDs. Install the CUDA-12.8 PyTorch build explicitly;
the generic optional dependency alone does not prove the `+cu128` runtime.
`local_files_only=true` permits cache-only preparation/loading with no network
fallback. The pure cohort reconstruction has been CPU-audited against all ten
hash/size-verified original cached Parquet shards: all 103022 source rows, all
160 selected identities and hashes, all 96/32/32 split assignments, and all
96 TRAIN fold assignments matched exactly. Aggregate audit evidence SHA256 is
`b29ee5bdc6a7a44ba4639d7fb1e6c96d55600b8056043d229e97257b9687d4c2`.
The production plan passed the independently pinned cohort trust checks with
the verified real cached cohort and permitted exactly 3840 maximum operations,
with zero TEST requests; no operations were executed.
The `load_dataset` preparation adapter, downloads and real CUDA integration
remain unexecuted; the cohort audit is not evidence of GPU regeneration.

Feature extraction verifies every recorded payload's size and SHA256 before
parsing, validates BF16 matrix geometry/content, constructs prefix features
before joining outcomes, and preserves exclusions.
Nonfinite readout scalars retain explicit per-field conversion metadata in
receipts; the derivative report aggregates those counts and any additional
nonfinite-to-missing feature conversions without changing numerical values.
It produces separately
versioned private `train.npz` / `validation.npz` derivatives with the same 11
numeric fields, 512 lexical buckets, 28x2048 activations, topics, problem groups,
row IDs, difficulty, labels and original-ID-derived five-fold assignments.
These new stochastic derivatives must never use the historical release's
provenance manifest or replace its results. No probes are fitted here.

After reviewing the new TRAIN support and declaring that new version's
evaluation intent, the same CPU modeling modules can train and evaluate the
regenerated numerical inputs without comparing them to historical outcomes:

```python
from thinking_completion.data import load_dataset
from thinking_completion.modeling import select, evaluate

train = load_dataset(Path("artifacts/new-regeneration/numeric/train.npz"))
selection = select(train, Path("artifacts/new-regeneration/selection"))
validation = load_dataset(Path("artifacts/new-regeneration/numeric/validation.npz"))
result = evaluate(train, validation, Path("artifacts/new-regeneration/selection"),
                  Path("artifacts/new-regeneration/evaluation"))
```

Selection/evaluation fail if predeclared class or problem-diversity support is
insufficient; never backfill it. `tcp reproduce` is specifically the pinned
historical numeric-release workflow, not this new-study analysis entry point.

Terminal readouts and the original 1024/2048 activation coordinates are omitted:
they are unnecessary for this completion endpoint. Reproducing the original
correctness/recovery diagnosis would need an independently reviewed historical
route, full readout/evaluator contracts and an explicit new authorization. That
extension is not implemented or tested by this API. Fresh outcomes, exclusions
and selected probes may differ across hardware even with identical revisions
and per-rollout seeds; historical bit identity is not promised.
