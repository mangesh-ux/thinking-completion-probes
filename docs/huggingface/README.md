---
license: mit
pretty_name: Thinking Completion Probes — Qwen3-1.7B Numeric Derivatives
language:
- en
tags:
- activation-probing
- thinking-completion
- interpretability
- reproducibility
- numerical-features
- qwen3
size_categories:
- n<1K
---

# Thinking Completion Probes

An anonymized numerical derivative release for an **exploratory** study of
Qwen3-1.7B: does its residual state at generated token 512 improve a forecast of
natural thinking completion within 4096 generated tokens, beyond specified
confidence and lexical controls? **Completion is not answer correctness.**

[Code and study](https://github.com/mangesh-ux/thinking-completion-probes) ·
[GitHub v1.0.0 release](https://github.com/mangesh-ux/thinking-completion-probes/releases/tag/v1.0.0) ·
[Methodology](https://github.com/mangesh-ux/thinking-completion-probes/blob/d71422e0f59fddcbff8960c876201301b8be4b4c/docs/methodology.md) ·
[Release/privacy policy](https://github.com/mangesh-ux/thinking-completion-probes/blob/d71422e0f59fddcbff8960c876201301b8be4b4c/docs/release_policy.md)

Use this bundle to reproduce the published probe/control comparison or explore
alternative analyses on TRAIN. Preserve problem-level grouping and the supplied
folds; validation is already published evidence, not a fresh holdout for tuning
new claims. Confirmatory extensions need a separately specified, new held-out
study. This publication does not launch one.

## Scope and original outcome

The original V2-A1 experiment asked about wrong→correct answer recovery.
Collection completed: 768 base trajectories, 768 terminal readouts, 3840
checkpoint/readout slots and 3793 available activation matrices, with zero
interrupted/failed collection operations. Unavailable slots were preserved.
Answer readout/reference evaluation could not establish eligible wrong→correct
and wrong→wrong populations. That endpoint is **NON_EVALUABLE**, not a negative
activation result. No original recovery probes or test evaluation were completed.

The separately locked completion follow-up reused existing trajectories,
prefixes, confidence features and activations without new GPU inference. It is
not evidence of causal reasoning understanding, remaining-token estimation,
accuracy-preserving early stopping or inference-cost savings.

## Dataset construction and labels

An answer-independent, fixed 160-problem difficulty-7/8 cohort was grouped into
96 TRAIN, 32 validation and 32 unused test problems. Six seeded stochastic
trajectories per TRAIN/validation problem yield the released 768 rows.

| Split | Rows | Independent problems | Natural-close rows | Full-cap no-close rows | Mixed-outcome problems |
|---|---:|---:|---:|---:|---:|
| TRAIN | 576 | 96 | 151 | 425 | 46 |
| Validation | 192 | 32 | 67 | 125 | 17 |

The target is conditional on an available, still-open 512-token reasoning prefix.
`labels=1` means the first natural `</think>` starts at or after that checkpoint
and within the 4096-token generation; `labels=0` means a genuine full-cap
generation without that close. Already-closed, short, failed, early-EOS or
unknown-termination rows are exclusions, not negatives. Historical TRAIN and
validation had no primary-checkpoint attrition. The original test split is absent.

Predictors exclude eventual answers, correctness, close position, termination
reason and remaining-token count. Features were reconstructed after collection
from prefix information: **prefix-information-only, not prospectively timestamped
pre-outcome online measurements**. Labels use the subsequent natural trajectory.

## Files and schema

This is a NumPy NPZ/JSON bundle, not a row-text dataset. Use the pinned Python
loader/example below; automatic `datasets.load_dataset` support is not provided.
`train.npz` and `validation.npz` contain the same fields, with `N=576` and `192`:

| Field | Shape / type | Meaning |
|---|---|---|
| `activations` | `[N,28,2048]` float32 | All zero-based transformer blocks' post-block final-prefix-token residuals, before final normalization |
| `numeric` | `[N,11]` float64 | Four structural and seven frozen readout confidence/agreement features |
| `numeric_feature_names` | `[11]` Unicode | Exact numeric column names |
| `lexical` | `[N,512]` float64 | Fixed SHA256-bin counts of the last 128 generated prefix tokens; collisions, no order |
| `labels` | `[N]` int8 | Natural thinking close (1) versus full-cap no-close (0) |
| `groups`, `row_ids` | `[N]` Unicode | Anonymized problem grouping and row identifiers |
| `fold_ids` | `[N]` int8 | Frozen TRAIN folds 0–4; validation uses −1 |
| `topics`, `difficulty` | `[N]` Unicode / float64 | Categorical topic and difficulty 7 or 8 |

Activations replay the exact prompt plus saved generated-512 prefix **without an
answer cue**. Original BF16 values are represented losslessly as float32.
Confidence features come from separate greedy, 64-token counterfactual readouts;
their original measurement limitations are retained. The three answer-span
fields (`mean_answer_token_logprob`, `min_answer_token_logprob`,
`answer_token_count`) are NaN in 566/576 TRAIN rows and all 192 validation rows.
Entropy, margin, agreement and availability fields are not thereby all missing.
`previous_readout_available` means both current and preceding boxed strings
exist, not that both receipt files exist. Problem-token count excludes the chat
template. Missingness is explicitly handled using TRAIN-fitted preprocessing.

`fitted-models.json` contains frozen model/preprocessor states;
`heldout-predictions.npz` contains paired validation predictions. `manifest.json`
records schemas, support, privacy exclusions and original provenance hashes
separately from the newly serialized/anonymized file hashes. Scientific values
are preserved; byte equality to the original private NPZ archives is not asserted.

## Frozen analysis and result

Models are L2 logistic regressions. A uses structural/topic features; B adds
seven readout features; Blex adds hashed prefix counts; C adds one activation
layer to B; Clex adds that layer to Blex. **Clex versus Blex is primary.**
Five problem-grouped, difficulty-stratified TRAIN folds select among 128 fixed
configurations (640 CV fits), followed by five final TRAIN fits. All preprocessing
is fit within each TRAIN partition. The Clex-selected layer is zero-based 15
(block 16); all selected regularization values are 0.001. C uses that layer,
selecting only its own regularization. Preserve supplied folds: rehashing anonymous
IDs would change the analysis. Validation does not select layers or fit models.

On 32 held-out validation problems, problem-averaged log loss was **0.6687 nats
for Blex versus 0.6134 for Clex**: baseline-minus-added improvement **0.0553 nats
per problem**, with a paired problem-cluster 95% interval **[0.0212, 0.0888]**.
The 2000 resamples preserve repeated sampled problems and condition on frozen
trained models; they do not include training, layer-selection or exploratory
endpoint-selection uncertainty. Secondary comparisons are descriptive and not
multiplicity-corrected. The 192 validation rows are not 192 independent problems.

This supports incremental predictive utility relative to these controls, not
information absent from the tokens. One model/BF16 backend, one checkpoint, a
capped horizon, a weak lexical control and only 32 validation problems limit
generalization. Original test remaining unused does not make the post-collection
endpoint confirmatory. No causal intervention or deployment policy was tested.

## Integrity and provenance

Dataset revision: **`v1.0.0`**. The example resolves the tag to an immutable Hub
commit and verifies the independently pinned manifest plus every full file hash
and byte count before numerical verification.

| File | Bytes | SHA256 |
|---|---:|---|
| `train.npz` | 62236053 | `6490071a17cd7f5779de97645573db885ce55a85191758957a36c7cc782e6fc2` |
| `validation.npz` | 20746215 | `0a25de9d40bd0171122c98315e45167e3984e8c64d7aede4288d6f46d39c3f89` |
| `fitted-models.json` | 546069 | `da1b0e502732a2c779cba6b263e0df1094b744aaac5b93f454960d087b5d7843` |
| `heldout-predictions.npz` | 8610 | `e4190d59a41a4caa437044042d2628f93e79e6109d2a31324ef3869eee476507` |
| `manifest.json` | 5690 | `fed6a224ef7b6576de47f27cc2d5ef42e171999df57ffc1aa66dfb2e3f431c7f` |

Historical authority is
[reasoning-recovery-probes at `27ecb795ea8194a5fe3c3d603a6b775e7f705088`](https://github.com/mangesh-ux/reasoning-recovery-probes/tree/27ecb795ea8194a5fe3c3d603a6b775e7f705088).
Released code v1.0.0 is pinned to
[`d71422e0f59fddcbff8960c876201301b8be4b4c`](https://github.com/mangesh-ux/thinking-completion-probes/tree/d71422e0f59fddcbff8960c876201301b8be4b4c).
The separate GitHub ZIP has SHA256
`baea58d3a5156bc70bdf46f0c105cc00ba1bf92db0145051969475215d1d2383`;
it is not duplicated here. Hashes establish integrity, not scientific validity.

## CPU reproduction

Use Python **3.12** (historical stack: 3.12.3), the code's pinned CPU requirements
and the separately pinned delivery client. Linux/macOS example; on Windows use
`.venv\Scripts\Activate.ps1` for activation.

```bash
git clone https://github.com/mangesh-ux/thinking-completion-probes.git
cd thinking-completion-probes
git checkout d71422e0f59fddcbff8960c876201301b8be4b4c
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e . -r requirements/cpu.txt huggingface_hub==1.19.0
hf download mangesh-ux/thinking-completion-probes cpu_reproduce.py \
  --repo-type dataset --revision v1.0.0 --local-dir hf-example
python hf-example/cpu_reproduce.py --output runs/hf-verify-v1
```

By default, `cpu_reproduce.py` anonymously downloads only the five numeric bundle
files, restores saved models and verifies 39 historical statistics and paired
predictions. **It does not fit models**, download upstream model weights or
dataset text, run inference, collect test data or provision resources. Output is
write-once; failures and partial downloads are retained. Choose a fresh output.
Optional `--refit` runs the unchanged 640 grouped-CV plus five final TRAIN fits:

```bash
python hf-example/cpu_reproduce.py --output runs/hf-refit-v1 --refit
```

## Privacy, license and citation

No questions, reference answers, reasoning text, raw prompt/generated token IDs,
original problem IDs, source row indices, model/tokenizer files, private archives,
cloud logs, secrets, machine paths or unused test problems are released. No
identity lookup table is distributed. Anonymous group ordering preserves seeded
bootstrap behavior. Activations/features may encode source information:
**anonymization is not formal privacy or a guarantee of non-invertibility**.

Our code, documentation and generated numerical derivatives are
[MIT](https://huggingface.co/datasets/mangesh-ux/thinking-completion-probes/blob/v1.0.0/LICENSE).
This does not relicense third-party data or weights. DeepMath-103K's
[pinned dataset card](https://huggingface.co/datasets/zwhe99/DeepMath-103K/blob/5cf055d1fe3d7a2eb19719ac020211469736ae44/README.md)
declares MIT; consult its upstream-source acknowledgements and applicable terms.
Qwen3-1.7B's
[pinned model card](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md)
declares Apache-2.0. Those revision pins apply to dataset and model/tokenizer,
respectively; no current `main` substitution is implied.

Cite [CITATION.cff](https://huggingface.co/datasets/mangesh-ux/thinking-completion-probes/blob/v1.0.0/CITATION.cff),
[DeepMath-103K (He et al., arXiv:2504.11456)](https://arxiv.org/abs/2504.11456), and
[Qwen3 (Qwen team, arXiv:2505.09388)](https://arxiv.org/abs/2505.09388):

> Mangesh (mangesh-ux). *Thinking Completion Probes: Exploratory Qwen3 Activation
> Forecasting*, version 1.0.0 (2026). https://github.com/mangesh-ux/thinking-completion-probes
