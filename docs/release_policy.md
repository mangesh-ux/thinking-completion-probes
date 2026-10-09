# Artifact release and upstream attribution

This new publication release is distinct from the original private research
archive. Historical files, labels and conclusions are not modified. The numerical
derivative release is authorized by the repository-publication task; it is not
a license to publish the original raw scientific archive.

## Included

The derivative bundle contains checkpoint-512 numeric confidence/structural
features, fixed hashed prefix-token counts, all 28 residual vectors, binary
completion labels, frozen training-fold assignments, categorical topics,
anonymized grouping keys and row identifiers. It also contains frozen linear
model states and saved held-out probabilities. BF16 activation values are
represented losslessly as float32; no new inference or labels are introduced.

Sorted anonymous group keys preserve the original bootstrap ordering. They do
not permit reconstruction of the raw problem ID from a published lookup table;
no such table is released. Features can nevertheless encode information about
their source, so this is research data—not a claim of formal privacy protection.

## Not included

Questions, reference answers, natural reasoning text, raw prompt/generated token
IDs, original problem IDs, source row indices, cloud logs, secrets, machine paths,
the private recovery archive, the 32 ungenerated original test problems and model
weights are not distributed. Download the upstream model and dataset directly
for regeneration. Use the recorded revisions; do not substitute current `main`.

## Terms and attribution

Our software, documentation and generated numerical derivative bundle are
released under the repository MIT license. This does not relicense third-party
data or weights. Keep this notice with redistributed bundles.

- Dataset: zwhe99/DeepMath-103K, revision
  `5cf055d1fe3d7a2eb19719ac020211469736ae44`. Its
  [pinned dataset card](https://huggingface.co/datasets/zwhe99/DeepMath-103K/blob/5cf055d1fe3d7a2eb19719ac020211469736ae44/README.md)
  declares MIT. The dataset combines several upstream sources; users should
  consult its acknowledgements and applicable source terms before redistributing
  regenerated problem content. We distribute no source text or reference answers.
- Model: Qwen/Qwen3-1.7B, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Its
  [pinned model card](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md)
  declares Apache-2.0. We distribute neither weights nor tokenizer files.
- Cite DeepMath-103K (He et al., arXiv:2504.11456), Qwen3 (Qwen team,
  arXiv:2505.09388), and this repository when building on the study.

The release manifest records original input hashes separately from newly
serialized/anonymized file hashes. Equality of scientific values is verified;
byte equality to the original NPZ archives is deliberately not asserted.
