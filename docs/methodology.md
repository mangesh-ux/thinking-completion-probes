# Methodology and scientific boundaries

## Original recovery endpoint

The frozen original V2-A1 asked whether confidence-plus-activations predicted
wrong→correct recovery better than confidence alone. Its collection was complete:
768 base trajectories, 768 terminal readouts, 3840 checkpoint/readout slots and
3793 available activation matrices, with zero interrupted/failed collection
operations. Unavailable slots were preserved. Selection could not form eligible
wrong→correct/wrong→wrong populations; no recovery probes or original test
evaluation were completed. Operational completeness is not scientific eligibility.

The bounded 64-token greedy readouts frequently lacked usable boxed answers;
continued thinking and extraction limitations were plausible measurement issues,
not proof that the reasoning state was wrong. A separate fixed TRAIN-only
reference-parsing diagnostic yielded only 4 nonempty parses out of 12, failing
its 10/12 and per-stratum 5/6 gates. Therefore a larger readout budget alone would
not establish correct-answer labels. No GPU measurement repair was executed.

## Reused completion endpoint

The subsequently locked exploratory protocol used natural close-marker outcomes
already recorded in the base trajectories, not counterfactual readout correctness.
No endpoint definition is selected anew in this reproduction repository.
TRAIN had 151 positive/425 negative rows across 96 problems; validation had
67/125 across 32. Positive/negative problem support overlapped: TRAIN 52/90
problems (46 mixed), validation 20/29 (17 mixed).

Prefix predictors exclude eventual answer, correctness, close position,
termination reason and remaining-token count. The natural trajectory provides
labels only after predictor construction. Historical feature reconstruction was
not a live deployment; source ordering and prefix-only inputs, rather than
physical wall-clock extraction order, support the leakage claim.

## Modeling and uncertainty

Each problem has equal total weight. Within a fit partition, row weight is
`(number of rows / number of problems) / rows for that problem`. Numeric medians
use observed TRAIN values; all-missing columns receive zero and a missingness
indicator. Topic encoding ignores unseen evaluation topics. Weighted variance
must exceed `1e-12` to retain a feature; scales are weighted TRAIN standard
deviations. Every fold learns its own preprocessing.

Stored TRAIN folds are preserved through numeric anonymization; rehashing public
IDs would silently change model selection. Regularization and layers are selected
from 128 fixed configurations (640 CV fits), followed by five final TRAIN fits.
Selection-CV loss is not an unbiased held-out estimate. Validation never fits
preprocessing, a layer or a coefficient. A serialized selection reserves exactly
one evaluation attempt; failed output is preserved.

Paired log loss is baseline-minus-added, averaged within each problem, then
across problems. The bootstrap samples problems with replacement, retaining
duplicate occurrences. Resamples missing a class are counted and excluded under
the frozen support policy; all historical 2000 were valid. Mixed-outcome
concordance compares positive/negative trajectories within a problem with
half-credit ties. This does not turn 192 rows into 192 independent problems.

The primary comparison controls for fixed hashed prefix counts. Secondary
comparisons and matched analyses are descriptive, without multiplicity correction.
Intervals do not include uncertainty from training, layer selection or the
exploratory change of research question. Original test was not used, but that
alone does not make a post-collection endpoint a confirmatory trial.

## What remains open

Completion may encode recognizable token patterns, style, or propensity to
respect a cap rather than mathematical progress. Tests of causal interventions,
cross-model transfer, uncapped horizon behavior, prospective extraction and
actual accuracy-preserving stopping are absent. Repairing correctness requires
both defensible checkpoint elicitation and independently validated reference
evaluation, with train gates before any confirmation or new held-out collection.
These are proposals for new authorized studies, not results of this one.
