"""Frozen cohort, boundary and feature definitions, without a model dependency."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Callable, Iterable, Mapping, Sequence

NUMERIC_FEATURE_NAMES = (
    "checkpoint_token_position", "normalized_checkpoint_position",
    "problem_token_count", "difficulty", "mean_answer_token_logprob",
    "min_answer_token_logprob", "answer_token_count",
    "first_readout_token_entropy_nats", "first_readout_token_probability_margin",
    "previous_readout_agreement", "previous_readout_available",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_sha(value: object) -> str:
    return sha(canonical(value))


def text_sha(value: str) -> str:
    return sha(value.encode("utf-8"))


def select_cohort(rows: Iterable[Mapping], *, selection_seed: int = 20260929,
                  split_seed: int = 20260930) -> list[dict]:
    """Fix 160 answer-independent identities before validating selected answers.

    Every source row is structurally checked. Duplicate control is shared
    across the two difficulty strata. Invalid selected answers abort the
    fixed cohort, never replace it. Answers are hashed, not persisted.
    """
    candidates = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("Dataset source row is not a mapping")
        for field in ("question", "topic"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("Dataset structural string is missing")
        difficulty = row.get("difficulty")
        if type(difficulty) not in (int, float) or not math.isfinite(difficulty):
            raise ValueError("Dataset difficulty is invalid")
        question_hash = text_sha(row["question"])
        candidates.append({
            "problem_id": f"deepmath:{index}:{question_hash[:16]}",
            "source_index": index, "difficulty": float(difficulty),
            "topic": row["topic"], "question": row["question"],
            "question_sha256": question_hash,
            "metadata_sha256": object_sha({"difficulty": float(difficulty),
                                           "topic": row["topic"]}),
            "duplicate_cluster_sha256": text_sha(" ".join(row["question"].split())),
            "_answer": row.get("final_answer"),
        })
    selected, clusters = [], set()
    for difficulty in (7.0, 8.0):
        ranked = sorted((r for r in candidates if r["difficulty"] == difficulty),
                        key=lambda r: (text_sha(f"{selection_seed}\0{difficulty:.1f}\0"
                                               f"{r['problem_id']}\0{r['source_index']}"),
                                       r["source_index"]))
        stratum = []
        for record in ranked:
            cluster = record["duplicate_cluster_sha256"]
            if cluster in clusters:
                continue
            clusters.add(cluster)
            stratum.append(dict(record))
            if len(stratum) == 80:
                break
        if len(stratum) != 80:
            raise ValueError("Cannot obtain frozen 80 unique rows per difficulty")
        selected.extend(stratum)
    invalid = sum(not isinstance(r["_answer"], str) or not r["_answer"].strip()
                  for r in selected)
    if invalid:
        raise ValueError(f"Fixed cohort invalid answers: {invalid}; replacement_count=0")
    for record in selected:
        record["reference_answer_sha256"] = text_sha(record.pop("_answer"))
    for difficulty in (7.0, 8.0):
        ranked = sorted((r for r in selected if r["difficulty"] == difficulty),
                        key=lambda r: (text_sha(f"{split_seed}\0{difficulty:.1f}\0"
                                               f"{r['duplicate_cluster_sha256']}\0{r['problem_id']}"),
                                       r["source_index"]))
        for index, record in enumerate(ranked):
            record["split"] = "train" if index < 48 else "validation" if index < 64 else "test"
            if record["split"] == "test":
                record.pop("question")  # Identity only; TEST cannot reach generation.
    return sorted(selected, key=lambda r: r["problem_id"])


def rollout_seed(problem_id: str, index: int, root: int = 20260929) -> int:
    return int(text_sha(f"{root}\0{problem_id}\0{index}")[:16], 16) % 2147483646 + 1


def fold_assignments(records: Sequence[Mapping]) -> dict[str, int]:
    output = {}
    for difficulty in (7.0, 8.0):
        ranked = sorted((r for r in records if r["split"] == "train" and
                         r["difficulty"] == difficulty),
                        key=lambda r: (text_sha("completion-forecasting-v1|fold|" +
                                               r["problem_id"]), r["problem_id"]))
        for index, record in enumerate(ranked):
            output[record["problem_id"]] = index % 5
    return output


def first_marker(tokens: Sequence[int], marker: Sequence[int]) -> int | None:
    if not marker:
        raise ValueError("Empty close marker")
    return next((i for i in range(len(tokens) - len(marker) + 1)
                 if list(tokens[i:i + len(marker)]) == list(marker)), None)


def reasoning_boundary(tokens: Sequence[int], marker: Sequence[int], eos: Sequence[int],
                       termination: str, cap: int = 4096) -> dict:
    if any(type(t) is not int or t < 0 for t in tokens) or len(tokens) > cap:
        raise ValueError("Invalid or over-budget generated token sequence")
    if set(marker) & set(eos):
        raise ValueError("Close marker intersects EOS")
    close = first_marker(tokens, marker)
    first_eos = next((i for i, t in enumerate(tokens) if t in eos), None)
    if close is not None and (first_eos is None or first_eos >= close):
        return {"status": "CLOSE_MARKER_FOUND", "terminal_position": close,
                "prefix_sha256": object_sha(list(tokens[:close])), "close_start": close}
    if close is None and termination == "MAX_NEW_TOKENS" and len(tokens) == cap:
        return {"status": "CAP_PREFIX", "terminal_position": len(tokens),
                "prefix_sha256": object_sha(list(tokens)), "close_start": None}
    return {"status": "EARLY_NO_CLOSE", "terminal_position": None,
            "prefix_sha256": None, "close_start": None}


def completion_target(boundary: Mapping, position: int = 512) -> tuple[int | None, str]:
    terminal = boundary["terminal_position"]
    if boundary["status"] == "CLOSE_MARKER_FOUND":
        return (None, "ALREADY_CLOSED") if terminal < position else (1, "CLOSE_WITHIN_4096")
    if boundary["status"] == "CAP_PREFIX":
        return 0, "NO_CLOSE_WITHIN_4096"
    return None, "UNKNOWN_OR_EARLY_TERMINATION"


def lexical_counts(prefix: Sequence[int]) -> list[int]:
    if len(prefix) != 512:
        raise ValueError("Lexical feature needs exact 512-token prefix")
    output = [0] * 512
    for token in prefix[-128:]:
        if type(token) is not int or token < 0:
            raise ValueError("Invalid lexical token")
        bucket = int(text_sha("completion-forecasting-v1|trailing-token-hash|" +
                              str(token))[:16], 16) % 512
        output[bucket] += 1
    return output


def _escaped(text: str, index: int) -> bool:
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        cursor -= 1
    return (index - cursor - 1) % 2 == 1


def boxed_boundary(text: str) -> dict:
    """Last balanced box, including nested boxes; malformed tails do not override it."""
    cursor, balanced, malformed = 0, None, None
    while True:
        start = text.find(r"\boxed", cursor)
        if start < 0:
            break
        cursor = start + len(r"\boxed")
        if cursor < len(text) and text[cursor].isalpha():
            continue
        opening = cursor
        while opening < len(text) and text[opening].isspace():
            opening += 1
        closing = None
        if opening < len(text) and text[opening] == "{":
            depth = 1
            for index in range(opening + 1, len(text)):
                if text[index] in "{}" and not _escaped(text, index):
                    depth += 1 if text[index] == "{" else -1
                    if depth == 0:
                        closing = index
                        break
        if closing is None:
            malformed = {"status": "MALFORMED_BOXED_ANSWER", "answer": None,
                         "opening": None, "closing": None}
            continue
        answer = text[start:closing + 1] if text[opening + 1:closing].strip() else None
        balanced = {"status": "EXTRACTED" if answer else "EMPTY_BOXED_ANSWER",
                    "answer": " ".join(answer.split()) if answer else None,
                    "opening": opening, "closing": closing}
    return balanced or malformed or {"status": "MISSING_BOXED_ANSWER", "answer": None,
                                      "opening": None, "closing": None}


def reduce_readout(tokens: Sequence[int], selected_logprobs: Sequence[float],
                   first_entropy: float | None, first_margin: float | None,
                   decode: Callable[[Sequence[int]], str]) -> dict:
    if len(tokens) != len(selected_logprobs):
        raise ValueError("Token/score alignment differs")
    full = decode(tokens)
    box = boxed_boundary(full)
    spans, previous, reason = [], "", None
    for end in range(1, len(tokens) + 1):
        current = decode(tokens[:end])
        if not current.startswith(previous):
            reason = "decoder_prefixes_are_not_monotonic"
            break
        spans.append((len(previous), len(current)))
        previous = current
    if previous != full and reason is None:
        reason = "prefix_decoding_does_not_reconstruct_full_text"
    included = []
    if reason is None and box["opening"] is not None:
        opening = next((i for i, (a, b) in enumerate(spans) if a <= box["opening"] < b), None)
        closing = next((i for i, (a, b) in enumerate(spans) if a <= box["closing"] < b), None)
        if opening is not None and closing is not None:
            included = list(range(opening + 1, closing))
    values = [selected_logprobs[i] for i in included]
    return {"extraction_status": box["status"], "normalized_answer": box["answer"],
            "mean_answer_token_logprob": math.fsum(values) / len(values) if values else None,
            "min_answer_token_logprob": min(values) if values else None,
            "answer_token_count": len(values) if values else None,
            "first_readout_token_entropy_nats": first_entropy,
            "first_readout_token_probability_margin": first_margin,
            "score_alignment_reason": reason}


def encode_nonfinite_scalars(scores: Mapping) -> tuple[dict, list[str]]:
    """Retain explicit per-field audit metadata when nonfinite scalars become missing."""
    converted, fields = dict(scores), []
    for field in NUMERIC_FEATURE_NAMES[4:9]:
        value = converted.get(field)
        if type(value) in (int, float) and not math.isfinite(value):
            converted[field] = None
            fields.append(field)
    return converted, fields


def numeric_features(record: Mapping, problem_tokens: int, readouts: Sequence[Mapping],
                     *, nonfinite_counts: dict | None = None) -> list:
    """Build predictors from checkpoint readouts only; no future outcome is accepted."""
    previous = None
    numeric = None
    for readout in sorted(readouts, key=lambda r: r["position"]):
        scores = readout.get("scores", {}) if readout["status"] == "COMPLETED" else {}
        answer = scores.get("normalized_answer")
        available = previous is not None and answer is not None
        if readout["position"] == 512:
            numeric = [512.0, 512 / 4096, float(problem_tokens), float(record["difficulty"])]
            numeric.extend(scores.get(name) for name in NUMERIC_FEATURE_NAMES[4:9])
            numeric.extend([float(available and answer == previous), float(available)])
        if readout["status"] == "COMPLETED":
            previous = answer  # A missing answer breaks agreement, not an available receipt.
    if numeric is None:
        raise ValueError("Primary checkpoint readout receipt is absent")
    output = []
    for name, value in zip(NUMERIC_FEATURE_NAMES, numeric, strict=True):
        if value is not None and not math.isfinite(float(value)):
            if nonfinite_counts is not None:
                nonfinite_counts[name] = nonfinite_counts.get(name, 0) + 1
            output.append(None)
        else:
            output.append(float(value) if value is not None else None)
    return output
