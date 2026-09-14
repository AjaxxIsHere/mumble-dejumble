"""Failure analysis: bucket every failed prediction with a "why", so dataset
improvements can be targeted instead of random (spec §14).

Usage: python -m evaluation.failure_analysis predictions.jsonl --out failure_analysis.json
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from data.common import normalize, write_json
from evaluation.metrics import (
    _canonicals,
    _levenshtein,
    correction_failures,
    duplications,
    is_long,
    score_record,
)

FAILURE_BUCKETS = (
    "correction_resolution",
    "repetition_removal",
    "hallucination",
    "entity_corruption",
    "information_deletion",
    "formatting_failure",
    "long_context_failure",
    "verbosity",
    "partial_rewrite",
    "punctuation_only",
)

SUGGESTIONS = {
    "correction_resolution": "add more multi-token and multi-entity corrections; vary markers",
    "repetition_removal": "add more phrase-level and clause-echo repetitions",
    "hallucination": "add more preservation/negative examples; strengthen entity anchors",
    "entity_corruption": "add near-miss entity pairs (adjacent dates, similar numbers)",
    "information_deletion": "add more preservation cases; consider output length budget",
    "formatting_failure": "add more formatting command variants",
    "long_context_failure": "increase long-form examples; check max_seq_length coverage",
    "verbosity": "add explicit 'output only the final text' contrasts",
    "partial_rewrite": "add more no-change and punctuation-only examples",
    "punctuation_only": "add punctuation-only cleanup examples",
}


def _token_recall(expected: str, actual: str) -> float:
    # Alnum tokens: punctuation is not content ("pm." vs "pm" must match), and
    # markdown markers ("- ", "# ", "1. ") are structure, not content.
    exp = Counter(re.findall(r"[a-z0-9']+", normalize(expected)))
    act = Counter(re.findall(r"[a-z0-9']+", normalize(actual)))
    if not exp:
        return 1.0
    hits = sum(min(n, act.get(t, 0)) for t, n in exp.items())
    return hits / sum(exp.values())


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def bucket(record: dict, actual: str, scored: dict) -> str | None:
    """First-match bucket (priority order). Returns None for acceptable output."""
    expected = record["output"]
    transforms = set(record.get("transformations") or [])

    if not scored["correction_resolution"]:
        return "correction_resolution"

    if "repetition" in transforms and (duplications(actual) - duplications(expected)):
        return "repetition_removal"
    if record.get("preservation_case") and not scored["preservation_ok"]:
        return "repetition_removal"  # "deleted meaningful repetition"

    if scored["hallucination_flag"]:
        return "hallucination"

    # Formatting before content checks: a lost list/heading structure explains
    # the "missing" tokens, not information deletion.
    if record.get("formatting") and not scored["formatting_ok"]:
        return "formatting_failure"

    expected_canon = _canonicals(expected)
    actual_canon = _canonicals(actual)
    missing = expected_canon - actual_canon
    if missing and all(
        any(_levenshtein(list(e), list(a)) <= 2 for a in actual_canon) for e in missing
    ):
        return "entity_corruption"

    if (
        _token_recall(expected, actual) < 0.8
        and not (transforms & {"self_correction", "repetition"})
    ):
        return "information_deletion"

    if is_long(record) and (scored["token_ratio"] < 0.5 or scored["entity_preservation"] < 1.0):
        return "long_context_failure"

    if scored["artifact_flag"] or scored["token_ratio"] > 1.5:
        return "verbosity"

    if not scored["normalized_match"] and _alnum(expected) == _alnum(actual):
        return "punctuation_only"

    if not scored["normalized_match"] and _token_recall(expected, actual) >= 0.9:
        return "partial_rewrite"

    return None


def _why(record: dict, actual: str, scored: dict, bucket_name: str) -> str:
    if bucket_name == "correction_resolution":
        return "; ".join(scored["correction_failures"])
    if bucket_name == "repetition_removal":
        if record.get("preservation_case") and not scored["preservation_ok"]:
            return "deleted meaningful repetition"
        return f"retained duplicates: {sorted(duplications(actual) - duplications(record['output']))}"
    if bucket_name == "hallucination":
        expected_canon = _canonicals(record["output"])
        hints = []
        for h in scored["hallucinated"]:
            near = min(
                ((_levenshtein(list(h), list(e)), e) for e in expected_canon),
                default=(None, None),
            )
            hints.append(f"{h} (nearest expected: {near[1]}, distance {near[0]})")
        return "; ".join(hints)
    if bucket_name == "entity_corruption":
        missing = _canonicals(record["output"]) - _canonicals(actual)
        return f"mangled: {sorted(missing)}"
    if bucket_name == "information_deletion":
        return f"content recall {_token_recall(record['output'], actual):.2f}"
    if bucket_name == "formatting_failure":
        return f"expected type {record['formatting']['type']}"
    if bucket_name == "long_context_failure":
        return (f"token_ratio {scored['token_ratio']:.2f}, "
                f"entity_preservation {scored['entity_preservation']:.2f}")
    if bucket_name == "verbosity":
        return "artifact phrases" if scored["artifact_flag"] else f"token_ratio {scored['token_ratio']:.2f}"
    if bucket_name == "partial_rewrite":
        return f"content recall {_token_recall(record['output'], actual):.2f}"
    return ""


def analyze(rows: list[dict], max_examples: int = 20) -> dict:
    """rows: [{"record": {...}, "actual": "..."}]."""
    bucket_counts: Counter = Counter()
    examples: dict[str, list[dict]] = {b: [] for b in FAILURE_BUCKETS}
    review_rows: list[dict] = []
    for row in rows:
        record, actual = row["record"], row["actual"]
        scored = score_record(record, actual)
        b = bucket(record, actual, scored)
        if b is None:
            continue
        bucket_counts[b] += 1
        review_rows.append({
            "id": record.get("id"), "bucket": b, "input": record["input"],
            "expected": record["output"], "actual": actual,
            "why": _why(record, actual, scored, b),
        })
        if len(examples[b]) < max_examples:
            examples[b].append({
                "id": record.get("id"),
                "input": record["input"],
                "expected": record["output"],
                "actual": actual,
                "transformations": record.get("transformations"),
                "failure_type": b,
                "why": _why(record, actual, scored, b),
            })
    total_failures = sum(bucket_counts.values())
    return {
        "total_records": len(rows),
        "total_failures": total_failures,
        "failure_rate": total_failures / len(rows) if rows else 0.0,
        "bucket_counts": {b: bucket_counts[b] for b in FAILURE_BUCKETS},
        "examples": examples,
        "suggestions": {b: SUGGESTIONS[b] for b in FAILURE_BUCKETS if bucket_counts[b]},
    }


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--out")]
    out_arg = next((a.split("=", 1)[1] for a in argv[1:] if a.startswith("--out=")), None)
    if not args:
        print("usage: python -m evaluation.failure_analysis predictions.jsonl [--out=path]", file=sys.stderr)
        return 2
    rows = []
    with open(args[0], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                actual = row.pop("actual", "")
                rows.append({"record": row, "actual": actual})
    result = analyze(rows)
    if out_arg:
        write_json(Path(out_arg), result)
        review_csv = Path(out_arg).with_name(Path(out_arg).stem + "_review.csv")
        with open(review_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "bucket", "input", "expected", "actual", "why"])
            writer.writeheader()
            for bucket_name, exs in result["examples"].items():
                for ex in exs:
                    writer.writerow({
                        "id": ex["id"], "bucket": ex["failure_type"], "input": ex["input"],
                        "expected": ex["expected"], "actual": ex["actual"], "why": ex["why"],
                    })
        print(f"wrote {out_arg} ({result['total_failures']} failures)")
    else:
        print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
