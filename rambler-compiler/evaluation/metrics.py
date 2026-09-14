"""Deterministic rule-judge metrics. Torch-free by design — runs on the CPU
laptop; every metric derives from record metadata, never from regex-guessing
the noisy input.

Usage: python -m evaluation.metrics predictions.jsonl --out validation_metrics.json
"""

import json
import re
import sys
from pathlib import Path

from data.common import normalize, write_json
from data.generators.entities import entity_spans
from data.generators.injectors import FILLERS

try:
    import jiwer  # optional; stdlib fallback below
except ImportError:  # pragma: no cover
    jiwer = None

_ARTIFACT_PATTERNS = (
    "here is ", "here's ", "sure!", "sure, ", "the cleaned", "cleaned version",
    "i've cleaned", "as requested", "here you go", "i hope this helps",
)

_FILLER_ALTERNATION = re.compile(
    r"\b(?:" + "|".join(sorted((re.escape(f) for f in FILLERS), key=len, reverse=True)) + r")\b"
)


# --------------------------------------------------------------------------
# String-distance helpers (stdlib fallback when jiwer is unavailable)
# --------------------------------------------------------------------------

def _levenshtein(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def levenshtein_wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / len(r)


def _levenshtein_cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(list(ref), list(hyp)) / len(ref)


def _wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    if jiwer is not None:
        try:
            return float(jiwer.wer(ref, hyp)), float(jiwer.cer(ref, hyp))
        except Exception:
            pass
    return levenshtein_wer(ref, hyp), _levenshtein_cer(ref, hyp)


# --------------------------------------------------------------------------
# Canonical value + duplication helpers
# --------------------------------------------------------------------------

def _canonicals(text: str) -> set[str]:
    return {s.canonical for s in entity_spans(text)}


def _value_canon(value: str) -> str:
    """Canonical of a correction value: the entity inside it ('version 3' -> '3')."""
    spans = entity_spans(value)
    return spans[0].canonical if spans else normalize(value)


def duplications(text: str) -> set[str]:
    """Adjacent duplicated word / 2-gram tokens in a text."""
    tokens = text.lower().split()
    dups = set()
    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i + 1] and tokens[i] not in dups:
            dups.add(tokens[i])
        if i + 3 <= len(tokens) and tokens[i:i + 2] == tokens[i + 2:i + 4]:
            dups.add(" ".join(tokens[i:i + 2]))
    return dups


def _value_present(value: str, text_norm: str, text_canon: set[str]) -> bool:
    """Is a correction value present in a text?

    Primary: entity-canonical membership (the value extracts as an entity).
    Fallback: normalized substring — hand-written stress values are sometimes
    partial phrases ("the seventh" inside "the seventh of March"). Digits and
    <3-char values stay canonical-only so "3" can't match inside "3000".
    """
    canon = _value_canon(value)
    if canon in text_canon:
        return True
    v = normalize(value)
    if len(v) < 3 or v.isdigit():
        return False
    return v in text_norm


def correction_failures(record: dict, actual_canon: set[str], expected_canon: set[str],
                        actual_norm: str = "") -> list[tuple[str, str]]:
    """Chain-aware correction resolution. Returns the list of failures.

    Chained corrections ("Tuesday no Thursday no Monday"): the intermediate
    kept value is superseded and must NOT survive. Independent corrections on
    different entities: every kept value must survive. An abandoned value may
    survive only if the expected output itself legitimately contains it
    (e.g. "John isn't available Tuesday" keeps the second Tuesday).
    """
    corrs = record.get("corrections") or []
    fails: list[tuple[str, str]] = []
    for i, c in enumerate(corrs):
        if not c.get("kept"):
            continue  # deletion-style correction ("drop the market")
        a, k = _value_canon(c["abandoned"]), _value_canon(c["kept"])
        chained = (
            i < len(corrs) - 1
            and _value_canon(corrs[i + 1]["abandoned"]) == k
        )
        if chained:
            if _value_present(c["kept"], actual_norm, actual_canon):
                fails.append(("superseded-survived", k))
        elif not _value_present(c["kept"], actual_norm, actual_canon):
            fails.append(("kept-missing", k))
        if a not in expected_canon and _value_present(c["abandoned"], actual_norm, actual_canon):
            # only flag if the expected output doesn't legitimately contain it
            if c["abandoned"].lower() not in normalize(record.get("output", "")).lower():
                fails.append(("abandoned-retained", a))
    return fails


def is_long(record: dict) -> bool:
    return record.get("category") == "long" or (record.get("combo") or "").endswith("long")


# --------------------------------------------------------------------------
# Per-record scoring
# --------------------------------------------------------------------------

def score_record(record: dict, actual: str) -> dict:
    expected = record["output"]
    m: dict = {}
    m["exact_match"] = actual == expected
    m["normalized_match"] = normalize(actual) == normalize(expected)
    m["wer"], m["cer"] = _wer_cer(expected, actual)
    m["token_ratio"] = len(actual.split()) / max(1, len(expected.split()))

    expected_canon = _canonicals(expected)
    actual_canon = _canonicals(actual)
    m["entity_preservation"] = (
        len(expected_canon & actual_canon) / len(expected_canon) if expected_canon else 1.0
    )

    fails = correction_failures(
        record, actual_canon, expected_canon, actual_norm=normalize(actual)
    )
    m["correction_resolution"] = not fails
    m["correction_failures"] = [f"{kind}:{value}" for kind, value in fails]

    abandoned_canon = {
        _value_canon(c["abandoned"]) for c in (record.get("corrections") or []) if c.get("abandoned")
    }
    m["hallucinated"] = sorted(actual_canon - expected_canon - abandoned_canon)
    m["hallucination_flag"] = bool(m["hallucinated"])

    m["verbosity_flag"] = not (0.5 <= m["token_ratio"] <= 1.5)
    m["artifact_flag"] = _artifact_check(actual)
    m["formatting_ok"] = _formatting_ok(record, actual)
    m["preservation_ok"] = all(
        normalize(sub) in normalize(actual) for sub in (record.get("preserved") or [])
    )
    m["compositional_pass"] = _compositional(record, actual, m)
    return m


def _artifact_check(actual: str) -> bool:
    low = normalize(actual)
    return any(low.startswith(p) or low.endswith(p) for p in _ARTIFACT_PATTERNS)


def _formatting_ok(record: dict, actual: str) -> bool:
    fmt = record.get("formatting")
    if not fmt:
        return True
    lines = [l for l in actual.split("\n") if l.strip()]
    ftype = fmt.get("type")
    if ftype == "bullet_list":
        return len(lines) >= 2 and all(re.match(r"^(-|\d+\.)\s", l) for l in lines)
    if ftype == "heading":
        return len(lines) == 1 and bool(re.match(r"^#{1,3}\s", lines[0]))
    if ftype == "polite_message":
        return not any(re.match(r"^(\s*(-|\d+\.)\s|#{1,3}\s)", l) for l in actual.split("\n"))
    return True


def _compositional(record: dict, actual: str, m: dict) -> bool:
    """The headline per-record check: every listed transformation must have
    actually been performed on the actual output."""
    transforms = set(record.get("transformations") or [])
    if not transforms:
        return m["normalized_match"]
    checks: list[bool] = []
    expected_norm = normalize(record["output"])
    actual_norm = normalize(actual)
    if "filler" in transforms:
        expected_fillers = set(_FILLER_ALTERNATION.findall(expected_norm))
        actual_fillers = set(_FILLER_ALTERNATION.findall(actual_norm))
        checks.append(actual_fillers <= expected_fillers)
    if "repetition" in transforms:
        checks.append(duplications(actual) <= duplications(record["output"]))
    if "self_correction" in transforms:
        checks.append(m["correction_resolution"])
    if "reformulation" in transforms:
        checks.append(_reformulation_ok(record, actual_norm))
    if "formatting" in transforms:
        checks.append(m["formatting_ok"])
    if is_long(record):
        checks.append(m["entity_preservation"] == 1.0 and m["token_ratio"] >= 0.5)
    return all(checks)


def _reformulation_ok(record: dict, actual_norm: str) -> bool:
    for ref in (record.get("reformulations") or []):
        a_norm = normalize(ref.get("abandoned", ""))
        if not a_norm:
            continue
        if ref.get("mode") == "restart":
            # the abandoned prefix legitimately survives once inside the kept
            # text; a failure is the duplication remaining unresolved
            if actual_norm.count(a_norm) > 1:
                return False
        elif a_norm in actual_norm:
            return False
    return True


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _mean(values: list) -> float | None:
    if not values:
        return None
    nums = [float(v) if isinstance(v, bool) else float(v) for v in values]
    return sum(nums) / len(nums)


def _group(scored: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in scored:
        value = row["record"].get(key)
        groups.setdefault(str(value), []).append(row)
    return {k: _means_rows(v) for k, v in sorted(groups.items())}


def _means_rows(rows: list[dict]) -> dict:
    out = {}
    for metric in ("exact_match", "normalized_match", "wer", "cer", "entity_preservation",
                   "correction_resolution", "hallucination_flag", "verbosity_flag",
                   "artifact_flag", "formatting_ok", "preservation_ok", "compositional_pass"):
        out[metric] = _mean([r[metric] for r in rows])
    out["n"] = len(rows)
    return out


def compositional_accuracy(scored: list[dict]) -> dict:
    """The headline metric: mean compositional_pass over mixed-category records,
    also broken down per combination."""
    mixed = [r for r in scored if r["record"].get("category") == "mixed"]
    per_combo: dict[str, float | None] = {}
    combos: dict[str, list[dict]] = {}
    for row in mixed:
        combos.setdefault(str(row["record"].get("combo")), []).append(row)
    for combo, rows in sorted(combos.items()):
        per_combo[combo] = _mean([r["compositional_pass"] for r in rows])
    return {
        "overall": _mean([r["compositional_pass"] for r in mixed]),
        "n_mixed": len(mixed),
        "per_combo": per_combo,
    }


def aggregate(scored: list[dict]) -> dict:
    per_record = []
    for row in scored:
        entry = {k: v for k, v in row.items() if k != "record"}
        entry["id"] = row["record"].get("id")
        entry["category"] = row["record"].get("category")
        entry["combo"] = row["record"].get("combo")
        entry["level"] = row["record"].get("level")
        per_record.append(entry)
    return {
        "overall": _means_rows(scored),
        "per_category": _group(scored, "category"),
        "per_difficulty": _group(scored, "difficulty"),
        "per_combo": _group(scored, "combo"),
        "compositional": compositional_accuracy(scored),
        "per_record": per_record,
    }


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--out")]
    out_arg = next((a.split("=", 1)[1] for a in argv[1:] if a.startswith("--out=")), None)
    if not args:
        print("usage: python -m evaluation.metrics predictions.jsonl [--out=path]", file=sys.stderr)
        return 2
    rows = []
    with open(args[0], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    scored = [
        {"record": {k: v for k, v in row.items() if k != "actual"},
         **score_record({k: v for k, v in row.items() if k != "actual"}, row.get("actual", ""))}
        for row in rows
    ]
    result = aggregate(scored)
    if out_arg:
        write_json(Path(out_arg), result)
        print(f"wrote {out_arg}")
    else:
        print(json.dumps({k: v for k, v in result.items() if k != "per_record"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
