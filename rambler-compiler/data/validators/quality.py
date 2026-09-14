"""Gate checks V1-V10 and dataset statistics. Pure stdlib — never imports torch."""

import re
from collections import Counter
from dataclasses import dataclass

from data.common import CATEGORIES, TRANSFORMS, normalize
from data.generators.entities import (
    FIRST_NAMES,
    ITEMS,
    LOCATIONS,
    TECH_TERMS,
    entity_spans,
)

REQUIRED_KEYS = (
    "input", "output", "category", "transformations", "num_transformations", "difficulty",
    "domain", "entities", "corrections", "reformulations", "formatting", "preservation_case",
    "preserved", "combo", "template_id", "seed", "generator_version",
)
ENTITY_KEYS = ("names", "dates", "times", "numbers", "locations", "tech", "items")
META_KEY = {
    "name": "names", "item": "items", "location": "locations", "tech": "tech",
    "number": "numbers", "time": "times", "date": "dates",
}


@dataclass
class Issue:
    check: str  # "V1".."V10"
    severity: str  # "error" | "warning"
    detail: str


def _markdown_lines(output: str) -> list[str]:
    return [l for l in output.split("\n") if re.match(r"^(\s*(-|\d+\.)\s|#{1,3}\s)", l)]


def validate_record(record: dict, matrix: frozenset[str] = frozenset()) -> list[Issue]:
    issues: list[Issue] = []

    # V1 structure
    for key in REQUIRED_KEYS:
        if key not in record:
            issues.append(Issue("V1", "error", f"missing key {key!r}"))
    if record.get("category") not in CATEGORIES:
        issues.append(Issue("V1", "error", f"bad category {record.get('category')!r}"))
    if record.get("difficulty") not in ("easy", "medium", "hard"):
        issues.append(Issue("V1", "error", f"bad difficulty {record.get('difficulty')!r}"))
    transforms = tuple(record.get("transformations") or ())
    if record.get("num_transformations") != len(transforms):
        issues.append(Issue("V1", "error", "num_transformations != len(transformations)"))
    if not set(transforms) <= TRANSFORMS:
        issues.append(Issue("V1", "error", f"unknown transforms {set(transforms) - TRANSFORMS}"))
    if record.get("category") == "mixed":
        if record.get("combo") not in matrix:
            issues.append(Issue("V1", "error", f"mixed record combo {record.get('combo')!r} not in matrix"))
    elif record.get("combo") is not None:
        issues.append(Issue("V1", "error", "combo set on a non-mixed record"))

    input_, output = record.get("input", ""), record.get("output", "")

    # V2 non-identity
    if input_ == output and transforms:
        issues.append(Issue("V2", "error", "input == output despite transformations"))

    # V3 non-empty / no stray whitespace / standard sentence capitalization
    if len(input_.split()) < 3 or len(output.split()) < 3:
        issues.append(Issue("V3", "error", "input/output too short"))
    if input_ != input_.strip() or output != output.strip():
        issues.append(Issue("V3", "error", "leading/trailing whitespace"))
    if output and output[0].islower():
        issues.append(Issue("V3", "error", "output starts with a lowercase letter"))

    # V4 correction consistency (the core invariant)
    in_spans = {s.canonical for s in entity_spans(input_)}
    out_spans = {s.canonical for s in entity_spans(output)}
    for c in record.get("corrections") or []:
        a, k = normalize(c["abandoned"]), normalize(c["kept"])
        if a == k:
            issues.append(Issue("V4", "error", f"abandoned == kept ({a})"))
        if a not in in_spans:
            issues.append(Issue("V4", "error", f"abandoned {a!r} not extractable from input"))
        if k not in out_spans:
            issues.append(Issue("V4", "error", f"kept {k!r} not extractable from output"))
        if a in out_spans:
            issues.append(Issue("V4", "error", f"abandoned {a!r} leaked into output"))

    # V5 entity preservation (metadata == extraction, presence-wise)
    meta = record.get("entities") or {}
    meta_set = {v for key in ENTITY_KEYS for v in (meta.get(key) or [])}
    extracted = {s.canonical for s in entity_spans(output)}
    if meta_set != extracted:
        issues.append(
            Issue("V5", "error",
                  f"metadata mismatch: only-meta={sorted(meta_set - extracted)}, "
                  f"only-extracted={sorted(extracted - meta_set)}")
        )

    # V6 formatting validity
    fmt = record.get("formatting")
    lines = [l for l in output.split("\n") if l.strip()]
    if fmt:
        ftype = fmt.get("type")
        if ftype == "bullet_list":
            if len(lines) < 2 or not all(re.match(r"^(-|\d+\.)\s", l) for l in lines):
                issues.append(Issue("V6", "error", "bullet_list output is not a valid list"))
        elif ftype == "heading":
            if len(lines) != 1 or not re.match(r"^#{1,3}\s", lines[0]):
                issues.append(Issue("V6", "error", "heading output is not a single heading"))
        elif ftype == "polite_message":
            if _markdown_lines(output):
                issues.append(Issue("V6", "error", "polite_message output contains markdown"))
        else:
            issues.append(Issue("V6", "error", f"unknown formatting type {ftype!r}"))
    elif _markdown_lines(output):
        issues.append(Issue("V6", "error", "markdown structure in a non-formatting output"))

    # V7 length bounds (longform is flagged by category OR combo, since mixed
    # combos like rep+corr+long carry longform monologues under category "mixed")
    words = len(output.split())
    is_long = record.get("category") == "long" or (record.get("combo") or "").endswith("long")
    if is_long:
        if not (50 <= words <= 300):
            issues.append(Issue("V7", "error", f"long output has {words} words"))
    elif not transforms and words > 30:
        issues.append(Issue("V7", "error", f"identity output has {words} words"))
    elif words > 60:
        issues.append(Issue("V7", "error", f"output has {words} words"))

    # V10 preservation
    if record.get("preservation_case"):
        if not record.get("preserved"):
            issues.append(Issue("V10", "error", "preservation record has no preserved substrings"))
        for sub in record.get("preserved") or []:
            if sub not in output:
                issues.append(Issue("V10", "error", f"preserved {sub!r} missing from output"))

    return issues


# --------------------------------------------------------------------------
# V8 dedup (batch-level)
# --------------------------------------------------------------------------

def _shingles(text: str, n: int = 3) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _entity_signature(r: dict) -> tuple[str, ...]:
    meta = r.get("entities") or {}
    return tuple(sorted(v for key in ENTITY_KEYS for v in (meta.get(key) or [])))


def find_near_dups(records: list[dict], threshold: float = 0.85) -> set[str]:
    """Ids of records whose (input+output) 3-gram Jaccard vs another record in
    the same group is >= threshold.

    Groups are (template_id, entity-signature): the same template with the SAME
    entity fills is a true near-dup, while the same template with different
    entities is legitimate lexical variety and is not compared. Grouped (not
    global pairwise) to stay O(n) in practice.
    """
    offenders: set[str] = set()
    groups: dict = {}
    for r in records:
        if r.get("template_id"):
            group = (r["template_id"], _entity_signature(r))
        else:
            group = (r["category"], r["domain"], _entity_signature(r))
        groups.setdefault(group, []).append(r)
    for recs in groups.values():
        shingle_cache = [(_shingles(r["input"]) | _shingles(r["output"]), r.get("id")) for r in recs]
        for i in range(len(recs)):
            a, id_a = shingle_cache[i]
            for j in range(i + 1, len(recs)):
                b, id_b = shingle_cache[j]
                if _jaccard(a, b) >= threshold:
                    offenders.add(id_a)
                    offenders.add(id_b)
    return offenders


def dedup_issues(records: list[dict], threshold: float = 0.85) -> list[Issue]:
    issues: list[Issue] = []
    seen: dict[tuple[str, str], str] = {}
    for r in records:
        pair = (r["input"], r["output"])
        if pair in seen:
            issues.append(Issue("V8", "error", f"exact duplicate of {seen[pair]}"))
        seen[pair] = r.get("id", "?")
    for rid in find_near_dups(records, threshold):
        issues.append(Issue("V8", "error", f"near-duplicate record {rid}"))
    return issues


# --------------------------------------------------------------------------
# V9 vocab coverage (anti-association)
# --------------------------------------------------------------------------

def _word_in(entry: str, text_lower: str) -> bool:
    return re.search(r"(?<![A-Za-z])" + re.escape(entry) + r"(?![A-Za-z])", text_lower) is not None


def coverage_issues(records: list[dict], floors: dict | None = None) -> list[Issue]:
    """Every enumerated pool entry must appear >= floors['kept'] times as a kept
    value (any clean output) and >= floors['abandoned'] times as an abandoned
    correction value — so no word is ever "the correction word"."""
    if not floors:
        return []
    kept_floor = floors.get("kept", 5)
    abandoned_floor = floors.get("abandoned", 3)
    kept: Counter = Counter()
    abandoned: Counter = Counter()
    outputs = [r["output"].lower() for r in records]
    for entry in FIRST_NAMES + ITEMS + LOCATIONS + TECH_TERMS:
        low = entry.lower()
        kept[low] = sum(1 for out in outputs if _word_in(low, out))
    for r in records:
        for c in r.get("corrections") or []:
            abandoned[c["abandoned"].lower()] += 1

    issues: list[Issue] = []
    for pool in (FIRST_NAMES, ITEMS, LOCATIONS, TECH_TERMS):
        for entry in pool:
            low = entry.lower()
            if kept[low] < kept_floor:
                issues.append(Issue("V9", "error", f"kept value {entry!r} appears {kept[low]}x (< {kept_floor})"))
            if abandoned[low] < abandoned_floor:
                issues.append(
                    Issue("V9", "warning", f"abandoned value {entry!r} appears {abandoned[low]}x (< {abandoned_floor})")
                )
    return issues


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _percentiles(values: list[int]) -> dict:
    if not values:
        return {"p25": 0, "p50": 0, "p75": 0, "p90": 0, "p99": 0, "max": 0}
    s = sorted(values)

    def pct(p):
        idx = min(len(s) - 1, int(len(s) * p))
        return s[idx]

    return {"p25": pct(0.25), "p50": pct(0.50), "p75": pct(0.75),
            "p90": pct(0.90), "p99": pct(0.99), "max": s[-1]}


def dataset_stats(records: list[dict], extra: dict | None = None) -> dict:
    def wc(t):
        return len(t.split())

    filler_by_diff: dict[str, list[int]] = {}
    for r in records:
        if "filler" in r.get("transformations") or ():
            from data.generators.injectors import FILLERS

            alternation = "|".join(sorted((re.escape(f) for f in FILLERS), key=len, reverse=True))
            n = len(re.findall(r"\b(?:" + alternation + r")\b", r["input"].lower()))
            filler_by_diff.setdefault(r["difficulty"], []).append(n)

    stats = {
        "total": len(records),
        "per_category": dict(Counter(r["category"] for r in records)),
        "per_combo": dict(Counter(r.get("combo") for r in records)),
        "per_difficulty": dict(Counter(r["difficulty"] for r in records)),
        "per_domain": dict(Counter(r["domain"] for r in records)),
        "marker_histogram": dict(
            Counter(c["marker"] for r in records for c in (r.get("corrections") or []))
        ),
        "correction_kinds": dict(
            Counter(c["kind"] for r in records for c in (r.get("corrections") or []))
        ),
        "formatting_types": dict(
            Counter(r["formatting"]["type"] for r in records if r.get("formatting"))
        ),
        "input_word_percentiles": _percentiles([wc(r["input"]) for r in records]),
        "output_word_percentiles": _percentiles([wc(r["output"]) for r in records]),
        "avg_num_transformations": sum(r["num_transformations"] for r in records) / max(1, len(records)),
        "preservation_count": sum(1 for r in records if r.get("preservation_case")),
        "identity_count": sum(1 for r in records if not r.get("transformations")),
        "avg_fillers_by_difficulty": {
            d: sum(v) / len(v) for d, v in filler_by_diff.items()
        },
    }
    if extra:
        stats.update(extra)
    return stats
