"""Audit the user-generated dataset in 'Model Training Pipeline Final'.

Self-contained: no imports from data_pipeline (which was removed).
Run from project root:  python3 data_pipeline/audit_user_dataset.py
"""

from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "Model Training Pipeline Final"

# ---- inline EPS (same logic as the removed eps.py) ----
_ENTITY_PATTERNS = [
    r"\b\d+(?:[.,]\d+)?\b",
    r"\b(?:mon|tue|wed|thu|fri|sat|sun)\w*day\b",
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\b",
    r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _ENTITY_PATTERNS]
_PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
_WORD_RE = re.compile(r"[a-z0-9']+")


def extract_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for rx in _COMPILED:
        for m in rx.finditer(text):
            v = m.group(0).strip()
            if v and any(c.isalnum() for c in v):
                entities.add(v.lower())
    for m in _PROPER_NAME_RE.finditer(text):
        entities.add(m.group(0).strip().lower())
    return entities


def eps(input_text: str, output_text: str) -> tuple[float, list[str]]:
    ents = extract_entities(input_text)
    if not ents:
        return 1.0, []
    out_words = set(_WORD_RE.findall(output_text.lower()))
    missing = [e for e in ents if not all(w in out_words for w in _WORD_RE.findall(e))]
    return (len(ents) - len(missing)) / len(ents), missing


def load(name: str) -> list[dict]:
    path = BASE / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    train, val, test = load("train.jsonl"), load("validation.jsonl"), load("test.jsonl")
    print(f"rows: train={len(train)} val={len(val)} test={len(test)}")

    print("\n=== UNIQUE VALUE DISTRIBUTIONS (train) ===")
    for field in ("category", "difficulty", "domain", "generator_version", "combo", "formatting", "preservation_case"):
        counter = collections.Counter(str(r.get(field)) for r in train)
        print(f"{field} -> {dict(counter.most_common(12))}")
    tcounter = collections.Counter(t for r in train for t in r.get("transformations", []))
    print(f"transformations -> {dict(tcounter.most_common(15))}")
    print(f"num_transformations -> {dict(collections.Counter(r.get('num_transformations') for r in train))}")

    print("\n=== IDENTITY / NOOP INTEGRITY ===")
    noop = [r for r in train if r["bucket"] == "noop"]
    same = sum(1 for r in noop if r["input"].strip() == r["output"].strip())
    print(f"noop bucket: {len(noop)} rows, input==output: {same} ({100 * same / max(1, len(noop)):.1f}%)")
    identical_all = sum(1 for r in train if r["input"].strip() == r["output"].strip())
    print(f"input==output across ALL train: {identical_all}")

    print("\n=== SPLIT LEAKAGE (exact normalized input match) ===")
    def keys(rows):
        return set(re.sub(r"\s+", " ", r["input"].strip().lower()) for r in rows)
    tk, vk, sk = keys(train), keys(val), keys(test)
    print(f"train∩val: {len(tk & vk)} | train∩test: {len(tk & sk)} | val∩test: {len(vk & sk)}")

    print("\n=== EMPTY / SHORT / DUPLICATE INPUTS ===")
    empties = sum(1 for r in train if not r["input"].strip() or not r["output"].strip())
    print(f"empty inputs/outputs: {empties}")
    print(f"unique train inputs: {len(tk)} / {len(train)}")
    short = sum(1 for r in train if len(r["input"].split()) < 3)
    print(f"inputs shorter than 3 words: {short}")

    print("\n=== SWITCHBOARD TAGS LEFT IN INPUTS ===")
    tagged = [r for r in train if re.search(r"\[[A-Za-z]+\]|<<", r["input"])]
    print(f"rows with [tags]/<<>>: {len(tagged)}")
    for r in tagged[:3]:
        print("  ", r["input"][:100])

    print("\n=== ENTITY PRESERVATION (EPS, 2000-row sample) ===")
    rng = random.Random(0)
    sample = rng.sample(train, min(2000, len(train)))
    total = 0.0
    bad: list[tuple[dict, list[str]]] = []
    for r in sample:
        score, missing = eps(r["input"], r["output"])
        total += score
        if score < 1.0 and r["bucket"] != "noop":
            bad.append((r, missing))
    print(f"mean EPS: {total / len(sample):.4f} | rows losing entities: {len(bad)}")
    for r, missing in bad[:8]:
        print(f"  MISS {missing} | {r['input'][:70]!r} => {r['output'][:70]!r}")

    print("\n=== CROSS-FIELD SANITY ===")
    mismatch = sum(1 for r in train if r.get("num_transformations", 0) > 0 and r["input"].strip() == r["output"].strip())
    print(f"rows with num_transformations>0 but input==output: {mismatch}")
    corr = sum(1 for r in train if r.get("corrections"))
    print(f"rows with corrections field set: {corr}")
    preserved = collections.Counter(
        (r.get("preserved"), r.get("preservation_case")) for r in train
    )
    print(f"(preserved, preservation_case) -> {dict(preserved.most_common(8))}")


if __name__ == "__main__":
    main()
