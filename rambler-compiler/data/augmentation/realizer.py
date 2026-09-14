"""Rule-based recipe realizer (V4).

Realizes transformation recipes over ANY clean text — real speech (Nyra
intended transcripts) or template-generated text — using the deterministic
noise injectors in data/generators/injectors.py.

The clean target is NEVER invented: the caller supplies the authoritative
clean sentence, the realizer only adds the requested speech phenomena
(fillers, repetitions, self-corrections, reformulations/restarts) on top of
it. No external APIs, no network, no cost — the same seed reproduces the
same output byte-for-byte.

Supported transformations (a subset of V4_TRANSFORMS that the injectors can
realize): filler, repetition, self_correction, reformulation. Formatting
records are produced separately by `make_formatting` (see dataset_builder),
because a formatting example is a spoken command -> artifact pair, not a
noisy version of a clean sentence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from data.common import V4_GENERATOR_VERSION, md5_rng
from data.generators.composer import entities_from_text
from data.generators.entities import SlotError, entity_spans
from data.generators.injectors import (
    apply_correction,
    apply_filler,
    apply_reformulation,
    apply_repetition,
)

# Injectors run in this order: structural rewrites first on the clean text,
# surface noise last so it never corrupts a marker cluster or correction span.
# (Same canonical order as the template composer.)
CANONICAL_ORDER = ("self_correction", "reformulation", "repetition", "filler")

_ALL_KINDS = ("name", "item", "location", "tech", "number", "time", "date")

_REALIZABLE = frozenset(CANONICAL_ORDER)


class RealizationError(Exception):
    """Raised for recipes the realizer cannot express."""


@dataclass
class RealizedRecord:
    """The parts of a realized record the builder merges with plan metadata."""

    input: str
    output: str  # the clean text, unchanged
    entities: dict[str, list[str]]
    corrections: list[dict]
    reformulations: list[dict]
    seed: int


def realize(
    clean_text: str,
    *,
    transforms: tuple[str, ...],
    difficulty: str,
    seed_key: str,
    corrections: int = 1,
) -> RealizedRecord | None:
    """Apply `transforms` to `clean_text` with the deterministic injectors.

    Args:
        clean_text: Authoritative clean sentence (never modified).
        transforms: Realizable transformations (see CANONICAL_ORDER).
        difficulty: "easy" | "medium" | "hard" — controls injector intensity.
        seed_key: Stable string key; the same key + text reproduces the
            same realization byte-for-byte.
        corrections: How many self_corrections to perform when
            "self_correction" is in transforms.

    Returns:
        A RealizedRecord, or None when the recipe cannot be realized on this
        text (e.g. a self-correction needs an extractable entity the text
        does not contain). The caller reallocates or counts a gap.
    """
    unknown = set(transforms) - _REALIZABLE
    if unknown:
        raise RealizationError(
            f"transformations not realizable by the rule engine: {sorted(unknown)}"
        )
    if "self_correction" in transforms and corrections < 1:
        raise RealizationError("self_correction needs corrections >= 1")

    rng = md5_rng(seed_key)
    current_text = clean_text
    correction_meta: list[dict] = []
    reformulation_meta: list[dict] = []
    protected: list[tuple[int, int]] = []

    for transform in CANONICAL_ORDER:
        if transform not in transforms:
            continue
        if transform == "self_correction":
            for _ in range(corrections):
                # A second correction must land on a different entity kind so
                # two independent corrections never fight over one value.
                kinds = None
                if corrections > 1 and correction_meta:
                    used_kinds = {c["kind"] for c in correction_meta}
                    kinds = tuple(k for k in _ALL_KINDS if k not in used_kinds)
                try:
                    res = apply_correction(current_text, rng, difficulty, kinds=kinds)
                except SlotError:
                    return None
                correction_meta.extend(res.delta["corrections"])
                current_text = res.text
                span = res.delta["corrections"][-1].get("span")
                if span:
                    protected.append((span[0], span[1]))
        elif transform == "reformulation":
            res = apply_reformulation(current_text, rng, difficulty)
            reformulation_meta.extend(res.delta["reformulations"])
            current_text = res.text
        elif transform == "repetition":
            res = apply_repetition(current_text, rng, difficulty, protected=protected)
            current_text = res.text
        elif transform == "filler":
            res = apply_filler(current_text, rng, difficulty, protected=protected)
            current_text = res.text

    digest = hashlib.md5(seed_key.encode("utf-8")).digest()
    return RealizedRecord(
        input=current_text,
        output=clean_text,
        entities=entities_from_text(clean_text),
        corrections=correction_meta,
        reformulations=reformulation_meta,
        seed=int.from_bytes(digest[:8], "big"),
    )


def entity_kinds(text: str) -> set[str]:
    """Distinct entity kinds extractable from a text.

    The builder uses this to decide whether a clean text can carry a
    self-correction (and how many independent ones).
    """
    return {s.kind for s in entity_spans(text)}


def to_record(realized: RealizedRecord, *, plan: dict) -> dict:
    """Wrap a RealizedRecord in the canonical record schema.

    `plan` supplies: category, transforms, num_transformations, difficulty,
    domain, combo, bucket, source, template_id (None for real text).
    """
    return {
        "input": realized.input,
        "output": realized.output,
        "category": plan["category"],
        "transformations": sorted(plan["transforms"]),
        "num_transformations": len(plan["transforms"]),
        "difficulty": plan["difficulty"],
        "domain": plan["domain"],
        "entities": realized.entities,
        "corrections": realized.corrections,
        "reformulations": realized.reformulations,
        "formatting": None,
        "preservation_case": False,
        "preserved": [],
        "combo": plan.get("combo"),
        "template_id": None,
        "seed": realized.seed,
        "generator_version": V4_GENERATOR_VERSION,
        "bucket": plan["bucket"],
        "source": plan["source"],
    }
