"""Plan -> record orchestration.

Metadata is derived BY CONSTRUCTION (injectors return structured deltas;
corrections always resolve to the clean output's verbatim value). The only
re-extraction is the entity inventory, computed from the well-formed clean
output. `compose()` returns None for unresolvable plans (e.g. a required
entity kind the template bank cannot provide) — the builder reallocates.
"""

import random
from dataclasses import dataclass, field

from data.common import md5_rng
from data.generators import GENERATOR_VERSION
from data.generators.entities import PoolSampler, SlotError, entity_spans
from data.generators.injectors import (
    apply_correction,
    apply_filler,
    apply_reformulation,
    apply_repetition,
    make_formatting,
)
from data.generators.longform import compose_longform
from data.generators.templates import PRESERVATION_TEMPLATES, TEMPLATES, fill_pattern, fill_slots

# Injectors run in this order: structural rewrites first on the clean text,
# surface noise last so it never corrupts a marker cluster or correction span.
CANONICAL_ORDER = ("self_correction", "reformulation", "repetition", "filler")

_ENTITY_KEY = {
    "name": "names", "item": "items", "location": "locations", "tech": "tech",
    "number": "numbers", "time": "times", "date": "dates",
}
_ALL_KINDS = ("name", "item", "location", "tech", "number", "time", "date")


@dataclass(frozen=True)
class Plan:
    key: str
    category: str
    transforms: tuple[str, ...]
    domain: str
    difficulty: str
    longform: bool = False
    preservation: bool = False
    formatting: str | None = None
    corrections: int = 1
    correction_kinds: tuple[str, ...] = ()
    identity: bool = False
    combo: str | None = None


@dataclass
class GeneratorContext:
    seed: int
    templates: dict | None = None


def _templates_for(bank: dict, domain: str, required: frozenset[str]) -> list:
    return [t for t in bank[domain] if required <= set(t.slots)]


def compose(plan: Plan, ctx: GeneratorContext) -> dict | None:
    for _ in range(10):
        record = _compose_once(plan, ctx)
        if record is not None:
            return record
    return None


def _compose_once(plan: Plan, ctx: GeneratorContext) -> dict | None:
    rng = md5_rng(f"{ctx.seed}:{plan.key}")
    bank = ctx.templates if ctx.templates is not None else TEMPLATES

    corrections: list[dict] = []
    reformulations: list[dict] = []
    preserved: list[str] = []
    format_meta = None
    unit_offsets: list[tuple[int, int]] = []
    template_id: str | None = None

    # ---- 1. clean source (the output) ----
    if plan.identity:
        sampler = PoolSampler(rng)
        template = rng.choice(_templates_for(bank, plan.domain, frozenset()))
        clean_output, _ = fill_slots(template, sampler, rng)
        current_text = clean_output
        template_id = template.id
    elif plan.preservation:
        sampler = PoolSampler(rng)
        template = rng.choice(PRESERVATION_TEMPLATES)
        clean_output, mapping = fill_slots(template, sampler, rng)
        preserved = [fill_pattern(p, mapping) for p in template.preserved_patterns]
        current_text = clean_output
        template_id = template.id
    elif plan.longform:
        result = compose_longform(plan.domain, rng, plan.difficulty)
        clean_output = result.text
        current_text = result.text
        unit_offsets = [u["offsets"] for u in result.units]
    elif plan.formatting:
        fmt = make_formatting(rng, plan.formatting)
        format_meta = {"type": fmt["type"], "items": fmt["items"]}
        clean_output = fmt["output"]
        current_text = fmt["input"]
    else:
        sampler = PoolSampler(rng)
        required = frozenset(plan.correction_kinds)
        candidates = _templates_for(bank, plan.domain, required)
        if plan.corrections >= 2:
            # multi-correction plans need at least two DISTINCT entity kinds in
            # one text, so the second correction always has a fresh kind to use
            slot_kind = {
                "item": "item", "items": "item", "item_s": "item", "item_p": "item",
                "name": "name", "name2": "name", "location": "location",
                "tech": "tech", "tech_s": "tech", "tech_p": "tech",
                "number": "number", "price": "number", "percent": "number",
                "version": "number", "duration": "number", "port": "number",
                "id": "number", "time": "time", "date": "date",
            }
            candidates = [
                t for t in candidates
                if len({slot_kind[k] for k in t.slots if k in slot_kind}) >= 2
            ]
        if not candidates:
            return None
        template = rng.choice(candidates)
        clean_output, _ = fill_slots(template, sampler, rng)
        current_text = clean_output
        template_id = template.id

    # ---- 2. injectors in canonical order ----
    protected: list[tuple[int, int]] = []
    for transform in CANONICAL_ORDER:
        if transform not in plan.transforms:
            continue
        if transform == "self_correction":
            for _ in range(plan.corrections):
                kinds = plan.correction_kinds or None
                if kinds is None and plan.corrections > 1 and corrections:
                    # force the second correction onto a different entity kind
                    used_kinds = {c["kind"] for c in corrections}
                    kinds = tuple(k for k in _ALL_KINDS if k not in used_kinds)
                try:
                    res = apply_correction(current_text, rng, plan.difficulty, kinds=kinds)
                except SlotError:
                    return None
                corrections.extend(res.delta["corrections"])
                current_text = res.text
                span = res.delta["corrections"][-1].get("span")
                if span:
                    protected.append((span[0], span[1]))
        elif transform == "reformulation":
            res = apply_reformulation(current_text, rng, plan.difficulty)
            reformulations.extend(res.delta["reformulations"])
            current_text = res.text
        elif transform == "repetition":
            prot = list(protected)
            if plan.longform and unit_offsets:
                # restrict repetition to one unit: protect every other unit
                target = rng.choice(unit_offsets)
                prot.extend(o for o in unit_offsets if o != target)
            res = apply_repetition(current_text, rng, plan.difficulty, protected=prot)
            current_text = res.text
        elif transform == "filler":
            res = apply_filler(current_text, rng, plan.difficulty, protected=protected)
            current_text = res.text

    # ---- 3. entity inventory from the clean output ----
    entities: dict[str, list[str]] = {}
    for span in entity_spans(clean_output):
        key = _ENTITY_KEY[span.kind]
        entities.setdefault(key, [])
        if span.canonical not in entities[key]:
            entities[key].append(span.canonical)
    for key in ("names", "dates", "times", "numbers", "locations", "tech", "items"):
        entities.setdefault(key, [])
        entities[key].sort()

    # ---- 4. record ----
    return {
        "input": current_text,
        "output": clean_output,
        "category": plan.category,
        "transformations": sorted(plan.transforms),
        "num_transformations": len(plan.transforms),
        "difficulty": plan.difficulty,
        "domain": plan.domain,
        "entities": entities,
        "corrections": corrections,
        "reformulations": reformulations,
        "formatting": format_meta,
        "preservation_case": plan.preservation,
        "preserved": preserved,
        "template_id": template_id,
        "combo": plan.combo,
        "seed": _seed_of(plan, ctx),
        "generator_version": GENERATOR_VERSION,
    }


def _seed_of(plan: Plan, ctx: GeneratorContext) -> int:
    import hashlib

    digest = hashlib.md5(f"{ctx.seed}:{plan.key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def entities_from_text(text: str) -> dict[str, list[str]]:
    """Entity inventory of a clean text, keyed by record metadata keys."""
    entities: dict[str, list[str]] = {}
    for span in entity_spans(text):
        key = _ENTITY_KEY[span.kind]
        entities.setdefault(key, [])
        if span.canonical not in entities[key]:
            entities[key].append(span.canonical)
    for key in ("names", "dates", "times", "numbers", "locations", "tech", "items"):
        entities.setdefault(key, [])
        entities[key].sort()
    return entities
