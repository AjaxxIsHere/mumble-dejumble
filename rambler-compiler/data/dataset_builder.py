"""Dataset builder: quota allocation -> plan list -> compose -> gate validation
-> dedup -> stratified split -> JSONL + manifest + stats.

Two build paths:
- `build()` — the V2 fully-synthetic builder (template clean text + noise
  injectors, eight-category distribution). Kept as the deterministic legacy
  path; still pinned by tests/test_builder.py::TestReproducibility.
- `build_v4()` — the V4 hybrid builder: clean text comes from REAL dictated
  speech (Nyra intended transcripts) where possible and from the template
  engine elsewhere, and recipes are realized by the deterministic rule
  engine (no LLM, no API). Targets the five-bucket distribution the
  abandoned paid-API experiment planned: 20% no-op / 15% simple disfluencies
  / 25% corrections & restarts / 30% compositional / 10% formatting.

Both paths are fully deterministic: the same config reproduces byte-identical
splits.
"""

import random
from dataclasses import dataclass, field, replace
from pathlib import Path

from data.common import V4_GENERATOR_VERSION, md5_rng, parse_dataset_version, write_json, write_jsonl
from data.generators import GENERATOR_VERSION
from data.generators.composer import GeneratorContext, Plan, compose
from data.generators.templates import DOMAINS
from data.validators.quality import (
    coverage_issues,
    dataset_stats,
    find_near_dups,
    validate_record,
)

DEFAULT_MATRIX: dict[str, int] = {
    "filler+rep": 225,
    "filler+corr": 225,
    "rep+corr": 225,
    "rep+reform": 225,
    "corr+fmt": 225,
    "filler+rep+corr": 225,
    "filler+corr+fmt": 225,
    "rep+corr+long": 225,
    "filler+rep+corr+fmt": 225,
    "multi-corr": 225,
}

# combo id -> (transforms, longform, formatting, corrections)
_COMBO_SPECS: dict[str, tuple[tuple[str, ...], bool, str | None, int]] = {
    "filler+rep": (("filler", "repetition"), False, None, 1),
    "filler+corr": (("filler", "self_correction"), False, None, 1),
    "rep+corr": (("repetition", "self_correction"), False, None, 1),
    "rep+reform": (("repetition", "reformulation"), False, None, 1),
    "corr+fmt": (("self_correction", "formatting"), False, "random", 1),
    "filler+rep+corr": (("filler", "repetition", "self_correction"), False, None, 1),
    "filler+corr+fmt": (("filler", "self_correction", "formatting"), False, "random", 1),
    "rep+corr+long": (("repetition", "self_correction"), True, None, 1),
    "filler+rep+corr+fmt": (("filler", "repetition", "self_correction", "formatting"), False, "random", 1),
    "multi-corr": (("self_correction", "self_correction"), False, None, 2),
}


@dataclass
class BuildConfig:
    total: int = 15000
    distribution: dict[str, float] = field(default_factory=lambda: {
        "basic": 0.15, "filler": 0.10, "repetition": 0.10, "correction": 0.20,
        "reformulation": 0.10, "formatting": 0.05, "long": 0.15, "mixed": 0.15,
    })
    composition_matrix: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MATRIX))
    preservation_ratio: float = 0.10
    identity_ratio: float = 0.15
    difficulty_weights: dict[str, tuple[float, float, float]] = field(default_factory=lambda: {
        "basic": (0.60, 0.30, 0.10),
        "filler": (0.55, 0.30, 0.15),
        "repetition": (0.45, 0.35, 0.20),
        "correction": (0.30, 0.45, 0.25),
        "reformulation": (0.45, 0.35, 0.20),
        "formatting": (0.60, 0.30, 0.10),
        "long": (0.25, 0.45, 0.30),
        "mixed": (0.20, 0.40, 0.40),
    })
    seed: int = 42
    dataset_version: str = "20260901.2"
    coverage_floors: dict | None = field(default_factory=lambda: {"kept": 5, "abandoned": 3})
    retry_budget: int = 5


def _validate_config(cfg: BuildConfig) -> None:
    total_p = sum(cfg.distribution.values())
    assert abs(total_p - 1.0) < 1e-9, f"distribution sums to {total_p}"
    mixed_count = round(cfg.total * cfg.distribution.get("mixed", 0.0))
    assert sum(cfg.composition_matrix.values()) == mixed_count, (
        f"matrix sums to {sum(cfg.composition_matrix.values())}, mixed quota is {mixed_count}"
    )


def _allocate_plans(cfg: BuildConfig) -> list[Plan]:
    counts = {cat: round(cfg.total * p) for cat, p in cfg.distribution.items()}
    counts["mixed"] += cfg.total - sum(counts.values())  # absorb rounding

    difficulty_rng = random.Random(cfg.seed)

    def difficulty(category: str) -> str:
        return difficulty_rng.choices(("easy", "medium", "hard"),
                                      weights=cfg.difficulty_weights[category])[0]

    plans: list[Plan] = []
    counter = 0

    def add(category, transforms, *, combo=None, longform=False, formatting=None,
            corrections=1, identity=False, preservation=False, n=1, force_diff=None):
        nonlocal counter
        for _ in range(n):
            domain = DOMAINS[counter % len(DOMAINS)]
            diff = force_diff or difficulty(category)
            prefix = f"pres:{category}" if preservation else category
            if combo:
                key = f"mixed:{combo}:{counter:04d}"
            else:
                key = f"{prefix}:{counter:04d}"
            plans.append(Plan(
                key=key, category=category, transforms=transforms, domain=domain,
                difficulty=diff, longform=longform, preservation=preservation,
                formatting=formatting, corrections=corrections, identity=identity,
                combo=combo,
            ))
            counter += 1

    # basic: identity + minimal noise
    n_basic = counts["basic"]
    n_identity = round(n_basic * cfg.identity_ratio)
    add("basic", (), identity=True, n=n_identity)
    rest = n_basic - n_identity
    for i in range(rest):
        add("basic", ("filler",) if i % 3 < 2 else ("repetition",), force_diff="easy")

    # filler
    add("filler", ("filler",), n=counts["filler"])

    # repetition (minus the preservation sub-stream)
    n_rep = counts["repetition"]
    add("repetition", ("repetition",), n=round(n_rep * (1 - cfg.preservation_ratio)))
    add("repetition", ("filler",), preservation=True, n=n_rep - round(n_rep * (1 - cfg.preservation_ratio)),
        force_diff="easy")

    # correction
    n_corr = counts["correction"]
    add("correction", ("self_correction",), n=round(n_corr * (1 - cfg.preservation_ratio)))
    add("correction", ("filler",), preservation=True, n=n_corr - round(n_corr * (1 - cfg.preservation_ratio)),
        force_diff="easy")

    # reformulation
    n_reform = counts["reformulation"]
    add("reformulation", ("reformulation",), n=round(n_reform * (1 - cfg.preservation_ratio)))
    add("reformulation", ("filler",), preservation=True,
        n=n_reform - round(n_reform * (1 - cfg.preservation_ratio)), force_diff="easy")

    # formatting
    add("formatting", ("formatting",), formatting="random", n=counts["formatting"])

    # long
    n_long = counts["long"]
    for i in range(n_long):
        transforms = ("filler",) if i % 3 < 2 else ("filler", "repetition")
        add("long", transforms, longform=True)

    # mixed: explicit composition matrix
    for combo, n in cfg.composition_matrix.items():
        transforms, longform, formatting, corrections = _COMBO_SPECS[combo]
        add("mixed", transforms, combo=combo, longform=longform, formatting=formatting,
            corrections=corrections, n=n)

    return plans


def _retry_compose(plan: Plan, cfg: BuildConfig, ctx: GeneratorContext, matrix: frozenset[str]) -> dict | None:
    for attempt in range(cfg.retry_budget):
        p = plan if attempt == 0 else replace(plan, key=f"{plan.key}:r{attempt}")
        record = compose(p, ctx)
        if record is None:
            continue
        errors = [i for i in validate_record(record, matrix) if i.severity == "error"]
        if not errors:
            return record
    return None


def _dedup_regenerate(pairs, cfg, ctx, matrix, gaps: list[str]) -> list[tuple[Plan, dict]]:
    for _round in range(3):
        records = [r for _, r in pairs]
        offenders = find_near_dups(records)
        if not offenders:
            break
        new_pairs = []
        for plan, record in pairs:
            if record["id"] not in offenders:
                new_pairs.append((plan, record))
                continue
            regen = _retry_compose(plan, cfg, ctx, matrix)
            if regen is not None:
                regen["id"] = f"gen_{len(new_pairs):06d}"  # temp id for dedup bookkeeping
                new_pairs.append((plan, regen))
            else:
                gaps.append(plan.key)
        pairs = new_pairs
    return pairs


def _stratified_split(records: list[dict], seed: int, key: str = "category"):
    rng = random.Random(seed + 1)
    by_cat: dict[str, list[dict]] = {}
    for r in records:
        by_cat.setdefault(r[key], []).append(r)
    train, val, test = [], [], []
    for recs in by_cat.values():
        order = rng.sample(recs, len(recs))
        n = len(order)
        n_test = max(min(25, max(1, round(0.05 * n))), round(0.05 * n))
        n_val = round(0.10 * n)
        test += order[:n_test]
        val += order[n_test:n_test + n_val]
        train += order[n_test + n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build(cfg: BuildConfig, out_dir: Path | str) -> dict:
    _validate_config(cfg)
    out_dir = Path(out_dir)
    matrix = frozenset(cfg.composition_matrix)
    ctx = GeneratorContext(seed=cfg.seed)

    plans = _allocate_plans(cfg)
    pairs: list[tuple[Plan, dict]] = []
    gaps: list[str] = []
    for plan in plans:
        record = _retry_compose(plan, cfg, ctx, matrix)
        if record is None:
            gaps.append(plan.key)
            continue
        record["id"] = f"gen_{len(pairs):06d}"
        pairs.append((plan, record))

    pairs = _dedup_regenerate(pairs, cfg, ctx, matrix, gaps)

    records = [r for _, r in pairs]
    for i, r in enumerate(records):
        r["id"] = f"{r['category'][:3]}_{i:06d}"

    # V9 coverage (hard gate for the real build)
    hard_coverage = [i for i in coverage_issues(records, cfg.coverage_floors) if i.severity == "error"]
    if hard_coverage:
        raise RuntimeError(f"vocab coverage floors not met ({len(hard_coverage)} failures):\n"
                           + "\n".join(i.detail for i in hard_coverage[:20]))

    train, val, test = _stratified_split(records, cfg.seed)

    splits_dir = out_dir / "splits" / cfg.dataset_version
    splits_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(splits_dir / "train.jsonl", train)
    write_jsonl(splits_dir / "validation.jsonl", val)
    write_jsonl(splits_dir / "test.jsonl", test)

    manifest = {
        "dataset_version": cfg.dataset_version,
        "generator_version": GENERATOR_VERSION,
        "seed": cfg.seed,
        "distribution": cfg.distribution,
        "composition_matrix": cfg.composition_matrix,
        "preservation_ratio": cfg.preservation_ratio,
        "identity_ratio": cfg.identity_ratio,
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "per_category": {cat: sum(1 for r in records if r["category"] == cat)
                         for cat in sorted({r["category"] for r in records})},
        "gaps": len(gaps),
    }
    write_json(splits_dir / "manifest.json", manifest)

    stats = dataset_stats(records, extra={
        "dataset_version": cfg.dataset_version,
        "split_sizes": manifest["split_sizes"],
        "gaps": len(gaps),
    })
    write_json(out_dir / "dataset_stats.json", stats)

    return {"total": len(records), "gaps": len(gaps),
            "split_sizes": manifest["split_sizes"],
            "per_category": manifest["per_category"]}


# ---------------------------------------------------------------------------
# V4 build path: hybrid dataset (real clean speech + template clean text),
# realized by the deterministic rule engine (no LLM, no API).
#
# Target distribution over the whole dataset (the five buckets the abandoned
# paid-API experiment planned):
#   20% no-op (input == output)
#   15% simple disfluencies (fillers, repetitions)
#   25% corrections / restarts / reformulations
#   30% compositional (2-3 transformations at once)
#   10% formatting (spoken command -> artifact)
# ---------------------------------------------------------------------------

# Compositional combinations offered by the rule engine, with allocation
# weights. Labels are the canonical combo ids used by the quality gates and
# the evaluation tier.
_V4_COMBOS = (
    ("filler+rep", ("filler", "repetition")),
    ("filler+corr", ("filler", "self_correction")),
    ("rep+corr", ("repetition", "self_correction")),
    ("filler+rep+corr", ("filler", "repetition", "self_correction")),
)
_V4_COMBO_WEIGHTS = (0.30, 0.30, 0.25, 0.15)
_V4_MATRIX = frozenset(label for label, _ in _V4_COMBOS)

# Difficulty mix per bucket (easy, medium, hard).
_V4_DIFFICULTY_WEIGHTS = {
    "noop": (0.55, 0.30, 0.15),
    "simple": (0.50, 0.35, 0.15),
    "correction": (0.30, 0.45, 0.25),
    "compositional": (0.20, 0.40, 0.40),
    "formatting": (0.60, 0.30, 0.10),  # label only; formatting uses no injectors
}

# bucket -> (category for the whole bucket, canonical combo label or None)
_V4_FORMATTING_TYPES = ("bullet_list", "heading", "polite_message")


@dataclass
class V4BuildConfig:
    """Configuration for the V4 hybrid dataset build."""
    total: int = 15000
    seed: int = 42
    # Five-bucket target distribution (must sum to 1.0).
    noop_ratio: float = 0.20
    simple_ratio: float = 0.15
    correction_ratio: float = 0.25
    compositional_ratio: float = 0.30
    formatting_ratio: float = 0.10
    # Real-first mixing: this share of the noop/simple/correction/
    # compositional plans draws its clean text from REAL dictated speech
    # (Nyra intended transcripts). A real sentence may be reused across up to
    # `real_max_uses` DIFFERENT recipes (different realizations, so no
    # duplicate pairs), but only once as an identity no-op. Formatting stays
    # synthetic; anything the pool cannot cover falls back to templates.
    real_ratio: float = 0.85
    real_max_uses: int = 3
    nyra_split: str = "train"
    nyra_cache_dir: str | None = None
    nyra_limit: int | None = None
    # Real-text sanity bounds (the V1-V10 gates cap non-identity outputs at
    # 60 words and identity outputs at 30 words).
    max_clean_words: int = 60
    noop_max_clean_words: int = 30
    retry_budget: int = 8
    dedup_rounds: int = 3
    # Anti-association floors. Real-first builds keep few template outputs,
    # so a kept floor of 1 is enough to guarantee that no pool value exists
    # ONLY as an abandoned correction value (abandoned floor stays 2).
    coverage_floors: dict | None = field(
        default_factory=lambda: {"kept": 1, "abandoned": 2}
    )
    dataset_version: str = "20260904.1"

    def validate(self) -> None:
        ratios = [
            self.noop_ratio, self.simple_ratio, self.correction_ratio,
            self.compositional_ratio, self.formatting_ratio,
        ]
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError(f"bucket ratios sum to {sum(ratios)}, expected 1.0")
        if not (0.0 <= self.real_ratio <= 1.0):
            raise ValueError("real_ratio must be in [0, 1]")
        if self.real_max_uses < 1:
            raise ValueError("real_max_uses must be >= 1")
        if self.total < 1 or self.seed < 0:
            raise ValueError("total and seed must be >= 0")
        parse_dataset_version(self.dataset_version)


def _v4_bucket_counts(cfg: V4BuildConfig) -> dict[str, int]:
    """Integer bucket quotas; rounding error is absorbed by the no-op bucket."""
    raw = {
        "noop": cfg.noop_ratio, "simple": cfg.simple_ratio,
        "correction": cfg.correction_ratio,
        "compositional": cfg.compositional_ratio, "formatting": cfg.formatting_ratio,
    }
    counts = {bucket: round(cfg.total * ratio) for bucket, ratio in raw.items()}
    counts["noop"] += cfg.total - sum(counts.values())
    return counts


@dataclass
class _V4Plan:
    """One row of the V4 quota table."""
    key: str
    bucket: str          # noop | simple | correction | compositional | formatting
    kind: str            # human label for stats (e.g. "filler", "filler+corr")
    transforms: tuple[str, ...]
    category: str        # canonical category label (8-set)
    difficulty: str      # easy | medium | hard
    combo: str | None = None
    source: str = "template"  # real | template (decided at allocation time)


def _v4_allocate_plans(cfg: V4BuildConfig) -> list[_V4Plan]:
    """Deterministic quota allocation across the five buckets.

    Plan order: noop -> simple -> correction -> compositional -> formatting.
    Difficulty draws come from one seeded RNG; within a bucket, sub-kinds are
    drawn from the same RNG so the mix is reproducible.
    """
    counts = _v4_bucket_counts(cfg)
    real_quota = {b: int(n * cfg.real_ratio) for b, n in counts.items()}
    diff_rng = random.Random(cfg.seed)
    plans: list[_V4Plan] = []

    def draw_difficulty(bucket: str) -> str:
        return diff_rng.choices(
            ("easy", "medium", "hard"), weights=_V4_DIFFICULTY_WEIGHTS[bucket]
        )[0]

    counter = 0

    def plan(bucket: str, kind: str, transforms: tuple[str, ...], category: str,
             *, combo: str | None = None) -> _V4Plan:
        nonlocal counter
        p = _V4Plan(
            key=f"v4:{cfg.seed}:{counter:06d}",
            bucket=bucket, kind=kind, transforms=transforms, category=category,
            difficulty=draw_difficulty(bucket), combo=combo,
        )
        counter += 1
        return p

    for _ in range(counts["noop"]):
        p = plan("noop", "noop", (), "basic")
        p.source = "real" if real_quota["noop"] > 0 else "template"
        real_quota["noop"] -= 1 if p.source == "real" else 0
        plans.append(p)

    simple_kinds = (("filler", ("filler",), "filler"),
                    ("repetition", ("repetition",), "repetition"))
    for i in range(counts["simple"]):
        kind, transforms, category = simple_kinds[i % 2]
        p = plan("simple", kind, transforms, category)
        p.source = "real" if real_quota["simple"] > 0 else "template"
        real_quota["simple"] -= 1 if p.source == "real" else 0
        plans.append(p)

    corr_kinds = (("self_correction", ("self_correction",), "correction"),
                  ("reformulation", ("reformulation",), "reformulation"))
    for i in range(counts["correction"]):
        kind, transforms, category = corr_kinds[i % 2]
        p = plan("correction", kind, transforms, category)
        p.source = "real" if real_quota["correction"] > 0 else "template"
        real_quota["correction"] -= 1 if p.source == "real" else 0
        plans.append(p)

    labels = [label for label, _ in _V4_COMBOS]
    for _ in range(counts["compositional"]):
        label = diff_rng.choices(labels, weights=_V4_COMBO_WEIGHTS)[0]
        transforms = dict(_V4_COMBOS)[label]
        p = plan("compositional", label, transforms, "mixed", combo=label)
        p.source = "real" if real_quota["compositional"] > 0 else "template"
        real_quota["compositional"] -= 1 if p.source == "real" else 0
        plans.append(p)

    for i in range(counts["formatting"]):
        p = plan("formatting", "formatting", ("formatting",), "formatting")
        p.difficulty = "easy"
        plans.append(p)

    return plans


def _v4_real_pool(cfg: V4BuildConfig) -> list[dict]:
    """Load real clean texts (Nyra intended transcripts) into a sorted pool.

    Entries: {text, words, kinds, source_id}. Deterministic order keeps builds
    reproducible; the pool is empty (build continues template-only) when the
    dataset cannot be loaded offline.
    """
    entries: list[dict] = []
    try:
        from data.augmentation.realizer import entity_kinds
        from data.sources.nyra import load_nyra

        records = load_nyra(
            split=cfg.nyra_split, limit=cfg.nyra_limit, cache_dir=cfg.nyra_cache_dir
        )
    except Exception as exc:  # pragma: no cover - depends on the environment
        print(f"[v4] real clean-text source unavailable ({exc}); templates only")
        return entries
    seen: set[str] = set()
    for rec in records:
        text = (rec.get("output") or "").strip()
        if not text or "\n" in text or not text[0].isupper():
            continue
        words = len(text.split())
        if words < 3 or words > cfg.max_clean_words:
            continue
        if text in seen:
            continue
        seen.add(text)
        entries.append({
            "text": text,
            "words": words,
            "kinds": entity_kinds(text),
            "source_id": rec.get("source_id", ""),
        })
    entries.sort(key=lambda e: (e["source_id"], e["text"]))
    return entries


class _RealPool:
    """Deterministic round-robin over the real clean-text pool.

    REAL-FIRST semantics: a real sentence may back up to `max_uses` different
    recipes (each plan realizes it with its own seed, so the (input, output)
    pairs differ) — but only ONCE as an identity no-op, so identity records
    never repeat a sentence. Eligibility (word cap, entity kinds) and the
    per-entry counters keep assignment deterministic.
    """

    def __init__(self, entries: list[dict], max_uses: int):
        self.entries = entries
        self.max_uses = max_uses
        self.counts = [0] * len(entries)
        self.identity_used: set[int] = set()
        self.pointer = 0

    def next(self, *, max_words: int, min_kinds: int = 0, identity: bool = False) -> dict | None:
        n = len(self.entries)
        if n == 0:
            return None
        for _ in range(n):
            idx = self.pointer
            self.pointer = (self.pointer + 1) % n
            entry = self.entries[idx]
            if entry["words"] > max_words or len(entry["kinds"]) < min_kinds:
                continue
            if identity:
                if idx in self.identity_used:
                    continue
                self.identity_used.add(idx)
            else:
                if self.counts[idx] >= self.max_uses:
                    continue
                self.counts[idx] += 1
            return entry
        return None


def _v4_hard_errors(record: dict) -> list:
    return [i for i in validate_record(record, _V4_MATRIX) if i.severity == "error"]


def _v4_compose_template(plan: _V4Plan, cfg: V4BuildConfig, attempt: int) -> dict | None:
    """Template-sourced record via the legacy composer (template clean text +
    the same deterministic noise injectors)."""
    key = f"{plan.key}:r{attempt}" if attempt else plan.key
    rng = md5_rng(key)
    p = Plan(
        key=key,
        category=plan.category,
        transforms=plan.transforms,
        domain=rng.choice(DOMAINS),
        difficulty=plan.difficulty,
        identity=plan.bucket == "noop",
        formatting="random" if plan.bucket == "formatting" else None,
        corrections=1 if "self_correction" in plan.transforms else 0,
        combo=plan.combo,
    )
    record = compose(p, GeneratorContext(seed=cfg.seed))
    if record is None:
        return None
    record["bucket"] = plan.bucket
    record["source"] = "template"
    record["generator_version"] = V4_GENERATOR_VERSION
    return record


def _v4_stress_pairs() -> set[tuple[str, str]]:
    """(input, output) pairs of the hand-written hold-out stress sets."""
    from data.stress_test import build_stress_records
    from data.stress_test_v4 import build_stress_records_v4

    pairs = set()
    for record in build_stress_records() + build_stress_records_v4():
        pairs.add((record["input"], record["output"]))
    return pairs


def _v4_generate(plan: _V4Plan, cfg: V4BuildConfig, pool: _RealPool) -> dict | None:
    """Build one record for a plan; returns None when every attempt gaps."""
    from data.augmentation.realizer import realize, to_record

    if plan.source == "real":
        max_words = cfg.noop_max_clean_words if plan.bucket == "noop" else cfg.max_clean_words
        min_kinds = 1 if "self_correction" in plan.transforms else 0
        for attempt in range(cfg.retry_budget):
            entry = pool.next(
                max_words=max_words, min_kinds=min_kinds,
                identity=plan.bucket == "noop",
            )
            if entry is None:
                break
            realized = realize(
                entry["text"],
                transforms=plan.transforms,
                difficulty=plan.difficulty,
                seed_key=f"{plan.key}:r{attempt}" if attempt else plan.key,
            )
            if realized is None:
                continue
            record = to_record(realized, plan={
                "category": plan.category,
                "transforms": plan.transforms,
                "difficulty": plan.difficulty,
                "domain": "everyday",
                "combo": plan.combo,
                "bucket": plan.bucket,
                "source": "real",
            })
            if not _v4_hard_errors(record):
                record["template_id"] = None
                return record

    # Template fallback also covers real plans whose pool ran dry or whose
    # text could not realize the recipe.
    for attempt in range(cfg.retry_budget):
        record = _v4_compose_template(plan, cfg, attempt)
        if record is None:
            continue
        if not _v4_hard_errors(record):
            return record
    return None


def build_v4(cfg: V4BuildConfig, out_dir: Path | str | None = None) -> dict:
    """Build the V4 hybrid dataset and write train/validation/test splits.

    Fully deterministic for a fixed config + real pool: quota allocation,
    real-pool cursor, realization seeds, gates, dedup, and the stratified
    split are all seeded, so the same config reproduces byte-identical JSONL.

    Returns a summary dict (total, gaps, per-bucket counts, split sizes).
    """
    cfg.validate()
    out_dir = Path(out_dir or "data")
    out_dir.mkdir(parents=True, exist_ok=True)

    plans = _v4_allocate_plans(cfg)
    pool = _RealPool(_v4_real_pool(cfg), cfg.real_max_uses)

    # 1. Compose every plan (real clean text or template), gating each record.
    pairs: list[tuple[_V4Plan, dict]] = []
    gaps = 0
    for plan in plans:
        record = _v4_generate(plan, cfg, pool)
        if record is None:
            gaps += 1
            continue
        record["id"] = f"tmp_{len(pairs):06d}"
        pairs.append((plan, record))

    # 2. Near-duplicate regeneration (same group semantics as the V2 path).
    for _round in range(cfg.dedup_rounds):
        records = [r for _, r in pairs]
        offenders = find_near_dups(records)
        if not offenders:
            break
        new_pairs: list[tuple[_V4Plan, dict]] = []
        for plan, record in pairs:
            if record["id"] not in offenders:
                new_pairs.append((plan, record))
                continue
            regen = _v4_generate(plan, cfg, pool)
            if regen is None:
                gaps += 1
                continue
            regen["id"] = f"tmp_{len(new_pairs):06d}"
            new_pairs.append((plan, regen))
        pairs = new_pairs

    # 3. Hold-out guard: the hand-written stress records must never leak into
    # the training splits (spec §10-11, §23). Colliding records are
    # regenerated with a new plan key, or counted as gaps when they persist.
    stress_pairs = _v4_stress_pairs()
    guarded: list[tuple[_V4Plan, dict]] = []
    for plan, record in pairs:
        if (record["input"], record["output"]) not in stress_pairs:
            guarded.append((plan, record))
            continue
        for attempt in range(cfg.retry_budget):
            variant = replace(plan, key=f"{plan.key}:s{attempt}")
            regen = _v4_generate(variant, cfg, pool)
            if regen is not None and (regen["input"], regen["output"]) not in stress_pairs:
                regen["id"] = f"tmp_{len(guarded):06d}"
                guarded.append((variant, regen))
                break
        else:
            gaps += 1
    pairs = guarded

    # 4. Exact-duplicate elimination. Reusing a real sentence across recipes
    # can yield identical (input, output) pairs from different plans (e.g. a
    # one-filler easy realization on a text with a single gap). Regenerate
    # the later copy with a new key; drop it as a gap if it cannot be made
    # unique. Iteration order is fixed, so this pass is deterministic.
    seen_pairs: set[tuple[str, str]] = set()
    unique: list[tuple[_V4Plan, dict]] = []
    for plan, record in pairs:
        pair = (record["input"], record["output"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique.append((plan, record))
            continue
        for attempt in range(cfg.retry_budget):
            variant = replace(plan, key=f"{plan.key}:u{attempt}")
            regen = _v4_generate(variant, cfg, pool)
            if regen is None:
                continue
            regen_pair = (regen["input"], regen["output"])
            if regen_pair not in seen_pairs and regen_pair not in stress_pairs:
                seen_pairs.add(regen_pair)
                regen["id"] = f"tmp_{len(unique):06d}"
                unique.append((variant, regen))
                break
        else:
            gaps += 1
    pairs = unique

    # 5. Stable ordering + final ids (bucket-prefixed, deterministic).
    records = [r for _, r in pairs]
    records.sort(key=lambda r: (r["bucket"], r["id"]))
    for i, r in enumerate(records):
        r["id"] = f"v4_{i:06d}"

    # 6. V9 vocab-coverage hard gate (anti-association).
    if cfg.coverage_floors:
        hard_coverage = [
            i for i in coverage_issues(records, cfg.coverage_floors)
            if i.severity == "error"
        ]
        if hard_coverage:
            raise RuntimeError(
                f"v4 vocab coverage floors not met ({len(hard_coverage)} failures):\n"
                + "\n".join(i.detail for i in hard_coverage[:10])
            )

    # 7. Stratified split by bucket (85 / 10 / 5).
    train, val, test = _stratified_split(records, cfg.seed, key="bucket")

    # 5. V9 vocab-coverage hard gate (anti-association).
    if cfg.coverage_floors:
        hard_coverage = [
            i for i in coverage_issues(records, cfg.coverage_floors)
            if i.severity == "error"
        ]
        if hard_coverage:
            raise RuntimeError(
                f"v4 vocab coverage floors not met ({len(hard_coverage)} failures):\n"
                + "\n".join(i.detail for i in hard_coverage[:10])
            )

    # 6. Stratified split by bucket (85 / 10 / 5).
    train, val, test = _stratified_split(records, cfg.seed, key="bucket")

    splits_dir = out_dir / "splits" / cfg.dataset_version
    splits_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(splits_dir / "train.jsonl", train)
    write_jsonl(splits_dir / "validation.jsonl", val)
    write_jsonl(splits_dir / "test.jsonl", test)

    from collections import Counter

    per_bucket = dict(Counter(r["bucket"] for r in records))
    per_source = dict(Counter(r["source"] for r in records))
    targets = {
        "noop": cfg.noop_ratio, "simple": cfg.simple_ratio,
        "correction": cfg.correction_ratio,
        "compositional": cfg.compositional_ratio, "formatting": cfg.formatting_ratio,
    }

    manifest = {
        "dataset_version": cfg.dataset_version,
        "generator_version": V4_GENERATOR_VERSION,
        "seed": cfg.seed,
        "total": len(records),
        "real_ratio": cfg.real_ratio,
        "bucket_targets": targets,
        "per_bucket": per_bucket,
        "per_source": per_source,
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "per_category": dict(Counter(r["category"] for r in records)),
        "per_combo": dict(Counter(r["combo"] for r in records if r.get("combo"))),
        "gaps": gaps,
    }
    write_json(splits_dir / "manifest.json", manifest)

    stats = dataset_stats(records, extra={
        "dataset_version": cfg.dataset_version,
        "generator_version": V4_GENERATOR_VERSION,
        "split_sizes": manifest["split_sizes"],
        "per_bucket": per_bucket,
        "per_source": per_source,
        "bucket_targets": targets,
        "gaps": gaps,
    })
    write_json(out_dir / "dataset_stats.json", stats)

    return {
        "total": len(records),
        "gaps": gaps,
        "per_bucket": per_bucket,
        "per_source": per_source,
        "split_sizes": manifest["split_sizes"],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m data.dataset_builder --v4 [--total N] [--seed S]
    [--out DIR] [--real-ratio R] [--nyra-limit N] [--dataset-version V]"""
    import argparse

    parser = argparse.ArgumentParser(prog="data.dataset_builder",
                                     description="Rambler dataset builder (V2 legacy, V4 hybrid)")
    parser.add_argument("--v4", action="store_true",
                        help="build the V4 hybrid dataset (default out: data/)")
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="data")
    parser.add_argument("--real-ratio", type=float, default=None,
                        help="share of clean text from real speech (0-1)")
    parser.add_argument("--real-max-uses", type=int, default=None,
                        help="max recipes per real sentence (default 3)")
    parser.add_argument("--nyra-limit", type=int, default=None)
    parser.add_argument("--dataset-version", default=None)
    args = parser.parse_args(argv)

    if not args.v4:
        parser.print_help()
        return 0

    cfg = V4BuildConfig(
        total=args.total if args.total is not None else V4BuildConfig.total,
        seed=args.seed if args.seed is not None else V4BuildConfig.seed,
        real_ratio=args.real_ratio if args.real_ratio is not None else V4BuildConfig.real_ratio,
        real_max_uses=(args.real_max_uses if args.real_max_uses is not None
                       else V4BuildConfig.real_max_uses),
        nyra_limit=args.nyra_limit,
        dataset_version=(args.dataset_version or V4BuildConfig.dataset_version),
    )
    summary = build_v4(cfg, args.out)
    print(f"V4 dataset built: {summary['total']} records, {summary['gaps']} gaps")
    print(f"  per bucket: {summary['per_bucket']}")
    print(f"  per source: {summary['per_source']}")
    print(f"  splits: {summary['split_sizes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


