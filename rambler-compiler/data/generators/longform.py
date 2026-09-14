"""Realistic 50-300-word spoken monologues built from same-domain units.

Seam prevention: units are framed procedurally (opener frames, capitalized
connectives, a closer sentence), so chained templates read as one continuous
monologue rather than concatenated sentences.

Coherence: every extension clause carries the set of domains it is licensed
for — technical clauses ("in case the {tech_s} goes down") only attach to
technical/work/programming units, so monologues never produce non-sequitur
entity combinations. Extension predicates are plural-safe: no clause forces
singular agreement on a {item} that could be plural.
"""

import random
from dataclasses import dataclass

from data.generators.entities import PoolSampler, make_date, make_number, make_time
from data.generators.templates import (
    DOMAINS,
    _SLOT_RE,
    fill_slots,
    sample_kind,
    templates_for,
)

_OPENERS = (
    "Okay so here's the situation:",
    "So basically,",
    "Here's the thing:",
    "Let me explain what's going on:",
    "Alright so,",
    "Okay here's what I need:",
)
_CLOSERS = (
    "And that should cover everything for now.",
    "That's everything on my list.",
    "So that's the plan.",
    "That should be all for today.",
    "And that's about it.",
    "I think that covers it all.",
)
_CONNECTIVES = (
    "and then", "also", "oh and", "by the way", "on top of that", "plus", "another thing",
    "one more thing", "after that", "first of all", "then", "finally", "and honestly",
    "so basically", "here's the deal", "the thing is", "and one more thing", "in any case",
    "apart from that", "and yeah", "but the main thing is", "what else", "oh right",
    "also one more thing", "and on that note", "anyway", "moving on", "so anyway",
    "let me also add", "oh and also",
)

# Domain license groups for extension clauses.
_ALL_DOMAINS = frozenset(DOMAINS)
_WORKISH = frozenset({"work", "scheduling", "university", "programming", "technical",
                      "notes", "email"})
_TECHISH = frozenset({"work", "technical", "programming"})
_OUTDOORS = frozenset({"travel", "everyday", "shopping"})

# (pattern, allowed_domains) — clauses attach only to units whose domain is
# licensed, and every predicate is plural-safe with respect to {item}.
_EXTENSIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("because {name} asked for it", _ALL_DOMAINS),
    ("since we missed the {date} deadline", _WORKISH),
    ("before the {time} call", frozenset({"work", "scheduling", "messaging", "email"})),
    ("after we hear back from {name}", _ALL_DOMAINS),
    ("even though {location} is far", _OUTDOORS),
    ("so that we can finish by {date}", _ALL_DOMAINS),
    ("which should take about {duration}", _ALL_DOMAINS),
    ("unless {name} says otherwise", _ALL_DOMAINS),
    ("in case the {tech_s} goes down", _TECHISH),
    ("so {name} has time to review", frozenset({"work", "programming", "university", "email"})),
    ("if the weather in {location} stays clear", frozenset({"travel", "everyday"})),
    ("since the {item} arrived late", frozenset({"shopping", "work"})),
    ("which reminds me about the {item}", _ALL_DOMAINS),
    ("before {location} gets busy", _OUTDOORS),
    ("so we don't miss the {time} deadline", frozenset({"work", "scheduling"})),
    ("because the {tech_s} keeps failing", _TECHISH),
    ("if {name} is back by then", _ALL_DOMAINS),
    ("which is fine with {name}", _ALL_DOMAINS),
    ("as long as {date} still works", frozenset({"scheduling", "work"})),
    ("so we have the {item} ready", _ALL_DOMAINS),
)

_BUCKETS = {"easy": (55, 80), "medium": (90, 145), "hard": (155, 290)}
_UNIT_COUNTS = {"easy": 2, "medium": 3, "hard": 4}
_EXT_COUNTS = {"easy": (1, 2), "medium": (2, 3), "hard": (3, 4)}

# Connectives that read as conjuncts (no comma): "Oh and add paint..."
# vs adverbial phrases ("By the way, add paint...").
NO_COMMA_CONNECTIVES = {
    "and then", "also", "oh and", "plus", "another thing", "one more thing",
    "and one more thing", "also one more thing", "let me also add", "oh and also",
    "and yeah", "then",
}


@dataclass
class LongformResult:
    text: str
    units: list[dict]  # {"text", "offsets": (start, end), "connective": str | None}


def _lower_first(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


def _pick_extension(domain: str, rng: random.Random) -> str:
    licensed = [p for p, domains in _EXTENSIONS if domain in domains]
    if not licensed:  # never happens (all-domains clauses exist), but be safe
        licensed = [p for p, domains in _EXTENSIONS if domains == _ALL_DOMAINS]
    return rng.choice(licensed)


def _fill_extension(pattern: str, sampler: PoolSampler, rng: random.Random,
                    cache: dict[str, str], used: dict[str, set[str]]) -> str:
    return _SLOT_RE.sub(
        lambda m: sample_kind(m.group(1), sampler, rng, cache, used), pattern
    )


def _build_unit(domain: str, rng: random.Random, difficulty: str) -> str:
    template = rng.choice(templates_for(domain))
    sampler = PoolSampler(rng)
    cache: dict[str, str] = {}
    used: dict[str, set[str]] = {
        k: set() for k in ("number", "price", "percent", "version",
                           "duration", "port", "id", "time", "date")
    }
    text, _ = fill_slots(template, sampler, rng)
    # Extensions attach before the sentence terminator, never after it.
    terminator = ""
    if text.endswith((".", "?", "!")):
        terminator = text[-1]
        text = text[:-1]
    lo, hi = _EXT_COUNTS[difficulty]
    for _ in range(rng.randint(lo, hi)):
        extension = _pick_extension(domain, rng)
        text += ", " + _fill_extension(extension, sampler, rng, cache, used)
    return text + terminator


def _assemble(parts: list[tuple[str, str | None]]) -> tuple[str, list[dict]]:
    units = []
    offset = 0
    for text, connective in parts:
        units.append({"text": text, "offsets": (offset, offset + len(text)), "connective": connective})
        offset += len(text) + 1
    return " ".join(p[0] for p in parts), units


def compose_longform(domain: str, rng: random.Random, difficulty: str = "medium") -> LongformResult:
    lo, hi = _BUCKETS[difficulty]
    n_units = _UNIT_COUNTS[difficulty]
    used_connectives: set[str] = set()

    def pick_connective() -> str:
        available = [c for c in _CONNECTIVES if c not in used_connectives]
        if not available:
            available = list(_CONNECTIVES)
        conn = rng.choice(available)
        used_connectives.add(conn)
        return conn

    parts: list[tuple[str, str | None]] = []
    opener = rng.choice(_OPENERS)
    parts.append((f"{opener} {_lower_first(_build_unit(domain, rng, difficulty))}", None))
    for _ in range(n_units - 1):
        conn = pick_connective()
        cap = conn.capitalize()
        joiner = "" if conn in NO_COMMA_CONNECTIVES else ","
        parts.append((f"{cap}{joiner} {_lower_first(_build_unit(domain, rng, difficulty))}", conn))
    parts.append((rng.choice(_CLOSERS), None))

    # length adjustment: too short -> add a unit; too long -> drop a middle unit
    for _ in range(8):
        text, _ = _assemble(parts)
        if len(text.split()) >= lo:
            break
        conn = pick_connective()
        cap = conn.capitalize()
        joiner = "" if conn in NO_COMMA_CONNECTIVES else ","
        parts.insert(-1, (f"{cap}{joiner} {_lower_first(_build_unit(domain, rng, difficulty))}", conn))
    for _ in range(4):
        text, _ = _assemble(parts)
        if len(text.split()) <= hi or len(parts) <= 4:
            break
        parts.pop(-2)

    text, units = _assemble(parts)
    return LongformResult(text=text, units=units)
