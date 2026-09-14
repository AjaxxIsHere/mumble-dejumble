"""Transformation recipe system.

Typed, serializable recipes that describe exactly which speech phenomena
to introduce into a clean transcript. The recipe is EXPLICIT — the AI
decides HOW to realize the phenomenon, not WHAT the clean target should be.

Example:
    {
        "transformations": [
            {"type": "filler", "location": "sentence_start"},
            {"type": "self_correction", "field": "date",
             "abandoned": "Tuesday", "retained": "Thursday",
             "marker": "no wait"}
        ]
    }
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Transformation types
# ---------------------------------------------------------------------------

class TransformationType(str, Enum):
    FILLER = "filler"
    REPETITION = "repetition"
    PARTIAL_REPETITION = "partial_repetition"
    FALSE_START = "false_start"
    SELF_CORRECTION = "self_correction"
    PHRASE_CORRECTION = "phrase_correction"
    ENTITY_CORRECTION = "entity_correction"
    RESTART = "restart"
    CLAUSE_RESTART = "clause_restart"
    HESITATION = "hesitation"
    FORMATTING_COMMAND = "formatting_command"


# Valid compositions (transformations that can co-occur in one recipe)
# Not all combinations make linguistic sense.
_VALID_COMPOSITIONS: dict[frozenset[str], str] = {
    frozenset(["filler"]): "single",
    frozenset(["repetition"]): "single",
    frozenset(["partial_repetition"]): "single",
    frozenset(["false_start"]): "single",
    frozenset(["self_correction"]): "single",
    frozenset(["phrase_correction"]): "single",
    frozenset(["entity_correction"]): "single",
    frozenset(["restart"]): "single",
    frozenset(["clause_restart"]): "single",
    frozenset(["hesitation"]): "single",
    frozenset(["formatting_command"]): "single",
    frozenset(["filler", "repetition"]): "filler+rep",
    frozenset(["filler", "self_correction"]): "filler+corr",
    frozenset(["repetition", "self_correction"]): "rep+corr",
    frozenset(["filler", "repetition", "self_correction"]): "filler+rep+corr",
    frozenset(["filler", "repetition", "self_correction", "formatting_command"]): "filler+rep+corr+fmt",
    frozenset(["repetition", "self_correction", "formatting_command"]): "rep+corr+fmt",
    frozenset(["filler", "self_correction", "formatting_command"]): "filler+corr+fmt",
    frozenset(["self_correction", "formatting_command"]): "corr+fmt",
    frozenset(["filler", "repetition", "formatting_command"]): "filler+rep+fmt",
    frozenset(["self_correction", "self_correction"]): "multi-corr",  # via count=2
    frozenset(["restart", "self_correction"]): "restart+corr",
    frozenset(["clause_restart", "self_correction"]): "clause_restart+corr",
}


# ---------------------------------------------------------------------------
# Recipe data classes
# ---------------------------------------------------------------------------

@dataclass
class Transformation:
    """A single transformation to apply."""
    type: str  # TransformationType value
    # For corrections:
    abandoned: str | None = None  # the wrong value to introduce
    retained: str | None = None   # the correct value to keep
    field: str | None = None      # entity kind (name, date, time, number, etc.)
    marker: str | None = None     # correction marker (no wait, actually, etc.)
    # For fillers:
    location: str | None = None   # sentence_start, mid_sentence, etc.
    # For repetitions:
    target: str | None = None     # the word/phrase to repeat
    kind: str | None = None       # word, phrase, partial_word, clause
    # For formatting:
    format_type: str | None = None  # bullet_list, heading, polite_message
    # For false starts / restarts:
    abandoned_phrase: str | None = None  # the phrase to abandon
    # Generic
    count: int = 1


@dataclass
class Recipe:
    """A complete transformation recipe for one example."""
    recipe_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    transformations: list[Transformation] = field(default_factory=list)
    composition_label: str = ""  # auto-computed from transformations
    version: str = "v1.0"
    notes: str | None = None

    def __post_init__(self):
        self.composition_label = self._compute_composition()

    def _compute_composition(self) -> str:
        if not self.transformations:
            return "noop"
        types = frozenset(t.type for t in self.transformations)
        # Handle multi-correction (two self_correction entries)
        if types == frozenset(["self_correction"]) and len(self.transformations) == 2:
            return "multi-corr"
        return _VALID_COMPOSITIONS.get(types, "unknown:" + "+".join(sorted(types)))

    def to_dict(self) -> dict:
        return {
            "recipe_id": self.recipe_id,
            "transformations": [asdict(t) for t in self.transformations],
            "composition_label": self.composition_label,
            "version": self.version,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Recipe:
        transformations = [Transformation(**t) for t in d.get("transformations", [])]
        return cls(
            recipe_id=d.get("recipe_id", str(uuid.uuid4())[:12]),
            transformations=transformations,
            version=d.get("version", "v1.0"),
            notes=d.get("notes"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> Recipe:
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Recipe generation helpers
# ---------------------------------------------------------------------------

def make_noop_recipe() -> Recipe:
    """No transformations — input is already clean."""
    return Recipe(transformations=[], composition_label="noop")


def make_filler_recipe(
    location: str = "random",
    rng: random.Random | None = None,
) -> Recipe:
    rng = rng or random.Random()
    locations = ["sentence_start", "mid_sentence", "end_of_clause"]
    loc = location if location != "random" else rng.choice(locations)
    return Recipe(
        transformations=[Transformation(type="filler", location=loc)],
    )


def make_repetition_recipe(
    kind: str = "random",
    target: str | None = None,
    rng: random.Random | None = None,
) -> Recipe:
    rng = rng or random.Random()
    kinds = ["word", "phrase", "partial_word"]
    k = kind if kind != "random" else rng.choice(kinds)
    return Recipe(
        transformations=[Transformation(type="repetition", kind=k, target=target)],
    )


def make_correction_recipe(
    abandoned: str,
    retained: str,
    field: str = "date",
    marker: str = "no wait",
) -> Recipe:
    return Recipe(
        transformations=[Transformation(
            type="self_correction",
            abandoned=abandoned,
            retained=retained,
            field=field,
            marker=marker,
        )],
    )


def make_compositional_recipe(
    types: list[str],
    *,
    corrections: list[tuple[str, str, str]] | None = None,
    rng: random.Random | None = None,
) -> Recipe:
    """Build a recipe with multiple transformation types.

    Args:
        types: List of transformation type strings (e.g. ["filler", "repetition", "self_correction"]).
        corrections: List of (abandoned, retained, field) tuples for self_correction entries.
        rng: Random generator for filling in details.
    """
    rng = rng or random.Random()
    transformations = []

    for t in types:
        if t == "filler":
            transformations.append(Transformation(
                type="filler",
                location=rng.choice(["sentence_start", "mid_sentence"]),
            ))
        elif t == "repetition":
            transformations.append(Transformation(
                type="repetition",
                kind=rng.choice(["word", "phrase"]),
            ))
        elif t == "self_correction":
            # Corrections need abandoned/retained — use provided or defaults
            if corrections:
                for ab, ret, fld in corrections:
                    transformations.append(Transformation(
                        type="self_correction",
                        abandoned=ab,
                        retained=ret,
                        field=fld,
                        marker=rng.choice(["no wait", "actually", "wait"]),
                    ))
            else:
                transformations.append(Transformation(
                    type="self_correction",
                    marker=rng.choice(["no wait", "actually", "wait"]),
                ))
        elif t == "formatting_command":
            transformations.append(Transformation(
                type="formatting_command",
                format_type=rng.choice(["bullet_list", "heading", "polite_message"]),
            ))
        elif t == "restart":
            transformations.append(Transformation(type="restart"))
        elif t == "clause_restart":
            transformations.append(Transformation(type="clause_restart"))
        elif t == "false_start":
            transformations.append(Transformation(type="false_start"))
        elif t == "hesitation":
            transformations.append(Transformation(type="hesitation"))
        elif t == "partial_repetition":
            transformations.append(Transformation(type="partial_repetition"))
        else:
            raise ValueError(f"Unknown transformation type: {t}")

    return Recipe(transformations=transformations)


# ---------------------------------------------------------------------------
# Composition validation
# ---------------------------------------------------------------------------

def validate_composition(recipe: Recipe) -> tuple[bool, str]:
    """Check if a recipe's transformation combination is linguistically valid.

    Returns (is_valid, reason).
    """
    if not recipe.transformations:
        return True, "noop"

    types = tuple(t.type for t in recipe.transformations)

    # Two self_corrections are allowed (multi-corr)
    type_set = frozenset(types)
    if type_set in _VALID_COMPOSITIONS:
        return True, _VALID_COMPOSITIONS[type_set]

    # Check for multi-correction pattern
    if types.count("self_correction") >= 2 and len(type_set) == 1:
        return True, "multi-corr"

    return False, f"unsupported combination: {sorted(type_set)}"


def recipe_summary(recipe: Recipe) -> str:
    """Human-readable summary of a recipe."""
    if not recipe.transformations:
        return "noop (clean input)"
    parts = []
    for t in recipe.transformations:
        if t.type == "filler":
            parts.append(f"filler({t.location or '?'})")
        elif t.type == "repetition":
            parts.append(f"repetition({t.kind or '?'})")
        elif t.type == "self_correction":
            parts.append(f"correction({t.abandoned} → {t.retained})")
        elif t.type == "formatting_command":
            parts.append(f"formatting({t.format_type or '?'})")
        else:
            parts.append(t.type)
    return " + ".join(parts)
