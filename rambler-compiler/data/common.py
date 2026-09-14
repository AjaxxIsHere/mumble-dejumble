"""Shared stdlib helpers for the data and evaluation tiers.

This module must NEVER import torch or transformers — it runs on the CPU-only
development machine as well as inside Colab.
"""

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

# The production instruction. Single source of truth: this exact text is used at
# training time AND must be used by the Flutter app at inference time.
INSTRUCTION = (
    "You are a speech-to-text compiler. Rewrite the spoken transcript into the "
    "text the user intended to type. Remove disfluencies, stutters, repetitions, "
    "abandoned phrases, and incorrect earlier wording. Resolve self-corrections "
    "using the user's final intended wording. Preserve meaningful information, "
    "names, numbers, dates, times, and technical terms. Apply requested "
    "formatting. Output only the final text."
)

# The eight spec categories (spec §4/§13).
CATEGORIES = {
    "basic",
    "filler",
    "repetition",
    "correction",
    "reformulation",
    "formatting",
    "long",
    "mixed",
}

# Canonical transformation names used in record metadata (spec §5).
TRANSFORMS = {"filler", "repetition", "self_correction", "reformulation", "formatting"}

DIFFICULTIES = ("easy", "medium", "hard")

# V4 constants
V4_GENERATOR_VERSION = "v4.0.0"
V4_RECIPE_VERSION = "v1.0"

# Transformation types the V4 rule-based realizer can realize with the
# deterministic noise injectors (see data/generators/injectors.py).
V4_TRANSFORMS = {"filler", "repetition", "self_correction", "reformulation", "formatting"}

_DATASET_VERSION_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})\.(\d+)$")


def normalize(text: str) -> str:
    """NFKC -> strip surrounding double quotes -> lower -> collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).strip()
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return " ".join(text.lower().split())


def md5_rng(key: str) -> random.Random:
    """Deterministic RNG from a stable string key.

    Uses md5 (not hash()): Python's hash() is salted per process, md5 is not,
    so the same key yields the same draws across processes and Python runs.
    """
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_dataset_version(version: str) -> str:
    """Validate a dataset version like '20260831.1' and return it unchanged."""
    m = _DATASET_VERSION_RE.match(version)
    if not m:
        raise ValueError(f"invalid dataset version {version!r}; expected YYYYMMDD.N")
    month, day = int(m.group(2)), int(m.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise ValueError(f"invalid dataset version {version!r}; bad month/day")
    return version
