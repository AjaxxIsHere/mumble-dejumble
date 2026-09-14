"""Nyra disfluency speech dataset loader.

Loads `nyralabs/disfluency_speech_english` from HuggingFace, normalizes
verbatim transcripts to resemble Whisper output (strips bracket tags like
[UH], [UM], [laughter] that Whisper does not produce), and maps examples
into Rambler's canonical schema.

The intended_transcript is NEVER modified — it is the ground-truth target.

V4 uses this as the REAL clean-text source: intended transcripts become the
clean base that the rule-based realizer turns into messy speech, so the
semantic target is always a sentence a person actually dictated.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from data.common import write_jsonl

# ---------------------------------------------------------------------------
# Bracket-tag stripping (Whisper compatibility)
# ---------------------------------------------------------------------------

# Matches [UH], [UM], [laughter], [breath], [cough], [noise], etc.
_BRACKET_TAG_RE = re.compile(r"\[[A-Za-z_]+\]")

# Cutoff markers: "w*", "bru*", "d*" — Whisper produces partial words,
# but the asterisk notation is annotation-specific. Strip the asterisk
# so the partial word remains (e.g. "w*" → "w").
_CUTOFF_RE = re.compile(r"(\w)\*")


def normalize_verbatim(text: str) -> str:
    """Convert a Nyra verbatim transcript to Whisper-like format.

    - Strip bracket tags: [UH], [UM], [laughter], [breath], etc.
    - Strip cutoff asterisks: "w*" → "w"
    - Collapse multiple spaces
    - Strip leading/trailing whitespace
    """
    # Remove bracket tags
    text = _BRACKET_TAG_RE.sub("", text)
    # Strip cutoff asterisks (keep the partial word)
    text = _CUTOFF_RE.sub(r"\1", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


# ---------------------------------------------------------------------------
# Canonical record schema
# ---------------------------------------------------------------------------

@dataclass
class NyraRecord:
    """A single record mapped from the Nyra dataset."""
    id: str
    input: str          # normalized verbatim (Whisper-like)
    output: str         # intended_transcript (ground truth, untouched)
    source: str = "nyra"
    source_id: str = ""
    category: str = "nyra"
    transformations: list[str] = field(default_factory=list)
    num_transformations: int = 0
    difficulty: str = "medium"
    domain: str = "everyday"
    entities: dict = field(default_factory=dict)
    corrections: list = field(default_factory=list)
    reformulations: list = field(default_factory=list)
    formatting: dict | None = None
    preservation_case: bool = False
    preserved: list = field(default_factory=list)
    combo: str | None = None
    template_id: str | None = None
    seed: int = 0
    generator_version: str = "v4.0.0"
    source_dataset_version: str = ""
    # Nyra-specific metadata
    duration_in_s: float = 0.0
    speaker: str = ""
    original_verbatim: str = ""  # before normalization (for debugging)


def to_canonical(record: dict, *, source_dataset_version: str = "") -> dict:
    """Map a raw Nyra record to Rambler's canonical schema.

    The intended_transcript is preserved exactly as-is.
    The verbatim_transcript is normalized to resemble Whisper output.
    """
    raw_verbatim = record.get("verbatim_transcript", "")
    intended = record.get("intended_transcript", "")
    normalized = normalize_verbatim(raw_verbatim)

    return {
        "id": f"nyra_{record.get('id', 'unknown')}",
        "input": normalized,
        "output": intended,
        "source": "nyra",
        "source_id": record.get("id", ""),
        "category": "nyra",
        "transformations": [],
        "num_transformations": 0,
        "difficulty": "medium",
        "domain": "everyday",
        "entities": {},
        "corrections": [],
        "reformulations": [],
        "formatting": None,
        "preservation_case": False,
        "preserved": [],
        "combo": None,
        "template_id": None,
        "seed": 0,
        "generator_version": "v4.0.0",
        "source_dataset_version": source_dataset_version,
        "duration_in_s": record.get("duration_in_s", 0.0),
        "speaker": record.get("speaker", "unknown"),
        "original_verbatim": raw_verbatim,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_nyra(
    split: str = "train",
    limit: int | None = None,
    cache_dir: str | None = None,
) -> list[dict]:
    """Load the Nyra disfluency dataset from HuggingFace.

    Args:
        split: Dataset split to load (default: "train").
        limit: Maximum number of records to load (None = all).
        cache_dir: HuggingFace cache directory (None = default).

    Returns:
        List of canonical records.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "nyralabs/disfluency_speech_english",
        split=split,
        cache_dir=cache_dir,
    )
    # Keep only the text/metadata columns (drop any audio columns) so row
    # access stays cheap and deterministic.
    keep = {"id", "verbatim_transcript", "intended_transcript", "duration_in_s", "speaker"}
    extra = [c for c in ds.column_names if c not in keep]
    if extra:
        ds = ds.remove_columns(extra)

    records = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        raw = {
            "id": row.get("id", f"nyra_{i:06d}"),
            "verbatim_transcript": row.get("verbatim_transcript", ""),
            "intended_transcript": row.get("intended_transcript", ""),
            "duration_in_s": row.get("duration_in_s", 0.0),
            "speaker": row.get("speaker", "unknown"),
        }
        records.append(to_canonical(raw))

    return records


def load_nyra_from_cache(path: str | Path) -> list[dict]:
    """Load already-cached Nyra records from a JSONL file."""
    from data.common import load_jsonl

    return load_jsonl(path)


def save_nyra_cache(records: list[dict], path: str | Path) -> None:
    """Save Nyra records to a JSONL cache file."""
    write_jsonl(path, records)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def nyra_stats(records: list[dict]) -> dict:
    """Compute statistics over Nyra records."""
    from data.common import normalize

    inputs = [r["input"] for r in records]
    outputs = [r["output"] for r in records]

    def wc(text):
        return len(text.split())

    input_words = [wc(t) for t in inputs]
    output_words = [wc(t) for t in outputs]

    # Detect what disfluency types are present in the original verbatim
    has_bracket_tags = sum(
        1 for r in records if "[" in r.get("original_verbatim", "")
    )
    has_cutoffs = sum(
        1 for r in records if "*" in r.get("original_verbatim", "")
    )
    # No-op: input == output after normalization
    noop_count = sum(1 for i, o in zip(inputs, outputs) if normalize(i) == normalize(o))

    return {
        "total": len(records),
        "noop_count": noop_count,
        "noop_ratio": noop_count / max(1, len(records)),
        "has_bracket_tags": has_bracket_tags,
        "has_cutoffs": has_cutoffs,
        "input_word_percentiles": _percentiles(input_words),
        "output_word_percentiles": _percentiles(output_words),
        "avg_input_words": sum(input_words) / max(1, len(input_words)),
        "avg_output_words": sum(output_words) / max(1, len(output_words)),
    }


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {"p25": 0, "p50": 0, "p75": 0, "p90": 0, "max": 0}
    s = sorted(values)
    def pct(p):
        return s[min(len(s) - 1, int(len(s) * p))]
    return {
        "p25": pct(0.25), "p50": pct(0.50), "p75": pct(0.75),
        "p90": pct(0.90), "max": s[-1],
    }
