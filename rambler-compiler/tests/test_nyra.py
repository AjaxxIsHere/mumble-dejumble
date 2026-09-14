"""Tests for the Nyra dataset loader.

These tests verify the verbatim normalization and schema mapping
without requiring network access to HuggingFace.
"""

import pytest

from data.sources.nyra import normalize_verbatim, to_canonical, nyra_stats


class TestNormalizeVerbatim:
    """Tests for bracket-tag stripping and Whisper compatibility."""

    def test_strips_uh_tag(self):
        assert normalize_verbatim("[UH] well, I think so.") == "well, I think so."

    def test_strips_um_tag(self):
        assert normalize_verbatim("[UM] I want to go.") == "I want to go."

    def test_strips_multiple_tags(self):
        result = normalize_verbatim("[UH] well, [UM] I think [laughter] so.")
        assert "[UH]" not in result
        assert "[UM]" not in result
        assert "[laughter]" not in result
        assert "well," in result
        assert "I think" in result
        assert "so." in result

    def test_strips_cutoff_asterisks(self):
        result = normalize_verbatim("w* what did you say?")
        assert result == "w what did you say?"

    def test_strips_multiple_cutoffs(self):
        result = normalize_verbatim("I I rem* remember back.")
        assert "rem*" not in result
        assert "rem" in result

    def test_collapses_whitespace(self):
        result = normalize_verbatim("  well,   I   think  so.  ")
        assert result == "well, I think so."

    def test_handles_empty_string(self):
        assert normalize_verbatim("") == ""

    def test_handles_no_tags(self):
        text = "I think we should go."
        assert normalize_verbatim(text) == text

    def test_preserves_punctuation(self):
        result = normalize_verbatim("[UH] is the report ready?")
        assert result == "is the report ready?"

    def test_real_example_000101(self):
        verbatim = "So I I kind of gave up on the idea of using Quicken, at least for now."
        result = normalize_verbatim(verbatim)
        assert result == "So I I kind of gave up on the idea of using Quicken, at least for now."

    def test_real_example_000102(self):
        verbatim = "[UH] well, I actually my dad's my dad's almost ninety and he lives by himself and he's in good shape."
        result = normalize_verbatim(verbatim)
        assert "[UH]" not in result
        assert "well, I actually my dad's" in result


class TestToCanonical:
    """Tests for schema mapping."""

    def test_basic_mapping(self):
        raw = {
            "id": "TEST_001",
            "verbatim_transcript": "[UH] I want coffee.",
            "intended_transcript": "I want coffee.",
            "duration_in_s": 3.5,
            "speaker": "speaker1",
        }
        result = to_canonical(raw)

        assert result["id"] == "nyra_TEST_001"
        assert result["input"] == "I want coffee."
        assert result["output"] == "I want coffee."
        assert result["source"] == "nyra"
        assert result["source_id"] == "TEST_001"
        assert result["category"] == "nyra"
        assert result["generator_version"] == "v4.0.0"
        assert result["duration_in_s"] == 3.5
        assert result["speaker"] == "speaker1"

    def test_preserves_original_verbatim(self):
        raw = {
            "id": "TEST_002",
            "verbatim_transcript": "[UH] I want coffee.",
            "intended_transcript": "I want coffee.",
        }
        result = to_canonical(raw)
        assert result["original_verbatim"] == "[UH] I want coffee."

    def test_output_untouched(self):
        raw = {
            "id": "TEST_003",
            "verbatim_transcript": "[UH] well, [laughter] I think so.",
            "intended_transcript": "Well, I think so.",
        }
        result = to_canonical(raw)
        # Output should be exactly the intended transcript
        assert result["output"] == "Well, I think so."

    def test_handles_missing_fields(self):
        raw = {}
        result = to_canonical(raw)
        assert result["id"].startswith("nyra_")
        assert result["input"] == ""
        assert result["output"] == ""
        assert result["source"] == "nyra"


class TestNyraStats:
    """Tests for statistics computation."""

    def test_basic_stats(self):
        records = [
            {"input": "I want coffee.", "output": "I want coffee.", "original_verbatim": "I want coffee."},
            {"input": "Well I think so.", "output": "I think so.", "original_verbatim": "[UH] Well I think so."},
        ]
        stats = nyra_stats(records)
        assert stats["total"] == 2
        assert stats["noop_count"] == 1
        assert stats["noop_ratio"] == 0.5
        assert stats["has_bracket_tags"] == 1

    def test_empty_records(self):
        stats = nyra_stats([])
        assert stats["total"] == 0
        assert stats["noop_count"] == 0
