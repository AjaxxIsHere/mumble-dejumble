"""Tests for data.common — the shared stdlib helpers and the single source of
truth for the production instruction, categories, and canonical transform names."""

import json
import random

import pytest

from data.common import (
    CATEGORIES,
    INSTRUCTION,
    TRANSFORMS,
    load_json,
    load_jsonl,
    md5_rng,
    normalize,
    parse_dataset_version,
    write_json,
    write_jsonl,
)


class TestNormalize:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalize("  I Wanted   to  Send\tThe Files ") == "i wanted to send the files"

    def test_strips_surrounding_double_quotes(self):
        assert normalize('"I wanted to send the files Wednesday."') == "i wanted to send the files wednesday."

    def test_does_not_strip_internal_quotes(self):
        assert normalize('he said "send it" today') == 'he said "send it" today'

    def test_handles_empty_string(self):
        assert normalize("") == ""


class TestMd5Rng:
    def test_same_key_gives_identical_sequence(self):
        a = md5_rng("seed:plan_01")
        b = md5_rng("seed:plan_01")
        assert [a.randint(0, 10**6) for _ in range(20)] == [b.randint(0, 10**6) for _ in range(20)]

    def test_different_keys_give_different_sequences(self):
        a = md5_rng("seed:plan_01")
        b = md5_rng("seed:plan_02")
        assert [a.randint(0, 10**6) for _ in range(10)] != [b.randint(0, 10**6) for _ in range(10)]

    def test_returns_a_random_instance(self):
        assert isinstance(md5_rng("k"), random.Random)


class TestJsonlIo:
    def test_roundtrip_preserves_records(self, tmp_path):
        records = [
            {"id": "a_0001", "input": "uh hi", "output": "Hi."},
            {"id": "a_0002", "input": "the the file", "output": "the file", "transformations": ["repetition"]},
        ]
        p = tmp_path / "records.jsonl"
        write_jsonl(p, records)
        assert load_jsonl(p) == records

    def test_load_json_roundtrip(self, tmp_path):
        p = tmp_path / "stats.json"
        write_json(p, {"n": 3, "ok": True})
        assert load_json(p) == {"n": 3, "ok": True}


class TestDatasetVersion:
    def test_accepts_valid_version(self):
        assert parse_dataset_version("20260831.1") == "20260831.1"

    @pytest.mark.parametrize("bad", ["20260831", "bad", "20260831.x", "20260831.", ".1", "20261345.1"])
    def test_rejects_invalid_versions(self, bad):
        with pytest.raises(ValueError):
            parse_dataset_version(bad)


class TestConstants:
    def test_instruction_is_the_production_prompt(self):
        assert "speech-to-text compiler" in INSTRUCTION
        assert "Output only the final text." in INSTRUCTION
        assert len(INSTRUCTION) < 450  # keep the production prompt short (spec text is ~404 chars)

    def test_categories_are_the_eight_spec_categories(self):
        assert set(CATEGORIES) == {
            "basic",
            "filler",
            "repetition",
            "correction",
            "reformulation",
            "formatting",
            "long",
            "mixed",
        }

    def test_transforms_are_canonical(self):
        assert TRANSFORMS == {
            "filler",
            "repetition",
            "self_correction",
            "reformulation",
            "formatting",
        }
