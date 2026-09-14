"""Tests for the V4 rule-based realizer (data.augmentation.realizer).

Pins: deterministic realizations for a fixed seed, the clean target is never
modified, self-corrections need an extractable entity (SlotError -> None),
and unsupported transformations are rejected loudly.
"""

import pytest

from data.augmentation.realizer import (
    RealizationError,
    entity_kinds,
    realize,
    to_record,
)

CLEAN = "The meeting is on Tuesday at 3 PM in Berlin."


class TestDeterminism:
    def test_same_key_same_output(self):
        a = realize(CLEAN, transforms=("filler",), difficulty="easy", seed_key="k1")
        b = realize(CLEAN, transforms=("filler",), difficulty="easy", seed_key="k1")
        assert a.input == b.input
        assert a.seed == b.seed

    def test_output_never_modified(self):
        for transforms in (("filler",), ("repetition",), ("self_correction",),
                           ("reformulation",), ("filler", "repetition", "self_correction")):
            r = realize(CLEAN, transforms=transforms, difficulty="easy", seed_key="k2")
            assert r.output == CLEAN

    def test_different_seed_generally_differs(self):
        # Long text with many insertion gaps: two seeds land somewhere else.
        text = ("I need to send the quarterly report to the whole team before "
                "the board meeting on Friday and include the budget analysis.")
        outs = {
            realize(text, transforms=("filler",), difficulty="medium", seed_key=s).input
            for s in ("a", "b", "c", "d", "e")
        }
        assert len(outs) > 1


class TestTransformRealization:
    def test_noop_identity(self):
        r = realize(CLEAN, transforms=(), difficulty="easy", seed_key="noop")
        assert r.input == CLEAN
        assert r.output == CLEAN

    def test_filler_changes_input(self):
        r = realize(CLEAN, transforms=("filler",), difficulty="hard", seed_key="f")
        assert r.input != CLEAN

    def test_correction_metadata_invariant(self):
        r = realize(CLEAN, transforms=("self_correction",), difficulty="easy",
                    seed_key="corr")
        assert r.corrections, "expected a correction"
        c = r.corrections[0]
        assert c["kept"].lower() in CLEAN.lower()
        assert c["abandoned"] != c["kept"]
        assert c["abandoned"].lower() in r.input.lower()

    def test_reformulation_records_metadata(self):
        r = realize(CLEAN, transforms=("reformulation",), difficulty="easy",
                    seed_key="reform")
        assert r.reformulations

    def test_multi_correction_needs_distinct_kinds(self):
        text = "Meet Sarah on Tuesday."
        r = realize(text, transforms=("self_correction",), difficulty="easy",
                    seed_key="multi", corrections=2)
        if r is not None:
            assert len(r.corrections) == 2


class TestFailures:
    def test_correction_returns_none_without_entities(self):
        text = "I think we should go to the store now."
        if not entity_kinds(text):
            assert realize(text, transforms=("self_correction",), difficulty="easy",
                           seed_key="x") is None

    def test_unknown_transform_rejected(self):
        with pytest.raises(RealizationError):
            realize(CLEAN, transforms=("hesitation",), difficulty="easy", seed_key="x")

    def test_formatting_not_realizable(self):
        # Formatting is a spoken-command -> artifact pair, handled by the
        # builder's make_formatting path, never by the realizer.
        with pytest.raises(RealizationError):
            realize(CLEAN, transforms=("formatting",), difficulty="easy", seed_key="x")


class TestToRecord:
    def test_canonical_schema(self):
        r = realize(CLEAN, transforms=("filler",), difficulty="medium", seed_key="rec")
        record = to_record(r, plan={
            "category": "filler", "transforms": ("filler",), "difficulty": "medium",
            "domain": "everyday", "combo": None, "bucket": "simple", "source": "real",
        })
        assert record["input"] == r.input
        assert record["output"] == CLEAN
        assert record["generator_version"] == "v4.0.0"
        assert record["bucket"] == "simple"
        assert record["source"] == "real"
        assert record["num_transformations"] == 1
        assert record["transformations"] == ["filler"]
        assert record["template_id"] is None
