"""Tests for data.generators.composer — the Plan -> record orchestrator.

These pin the invariants the whole dataset hangs on:
- Same plan key + context seed -> byte-identical record (reproducibility).
- Corrections resolve to the clean output by construction.
- Compositional plans produce all their transformations with matching metadata.
- Formatting + correction composes: the artifact carries the corrected value.
"""

import random

from data.generators.composer import GeneratorContext, Plan, compose
from data.generators.entities import entity_spans


def make_plan(key="mix_a", **overrides):
    defaults = dict(
        key=key,
        category="mixed",
        transforms=("self_correction",),
        domain="work",
        difficulty="medium",
        longform=False,
        preservation=False,
        formatting=None,
        corrections=1,
        identity=False,
    )
    defaults.update(overrides)
    return Plan(**defaults)


CTX = GeneratorContext(seed=42)


class TestDeterminism:
    def test_same_key_and_seed_gives_identical_record(self):
        a = compose(make_plan(), CTX)
        b = compose(make_plan(), CTX)
        assert a == b

    def test_different_seed_gives_different_record(self):
        a = compose(make_plan(), GeneratorContext(seed=1))
        b = compose(make_plan(), GeneratorContext(seed=2))
        assert a != b

    def test_different_key_gives_different_record(self):
        a = compose(make_plan(key="a"), CTX)
        b = compose(make_plan(key="b"), CTX)
        assert a != b


class TestCorrectionInvariant:
    def test_abandoned_in_input_kept_in_output(self):
        record = compose(make_plan(category="correction", transforms=("self_correction",)), CTX)
        corr = record["corrections"][0]
        assert corr["abandoned"].lower() in record["input"].lower()
        assert corr["kept"].lower() in record["output"].lower()
        assert corr["abandoned"].lower() not in record["output"].lower()
        assert record["input"] != record["output"]


class TestComposition:
    def test_triple_composition(self):
        record = compose(
            make_plan(transforms=("filler", "repetition", "self_correction")), CTX
        )
        assert record["num_transformations"] == 3
        assert set(record["transformations"]) == {"filler", "repetition", "self_correction"}
        assert record["corrections"]
        assert record["input"] != record["output"]

    def test_multi_correction_uses_two_kinds(self):
        record = compose(
            make_plan(
                transforms=("self_correction", "self_correction"), corrections=2,
            ),
            CTX,
        )
        kinds = {c["kind"] for c in record["corrections"]}
        assert len(record["corrections"]) == 2
        assert len(kinds) == 2


class TestFormattingComposition:
    def test_artifact_carries_corrected_value(self):
        record = compose(
            make_plan(transforms=("self_correction", "formatting"), formatting="bullet_list"),
            CTX,
        )
        assert record["formatting"] is not None
        corr = record["corrections"][0]
        assert corr["kept"].lower() in record["output"].lower()
        assert corr["abandoned"].lower() not in record["output"].lower()
        # output must still be a structurally valid artifact
        lines = [l for l in record["output"].split("\n") if l.strip()]
        assert len(lines) >= 2


class TestLongform:
    def test_long_plan_produces_long_output(self):
        record = compose(
            make_plan(
                category="long", transforms=("filler",), domain="travel", longform=True,
                difficulty="hard",
            ),
            CTX,
        )
        assert len(record["output"].split()) >= 100
        assert record["input"] != record["output"]

    def test_long_with_correction_and_repetition(self):
        record = compose(
            make_plan(
                transforms=("repetition", "self_correction"), longform=True,
                difficulty="medium",
            ),
            CTX,
        )
        assert record["corrections"]
        assert record["num_transformations"] == 2


class TestPreservation:
    def test_preserved_substrings_survive(self):
        record = compose(
            make_plan(
                category="repetition", transforms=("filler",), preservation=True,
                difficulty="easy",
            ),
            CTX,
        )
        assert record["preservation_case"] is True
        assert record["preserved"]
        for sub in record["preserved"]:
            assert sub in record["output"]
        assert record["corrections"] == []


class TestIdentity:
    def test_identity_record_is_no_op(self):
        record = compose(
            make_plan(category="basic", transforms=(), identity=True), CTX
        )
        assert record["input"] == record["output"]
        assert record["num_transformations"] == 0


class TestMetadata:
    def test_entities_metadata_matches_extraction(self):
        record = compose(
            make_plan(transforms=("filler", "repetition", "self_correction")), CTX
        )
        extracted = {}
        for span in entity_spans(record["output"]):
            extracted.setdefault(span.kind, set()).add(span.canonical)
        span_kind_of = {
            "names": "name", "items": "item", "locations": "location", "tech": "tech",
            "numbers": "number", "times": "time", "dates": "date",
        }
        for key, values in record["entities"].items():
            if values:
                assert values == sorted(extracted[span_kind_of[key]]), key

    def test_slot_miss_returns_none(self):
        # A context whose template bank has no tech slot must yield None for a
        # plan that requires a tech correction (builder reallocates the slot).
        from data.generators.templates import Template

        bank = {
            "work": [
                Template(id="t1", domain="work", text="Send the {item} to {name}.", slots=("item", "name")),
            ]
        }
        ctx = GeneratorContext(seed=1, templates=bank)
        record = compose(
            make_plan(category="correction", transforms=("self_correction",),
                      correction_kinds=("tech",)),
            ctx,
        )
        assert record is None
