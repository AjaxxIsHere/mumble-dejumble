"""Tests for the Rambler Stress Test — the final hold-out.

Pins: 200 records, 50 per level, valid schema, the spec's smoke-test cases,
and NON-OVERLAP with generated splits (the stress set must never leak into
training/validation/test — spec §23).
"""

from data.dataset_builder import BuildConfig, build
from data.stress_test import (
    EXPECTED_PER_LEVEL,
    build_stress_records,
    validate_stress,
    write_stress_jsonl,
)


class TestShape:
    def test_200_records_50_per_level(self):
        records = build_stress_records()
        assert len(records) == 200
        counts = {}
        for r in records:
            counts[r["level"]] = counts.get(r["level"], 0) + 1
        assert counts == {"easy": 50, "medium": 50, "hard": 50, "extreme": 50}

    def test_validation_passes(self):
        assert validate_stress(build_stress_records()) == []

    def test_ids_unique_with_stress_prefix(self):
        ids = [r["id"] for r in build_stress_records()]
        assert all(i.startswith("stress_") for i in ids)
        assert len(ids) == len(set(ids))

    def test_jsonl_written(self, tmp_path):
        p = tmp_path / "stress.jsonl"
        assert write_stress_jsonl(p) == 200
        from data.common import load_jsonl

        records = load_jsonl(p)
        assert len(records) == 200


class TestGoldenProperty:
    def test_perfect_output_passes_every_metric(self):
        # The golden property: a prediction equal to the expected output must
        # pass correction resolution and compositional checks for ALL 200 records.
        from evaluation.metrics import score_record

        for r in build_stress_records():
            m = score_record(r, r["output"])
            assert m["correction_resolution"] is True, r["id"]
            assert m["compositional_pass"] is True, r["id"]
            assert m["exact_match"] is True, r["id"]


class TestSpecCases:
    def test_spec_smoke_test_is_present(self):
        # The exact case Experiment 1 failed — spec §11/§3.
        inputs = {r["input"] for r in build_stress_records()}
        assert (
            "uh i i wanted to send sarah the project files tomorrow actually no wednesday morning"
            in inputs
        )
        assert (
            "make a list of milk eggs bread actually remove eggs and add paratha" in inputs
        )
        assert (
            "tell alex the the server crashed because of port 8080 wait port 3000 and make it sound professional"
            in inputs
        )
        assert "I was going to say Tuesday, but Wednesday works better." in inputs


class TestNoOverlapWithSplits:
    def test_no_stress_record_appears_in_generated_splits(self, tmp_path):
        # Build a small dataset and assert zero overlap with the stress set.
        small = BuildConfig(
            total=240,
            distribution={
                "basic": 0.15, "filler": 0.10, "repetition": 0.10, "correction": 0.20,
                "reformulation": 0.10, "formatting": 0.05, "long": 0.15, "mixed": 0.15,
            },
            composition_matrix={"filler+rep": 12, "filler+corr": 12, "rep+corr": 12},
            preservation_ratio=0.10,
            identity_ratio=0.15,
            coverage_floors=None,
            seed=42,
            dataset_version="20260901.1",
        )
        build(small, tmp_path)
        from data.common import load_jsonl

        split_pairs = set()
        for name in ("train", "validation", "test"):
            for r in load_jsonl(tmp_path / "splits" / "20260901.1" / f"{name}.jsonl"):
                split_pairs.add((r["input"], r["output"]))
        for r in build_stress_records():
            assert (r["input"], r["output"]) not in split_pairs, r["id"]
