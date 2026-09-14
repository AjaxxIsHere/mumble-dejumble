"""Tests for the V4 hybrid builder (data.dataset_builder.build_v4).

Pins: deterministic byte-identical regeneration, the five-bucket distribution,
canonical record schema, no duplicate pairs, and the hybrid real/template
sources (real text via a monkeypatched Nyra loader — no network in tests).
"""

import json

import pytest

from data.common import load_jsonl
from data.dataset_builder import V4BuildConfig, build_v4

TEMPLATE_CFG = V4BuildConfig(
    total=240,
    seed=42,
    real_ratio=0.0,
    coverage_floors=None,
    dataset_version="20260904.1",
)


def _all_records(out):
    return (load_jsonl(out / "splits" / "20260904.1" / "train.jsonl")
            + load_jsonl(out / "splits" / "20260904.1" / "validation.jsonl")
            + load_jsonl(out / "splits" / "20260904.1" / "test.jsonl"))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("v4build")
    summary = build_v4(TEMPLATE_CFG, out)
    return out, summary


class TestDistribution:
    def test_bucket_counts_match_targets(self, built):
        out, _ = built
        counts = {}
        for r in _all_records(out):
            counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
        # 240 * (20/15/25/30/10)%; the no-op bucket absorbs rounding.
        assert abs(counts["noop"] - 48) <= 1
        assert counts["simple"] == 36
        assert counts["correction"] == 60
        assert counts["compositional"] == 72
        assert counts["formatting"] == 24

    def test_formatting_is_template_sourced(self, built):
        out, _ = built
        for r in _all_records(out):
            if r["bucket"] == "formatting":
                assert r["source"] == "template"


class TestSchema:
    def test_all_records_have_v4_schema(self, built):
        out, _ = built
        required = {
            "input", "output", "category", "transformations", "num_transformations",
            "difficulty", "domain", "entities", "corrections", "reformulations",
            "formatting", "preservation_case", "preserved", "combo", "template_id",
            "seed", "generator_version", "bucket", "source",
        }
        for r in _all_records(out):
            assert required <= set(r), r["id"]
            assert r["generator_version"] == "v4.0.0"
            assert r["input"].strip() == r["input"]

    def test_noop_means_identity(self, built):
        out, _ = built
        for r in _all_records(out):
            if r["bucket"] == "noop":
                assert r["input"] == r["output"]
            else:
                assert r["input"] != r["output"]

    def test_no_duplicate_pairs(self, built):
        out, _ = built
        pairs = {(r["input"], r["output"]) for r in _all_records(out)}
        assert len(pairs) == len(_all_records(out))

    def test_manifest_and_stats_written(self, built):
        out, _ = built
        manifest = json.loads((out / "splits" / "20260904.1" / "manifest.json").read_text())
        assert manifest["dataset_version"] == "20260904.1"
        assert manifest["generator_version"] == "v4.0.0"
        stats = json.loads((out / "dataset_stats.json").read_text())
        assert stats["per_bucket"]["formatting"] == 24


class TestReproducibility:
    def test_regeneration_is_byte_identical(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        build_v4(TEMPLATE_CFG, a)
        build_v4(TEMPLATE_CFG, b)
        for name in ("train", "validation", "test"):
            assert (a / "splits" / "20260904.1" / f"{name}.jsonl").read_bytes() == (
                b / "splits" / "20260904.1" / f"{name}.jsonl").read_bytes(), name


class TestRealSource:
    """Hybrid path with a fake Nyra loader — never touches the network."""

    FAKE_RECORDS = [
        {"source_id": "nyra_1", "output": "The meeting is on Tuesday at 3 PM in Berlin."},
        {"source_id": "nyra_2", "output": "Please send the invoice to Sarah by Friday."},
        {"source_id": "nyra_3", "output": "I think we should call the client about the contract."},
        {"source_id": "nyra_4", "output": "The server in the lab runs on port 3000."},
    ]

    def test_real_pool_used_and_reused(self, monkeypatch, tmp_path):
        def fake_load_nyra(split="train", limit=None, cache_dir=None):
            return [dict(r) for r in self.FAKE_RECORDS[:limit or len(self.FAKE_RECORDS)]]

        monkeypatch.setattr("data.sources.nyra.load_nyra", fake_load_nyra)
        cfg = V4BuildConfig(total=100, seed=3, real_ratio=0.8, real_max_uses=3,
                            coverage_floors=None, dataset_version="20260904.1")
        build_v4(cfg, tmp_path)
        records = _all_records(tmp_path)
        real = [r for r in records if r["source"] == "real"]
        # Every real sentence is used at least once (identity or recipe);
        # reuse across different recipes makes real records outnumber the pool.
        assert {r["output"] for r in real} == {r["output"] for r in self.FAKE_RECORDS}
        assert len(real) >= 8
        # Identity no-ops never repeat a sentence.
        identity_pairs = [(r["input"], r["output"]) for r in real if r["bucket"] == "noop"]
        assert len(identity_pairs) == len(set(identity_pairs))
        # Reuse across recipes must never produce duplicate (input, output) pairs.
        pairs = [(r["input"], r["output"]) for r in records]
        assert len(pairs) == len(set(pairs))


class TestHoldOutGuard:
    def test_no_stress_record_leaks_into_splits(self, built):
        from data.stress_test import build_stress_records
        from data.stress_test_v4 import build_stress_records_v4

        out, _ = built
        stress_pairs = {
            (s["input"], s["output"])
            for s in build_stress_records() + build_stress_records_v4()
        }
        for r in _all_records(out):
            assert (r["input"], r["output"]) not in stress_pairs, r["id"]


class TestValidation:
    def test_bad_ratios_rejected(self):
        with pytest.raises(ValueError):
            V4BuildConfig(noop_ratio=0.5, simple_ratio=0.5,
                          correction_ratio=0.1, compositional_ratio=0.1,
                          formatting_ratio=0.1).validate()
