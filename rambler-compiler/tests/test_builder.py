"""Tests for data.dataset_builder — plan allocation, dedup, stratified split,
and byte-identical regeneration (the reproducibility anchor)."""

import json
from pathlib import Path

import pytest

from data.common import load_jsonl
from data.dataset_builder import BuildConfig, build

SMALL_CFG = BuildConfig(
    total=240,
    distribution={
        "basic": 0.15, "filler": 0.10, "repetition": 0.10, "correction": 0.20,
        "reformulation": 0.10, "formatting": 0.05, "long": 0.15, "mixed": 0.15,
    },
    composition_matrix={
        "filler+rep": 12, "filler+corr": 12, "rep+corr": 12,
    },
    preservation_ratio=0.10,
    identity_ratio=0.15,
    coverage_floors=None,
    seed=42,
    dataset_version="20260901.1",
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("build")
    summary = build(SMALL_CFG, out)
    return out, summary


def read_split(out: Path, name: str):
    return load_jsonl(out / "splits" / "20260901.1" / f"{name}.jsonl")


class TestCounts:
    def test_category_counts_match_config(self, built):
        out, _ = built
        records = read_split(out, "train") + read_split(out, "validation") + read_split(out, "test")
        expected = {
            "basic": 36, "filler": 24, "repetition": 24, "correction": 48,
            "reformulation": 24, "formatting": 12, "long": 36, "mixed": 36,
        }
        counts = {}
        for r in records:
            counts[r["category"]] = counts.get(r["category"], 0) + 1
        assert counts == expected

    def test_matrix_combo_counts(self, built):
        out, _ = built
        records = read_split(out, "train") + read_split(out, "validation") + read_split(out, "test")
        combos = {}
        for r in records:
            if r["combo"]:
                combos[r["combo"]] = combos.get(r["combo"], 0) + 1
        assert combos == {"filler+rep": 12, "filler+corr": 12, "rep+corr": 12}


class TestSplit:
    def test_stratified_85_10_5(self, built):
        out, _ = built
        for name, frac in (("train", 0.85), ("validation", 0.10), ("test", 0.05)):
            records = read_split(out, name)
            for cat in ("basic", "filler", "repetition", "correction", "reformulation",
                        "formatting", "long", "mixed"):
                total = {"basic": 36, "filler": 24, "repetition": 24, "correction": 48,
                         "reformulation": 24, "formatting": 12, "long": 36, "mixed": 36}[cat]
                expected = round(total * frac)
                actual = sum(1 for r in records if r["category"] == cat)
                assert abs(actual - expected) <= 1, (name, cat, actual, expected)

    def test_every_category_present_in_test(self, built):
        out, _ = built
        cats = {r["category"] for r in read_split(out, "test")}
        assert len(cats) == 8


class TestQuality:
    def test_no_duplicate_pairs(self, built):
        out, _ = built
        records = read_split(out, "train") + read_split(out, "validation") + read_split(out, "test")
        pairs = {(r["input"], r["output"]) for r in records}
        assert len(pairs) == len(records)

    def test_all_records_have_ids_and_pass_structure(self, built):
        out, _ = built
        records = read_split(out, "train") + read_split(out, "validation") + read_split(out, "test")
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids))
        for r in records:
            assert r["input"].strip() == r["input"]
            assert r["generator_version"]

    def test_manifest_written(self, built):
        out, _ = built
        manifest = json.loads((out / "splits" / "20260901.1" / "manifest.json").read_text())
        assert manifest["dataset_version"] == "20260901.1"
        assert manifest["seed"] == 42

    def test_dataset_stats_written(self, built):
        out, _ = built
        stats = json.loads((out / "dataset_stats.json").read_text())
        assert stats["total"] >= 200  # gaps may reduce the count slightly


class TestReproducibility:
    def test_regeneration_is_byte_identical(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        build(SMALL_CFG, a)
        build(SMALL_CFG, b)
        for name in ("train", "validation", "test"):
            lines_a = (a / "splits" / "20260901.1" / f"{name}.jsonl").read_bytes()
            lines_b = (b / "splits" / "20260901.1" / f"{name}.jsonl").read_bytes()
            assert lines_a == lines_b, name
