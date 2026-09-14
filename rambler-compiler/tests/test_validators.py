"""Tests for data.validators.quality — the gate checks V1-V10."""

import copy

import pytest

from data.generators.composer import GeneratorContext, Plan, compose
from data.validators.quality import (
    Issue,
    coverage_issues,
    dataset_stats,
    dedup_issues,
    validate_record,
)

CTX = GeneratorContext(seed=42)
MATRIX = {"filler+corr", "rep+corr", "corr+fmt"}


def make_record(**overrides):
    key = overrides.pop("key", "v_test")
    plan = Plan(
        key=key, category="mixed", transforms=("filler", "self_correction"),
        domain="work", difficulty="medium", combo="filler+corr",
    )
    record = compose(plan, CTX)
    record.update(overrides)
    return record


def error_codes(issues):
    return {i.check for i in issues if i.severity == "error"}


class TestStructure:
    def test_good_record_passes(self):
        record = make_record()
        assert error_codes(validate_record(record, MATRIX)) == set()

    def test_bad_category_is_an_error(self):
        assert "V1" in error_codes(validate_record(make_record(category="nonsense"), MATRIX))

    def test_num_transformations_mismatch_is_an_error(self):
        record = make_record(num_transformations=5)
        assert "V1" in error_codes(validate_record(record, MATRIX))

    def test_mixed_record_needs_matrix_combo(self):
        record = make_record(combo="not-a-combo")
        assert "V1" in error_codes(validate_record(record, MATRIX))


class TestIdentity:
    def test_input_equals_output_with_transforms_is_an_error(self):
        record = make_record(input="same", output="same")
        assert "V2" in error_codes(validate_record(record, MATRIX))

    def test_identity_record_without_transforms_is_fine(self):
        record = make_record(input="same", output="same", transformations=[], num_transformations=0)
        assert "V2" not in error_codes(validate_record(record, MATRIX))


class TestCorrectionConsistency:
    def test_abandoned_missing_from_input_is_an_error(self):
        record = make_record()
        record["input"] = record["input"].replace(record["corrections"][0]["abandoned"], "xyzzy")
        assert "V4" in error_codes(validate_record(record, MATRIX))

    def test_abandoned_leaking_into_output_is_an_error(self):
        record = make_record()
        record["output"] = record["output"] + " " + record["corrections"][0]["abandoned"]
        assert "V4" in error_codes(validate_record(record, MATRIX))

    def test_kept_missing_from_output_is_an_error(self):
        record = make_record()
        record["output"] = record["output"].replace(record["corrections"][0]["kept"], "xyzzy")
        assert "V4" in error_codes(validate_record(record, MATRIX))


class TestEntitiesAndFormatting:
    def test_entity_metadata_mismatch_is_an_error(self):
        record = make_record()
        record["entities"]["names"].append("ghostperson")
        assert "V5" in error_codes(validate_record(record, MATRIX))

    def test_broken_bullet_output_is_an_error(self):
        record = make_record(formatting={"type": "bullet_list", "items": ["milk"]}, output="just one line")
        assert "V6" in error_codes(validate_record(record, MATRIX))

    def test_markdown_in_non_formatting_output_is_an_error(self):
        record = make_record(output="- milk\n- eggs")
        assert "V6" in error_codes(validate_record(record, MATRIX))

    def test_overlong_output_is_an_error(self):
        record = make_record(output="word " * 80)
        assert "V7" in error_codes(validate_record(record, MATRIX))

    def test_lowercase_output_start_is_an_error(self):
        record = make_record(output="send the files to Sarah.")
        assert "V3" in error_codes(validate_record(record, MATRIX))

    def test_markdown_and_digit_output_starts_are_fine(self):
        assert "V3" not in error_codes(validate_record(
            make_record(output="- Milk\n- Eggs"), MATRIX))


class TestPreservation:
    def test_missing_preserved_substring_is_an_error(self):
        record = make_record(preservation_case=True, preserved=["never appears"])
        assert "V10" in error_codes(validate_record(record, MATRIX))


class TestDedup:
    def test_exact_duplicate_pair_is_flagged(self):
        a = make_record()
        b = copy.deepcopy(a)
        b["id"] = "other"
        issues = dedup_issues([a, b])
        assert any(i.check == "V8" and i.severity == "error" for i in issues)

    def test_near_duplicate_pair_is_flagged(self):
        a = make_record()
        b = copy.deepcopy(a)
        b["id"] = "other"
        # change one word of the input -> high 3-gram overlap
        b["input"] = b["input"].replace("the", "a", 1)
        issues = dedup_issues([a, b])
        assert any(i.check == "V8" and i.severity == "error" for i in issues)

    def test_distinct_records_pass(self):
        a = make_record(key="k1")
        b = make_record(key="k2")
        issues = dedup_issues([a, b])
        assert error_codes(issues) == set()


class TestCoverage:
    def test_missing_vocab_coverage_is_flagged(self):
        a = make_record()
        floors = {"names": 100, "items": 100}  # impossible with two records
        issues = coverage_issues([a], floors)
        assert any(i.check == "V9" and i.severity == "error" for i in issues)

    def test_no_floors_means_no_issues(self):
        issues = coverage_issues([make_record()], None)
        assert error_codes(issues) == set()


class TestStats:
    def test_stats_have_core_fields(self):
        a = make_record(key="s1")
        b = make_record(key="s2")
        stats = dataset_stats([a, b])
        for field in ("total", "per_category", "per_difficulty", "per_domain", "marker_histogram"):
            assert field in stats, field
        assert stats["total"] == 2
