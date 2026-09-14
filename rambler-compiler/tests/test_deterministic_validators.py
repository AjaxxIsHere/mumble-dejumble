"""Tests for the deterministic quality gate."""

import pytest

from data.validators.deterministic import (
    check_schema_valid,
    check_non_empty,
    check_target_preservation,
    check_recipe_realized,
    check_no_entity_introduction,
    check_correction_metadata,
    check_length_bounds,
    validate_augmented_record,
    filter_records,
)


class TestCheckSchemaValid:
    def test_valid_record(self):
        result = check_schema_valid({"input": "hello", "output": "hello"})
        assert result.passed

    def test_missing_input(self):
        result = check_schema_valid({"output": "hello"})
        assert not result.passed
        assert result.reason_code == "MISSING_FIELDS"

    def test_empty_input(self):
        result = check_schema_valid({"input": "", "output": "hello"})
        assert not result.passed


class TestCheckNonEmpty:
    def test_valid(self):
        result = check_non_empty({"input": "hello world", "output": "hello world"})
        assert result.passed

    def test_short_input(self):
        result = check_non_empty({"input": "hi", "output": "hello world"})
        assert not result.passed
        assert result.reason_code == "EMPTY_INPUT"

    def test_short_output(self):
        result = check_non_empty({"input": "hello world", "output": "hi"})
        assert not result.passed
        assert result.reason_code == "EMPTY_OUTPUT"


class TestCheckTargetPreservation:
    def test_output_matches_clean(self):
        record = {"input": "uh hello", "output": "Hello.", "original_clean": "Hello."}
        result = check_target_preservation(record)
        assert result.passed

    def test_output_differs_from_clean(self):
        record = {"input": "uh hello", "output": "Goodbye.", "original_clean": "Hello."}
        result = check_target_preservation(record)
        assert not result.passed
        assert result.reason_code == "TARGET_MODIFIED"


class TestCheckRecipeRealized:
    def test_noop_recipe_same_text(self):
        record = {"input": "Hello.", "output": "Hello.", "recipe": {"transformations": []}}
        result = check_recipe_realized(record)
        assert result.passed

    def test_noop_recipe_different_text(self):
        record = {"input": "uh Hello.", "output": "Hello.", "recipe": {"transformations": []}}
        result = check_recipe_realized(record)
        assert not result.passed
        assert result.reason_code == "UNEXPECTED_CHANGE"

    def test_filler_recipe_with_change(self):
        record = {
            "input": "uh I want coffee.",
            "output": "I want coffee.",
            "recipe": {"transformations": [{"type": "filler"}]},
        }
        result = check_recipe_realized(record)
        assert result.passed

    def test_filler_recipe_no_change(self):
        record = {
            "input": "I want coffee.",
            "output": "I want coffee.",
            "recipe": {"transformations": [{"type": "filler"}]},
        }
        result = check_recipe_realized(record)
        assert not result.passed
        assert result.reason_code == "TRANSFORMATION_NOT_REALIZED"


class TestCheckCorrectionMetadata:
    def test_valid_correction(self):
        record = {
            "input": "Tuesday no wait Thursday.",
            "output": "Thursday.",
            "recipe": {
                "transformations": [{
                    "type": "self_correction",
                    "abandoned": "Tuesday",
                    "retained": "Thursday",
                }],
            },
        }
        result = check_correction_metadata(record)
        assert result.passed

    def test_abandoned_not_in_input(self):
        record = {
            "input": "I want coffee.",
            "output": "I want coffee.",
            "recipe": {
                "transformations": [{
                    "type": "self_correction",
                    "abandoned": "Tuesday",
                    "retained": "Thursday",
                }],
            },
        }
        result = check_correction_metadata(record)
        assert not result.passed
        assert result.reason_code == "ABANDONED_MISSING"

    def test_identical_values(self):
        record = {
            "input": "Tuesday Tuesday.",
            "output": "Tuesday.",
            "recipe": {
                "transformations": [{
                    "type": "self_correction",
                    "abandoned": "Tuesday",
                    "retained": "Tuesday",
                }],
            },
        }
        result = check_correction_metadata(record)
        assert not result.passed
        assert result.reason_code == "IDENTICAL_CORRECTION"


class TestCheckLengthBounds:
    def test_normal_lengths(self):
        record = {"input": "I want to go to the store today.", "output": "I want to go to the store."}
        result = check_length_bounds(record)
        assert result.passed

    def test_input_too_long(self):
        record = {
            "input": " ".join(["word"] * 100),
            "output": " ".join(["word"] * 10),
        }
        result = check_length_bounds(record)
        assert not result.passed
        assert result.reason_code == "EXCESSIVE_LENGTH"

    def test_output_expanded(self):
        record = {
            "input": "Hello world, this is a test.",
            "output": " ".join(["word"] * 20),
        }
        result = check_length_bounds(record)
        assert not result.passed
        assert result.reason_code == "OUTPUT_EXPANDED"


class TestValidateAugmentedRecord:
    def test_valid_augmented_record(self):
        record = {
            "id": "test_001",
            "input": "uh I want coffee.",
            "output": "I want coffee.",
            "original_clean": "I want coffee.",
            "recipe": {
                "transformations": [{"type": "filler", "location": "sentence_start"}],
            },
        }
        report = validate_augmented_record(record)
        assert report.passed
        assert report.checks["schema_valid"]
        assert report.checks["recipe_realized"]

    def test_invalid_record_fails(self):
        record = {"id": "test_002"}  # missing required fields
        report = validate_augmented_record(record)
        assert not report.passed


class TestFilterRecords:
    def test_filter_passes_valid(self):
        records = [
            {
                "input": "uh hello.",
                "output": "Hello.",
                "original_clean": "Hello.",
                "recipe": {"transformations": [{"type": "filler"}]},
            },
        ]
        passed, rejected = filter_records(records)
        assert len(passed) == 1
        assert len(rejected) == 0

    def test_filter_rejects_invalid(self):
        records = [
            {"id": "bad"},  # missing required fields
        ]
        passed, rejected = filter_records(records)
        assert len(passed) == 0
        assert len(rejected) == 1
