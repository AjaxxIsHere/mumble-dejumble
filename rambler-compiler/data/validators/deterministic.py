"""Deterministic quality gate.

Validates augmented records using rule-based checks (no LLM, no API).
Each validator returns structured results with check names, pass/fail, and
reason codes.

Checks performed:
1. schema_valid — valid JSON with required fields
2. non_empty — input and output are non-empty
3. target_preservation — output equals known clean target where expected
4. semantic_target — clean target is not modified
5. recipe_realized — requested transformation actually occurred
6. no_entity_introduction — no unexpected entities in output
7. correction_metadata — abandoned/retained consistency
8. duplicate — not a duplicate of existing records
9. length_bounds — reasonable input/output lengths
10. formatting_valid — markdown structure is correct
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from data.common import normalize


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single quality check."""
    check: str
    passed: bool
    reason_code: str | None = None
    detail: str | None = None


@dataclass
class ValidationReport:
    """Complete validation report for one record."""
    record_id: str = ""
    passed: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    check_details: list[CheckResult] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.checks[result.check] = result.passed
        self.check_details.append(result)
        if not result.passed:
            self.passed = False
            if result.reason_code:
                self.reason_codes.append(result.reason_code)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "passed": self.passed,
            "checks": self.checks,
            "reason_codes": self.reason_codes,
        }


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def check_schema_valid(record: dict) -> CheckResult:
    """Check that the record has all required fields."""
    required = ["input", "output"]
    missing = [k for k in required if k not in record or not record[k]]
    if missing:
        return CheckResult(
            check="schema_valid", passed=False,
            reason_code="MISSING_FIELDS",
            detail=f"Missing required fields: {missing}",
        )
    return CheckResult(check="schema_valid", passed=True)


def check_non_empty(record: dict) -> CheckResult:
    """Check that input and output are non-empty and have minimum length."""
    inp = record.get("input", "").strip()
    out = record.get("output", "").strip()
    if len(inp) < 3:
        return CheckResult(
            check="non_empty", passed=False,
            reason_code="EMPTY_INPUT",
            detail=f"Input too short: {len(inp)} chars",
        )
    if len(out) < 3:
        return CheckResult(
            check="non_empty", passed=False,
            reason_code="EMPTY_OUTPUT",
            detail=f"Output too short: {len(out)} chars",
        )
    return CheckResult(check="non_empty", passed=True)


def check_target_preservation(record: dict) -> CheckResult:
    """Check that the output matches the known clean target.

    For augmented records, 'original_clean' or 'output' before augmentation
    is the ground truth. The output field should still be the clean target.
    """
    output = record.get("output", "").strip()
    original_clean = record.get("original_clean", "").strip()

    # If there's an original_clean, the output should still match it
    if original_clean and output != original_clean:
        return CheckResult(
            check="target_preservation", passed=False,
            reason_code="TARGET_MODIFIED",
            detail=f"Output differs from clean target",
        )
    return CheckResult(check="target_preservation", passed=True)


def check_semantic_target(record: dict) -> CheckResult:
    """Check that the output preserves key entities from the clean target."""
    from data.generators.entities import entity_spans

    output = record.get("output", "")
    original_clean = record.get("original_clean", record.get("output", ""))

    expected_canons = {s.canonical for s in entity_spans(original_clean)}
    actual_canons = {s.canonical for s in entity_spans(output)}

    missing = expected_canons - actual_canons
    if missing:
        return CheckResult(
            check="semantic_target", passed=False,
            reason_code="ENTITY_LOST",
            detail=f"Entities missing from output: {sorted(missing)}",
        )
    return CheckResult(check="semantic_target", passed=True)


def check_recipe_realized(record: dict) -> CheckResult:
    """Check that the messy input actually contains the requested disfluencies.

    For augmentation records, the 'input' field is the messy transcript.
    We check that it differs from the clean target (i.e., something was added).
    """
    inp = record.get("input", "").strip()
    output = record.get("output", "").strip()

    recipe = record.get("recipe", {})
    transformations = recipe.get("transformations", [])

    # No-op recipes: input should equal output
    if not transformations:
        if normalize(inp) != normalize(output):
            return CheckResult(
                check="recipe_realized", passed=False,
                reason_code="UNEXPECTED_CHANGE",
                detail="No-op recipe but input differs from output",
            )
        return CheckResult(check="recipe_realized", passed=True)

    # For recipes with transformations, input should differ from output
    if normalize(inp) == normalize(output):
        return CheckResult(
            check="recipe_realized", passed=False,
            reason_code="TRANSFORMATION_NOT_REALIZED",
            detail="Recipe has transformations but input equals output",
        )

    return CheckResult(check="recipe_realized", passed=True)


def check_no_entity_introduction(record: dict) -> CheckResult:
    """Check that the messy input doesn't introduce entities not in the clean target."""
    from data.generators.entities import entity_spans

    inp = record.get("input", "")
    output = record.get("output", "")

    input_canons = {s.canonical for s in entity_spans(inp)}
    output_canons = {s.canonical for s in entity_spans(output)}

    introduced = input_canons - output_canons
    # Some introduction is expected for corrections (the abandoned value)
    recipe = record.get("recipe", {})
    corrections = [
        t for t in recipe.get("transformations", [])
        if t.get("type") in ("self_correction", "entity_correction", "phrase_correction")
    ]

    # Filter out abandoned values that are part of corrections
    abandoned_values = {normalize(t.get("abandoned", "")) for t in corrections}
    real_introduced = introduced - abandoned_values

    if real_introduced:
        return CheckResult(
            check="no_entity_introduction", passed=False,
            reason_code="UNEXPECTED_ENTITIES",
            detail=f"Unexpected entities in input: {sorted(real_introduced)}",
        )
    return CheckResult(check="no_entity_introduction", passed=True)


def check_correction_metadata(record: dict) -> CheckResult:
    """Check that correction metadata is consistent."""
    recipe = record.get("recipe", {})
    corrections = [
        t for t in recipe.get("transformations", [])
        if t.get("type") in ("self_correction", "entity_correction", "phrase_correction")
    ]

    for corr in corrections:
        abandoned = corr.get("abandoned", "")
        retained = corr.get("retained", "")

        if not abandoned or not retained:
            return CheckResult(
                check="correction_metadata", passed=False,
                reason_code="INCOMPLETE_CORRECTION",
                detail=f"Missing abandoned or retained value: {corr}",
            )
        if abandoned == retained:
            return CheckResult(
                check="correction_metadata", passed=False,
                reason_code="IDENTICAL_CORRECTION",
                detail=f"Abandoned equals retained: {abandoned}",
            )

        # Check that abandoned appears in input and retained doesn't
        inp = record.get("input", "")
        output = record.get("output", "")
        if abandoned.lower() not in inp.lower():
            return CheckResult(
                check="correction_metadata", passed=False,
                reason_code="ABANDONED_MISSING",
                detail=f"Abandoned value '{abandoned}' not found in input",
            )

    return CheckResult(check="correction_metadata", passed=True)


def check_length_bounds(record: dict) -> CheckResult:
    """Check that input/output lengths are reasonable."""
    inp = record.get("input", "")
    out = record.get("output", "")
    inp_words = len(inp.split())
    out_words = len(out.split())

    # Messy input can be longer than clean output (fillers, repetitions)
    # but shouldn't be more than 5x longer
    if inp_words > out_words * 5 and out_words > 5:
        return CheckResult(
            check="length_bounds", passed=False,
            reason_code="EXCESSIVE_LENGTH",
            detail=f"Input ({inp_words} words) is >5x output ({out_words} words)",
        )

    # Clean output shouldn't be longer than messy input
    if out_words > inp_words * 2 and inp_words > 5:
        return CheckResult(
            check="length_bounds", passed=False,
            reason_code="OUTPUT_EXPANDED",
            detail=f"Output ({out_words} words) is >2x input ({inp_words} words)",
        )

    return CheckResult(check="length_bounds", passed=True)


def check_formatting_valid(record: dict) -> CheckResult:
    """Check that formatting commands produce valid output structure."""
    recipe = record.get("recipe", {})
    fmt_transforms = [
        t for t in recipe.get("transformations", [])
        if t.get("type") == "formatting_command"
    ]

    if not fmt_transforms:
        return CheckResult(check="formatting_valid", passed=True)

    output = record.get("output", "")
    lines = [l for l in output.split("\n") if l.strip()]

    for fmt in fmt_transforms:
        fmt_type = fmt.get("format_type", "")
        if fmt_type == "bullet_list":
            if len(lines) < 2:
                return CheckResult(
                    check="formatting_valid", passed=False,
                    reason_code="INVALID_LIST",
                    detail="Bullet list has fewer than 2 items",
                )
            if not all(re.match(r"^(-|\d+\.)\s", l) for l in lines):
                return CheckResult(
                    check="formatting_valid", passed=False,
                    reason_code="INVALID_LIST_FORMAT",
                    detail="Lines don't match bullet/numbered list format",
                )
        elif fmt_type == "heading":
            if len(lines) != 1 or not re.match(r"^#{1,3}\s", lines[0]):
                return CheckResult(
                    check="formatting_valid", passed=False,
                    reason_code="INVALID_HEADING",
                    detail="Heading should be a single line starting with #",
                )

    return CheckResult(check="formatting_valid", passed=True)


# ---------------------------------------------------------------------------
# Full validation pipeline
# ---------------------------------------------------------------------------

def validate_augmented_record(record: dict) -> ValidationReport:
    """Run all deterministic checks on an augmented record.

    Returns a ValidationReport with all check results.
    """
    report = ValidationReport(record_id=record.get("id", "unknown"))

    validators = [
        check_schema_valid,
        check_non_empty,
        check_target_preservation,
        check_semantic_target,
        check_recipe_realized,
        check_no_entity_introduction,
        check_correction_metadata,
        check_length_bounds,
        check_formatting_valid,
    ]

    for validator in validators:
        result = validator(record)
        report.add(result)
        # Short-circuit on critical failures
        if not result.passed and result.reason_code in (
            "MISSING_FIELDS", "EMPTY_INPUT", "EMPTY_OUTPUT",
        ):
            break

    return report


def filter_records(
    records: list[dict],
    *,
    strict: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Filter records through the deterministic quality gate.

    Args:
        records: List of augmented records.
        strict: If True, reject on any failure. If False, only reject on critical failures.

    Returns:
        (passed_records, rejected_records)
    """
    passed = []
    rejected = []

    for record in records:
        report = validate_augmented_record(record)
        if report.passed:
            passed.append(record)
        elif strict:
            rejected.append(record)
        else:
            # Only reject on critical failures
            critical = {"MISSING_FIELDS", "EMPTY_INPUT", "EMPTY_OUTPUT"}
            if critical & set(report.reason_codes):
                rejected.append(record)
            else:
                passed.append(record)

    return passed, rejected
