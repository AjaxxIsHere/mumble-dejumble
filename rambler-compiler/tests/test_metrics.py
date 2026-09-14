"""Tests for evaluation.metrics — the deterministic rule-judge metric suite."""

import random

import pytest

from evaluation.metrics import (
    aggregate,
    compositional_accuracy,
    levenshtein_wer,
    score_record,
)


def make_record(**overrides):
    record = {
        "id": "t1",
        "input": "uh send it to John no wait Sarah on Tuesday at 3 PM",
        "output": "Send it to Sarah on Tuesday at 3 PM.",
        "category": "mixed",
        "transformations": ["filler", "self_correction"],
        "num_transformations": 2,
        "difficulty": "medium",
        "domain": "work",
        "entities": {"names": ["sarah"], "dates": ["tuesday"], "times": ["3 pm"],
                     "numbers": [], "locations": [], "tech": [], "items": []},
        "corrections": [{"abandoned": "John", "kept": "Sarah", "kind": "name", "marker": "no wait"}],
        "reformulations": [],
        "formatting": None,
        "preservation_case": False,
        "preserved": [],
        "combo": "filler+corr",
    }
    record.update(overrides)
    return record


PERFECT = "Send it to Sarah on Tuesday at 3 PM."


class TestScoreRecord:
    def test_perfect_output_scores_fully(self):
        m = score_record(make_record(), PERFECT)
        assert m["exact_match"] is True
        assert m["normalized_match"] is True
        assert m["entity_preservation"] == 1.0
        assert m["correction_resolution"] is True
        assert m["hallucination_flag"] is False
        assert m["verbosity_flag"] is False
        assert m["compositional_pass"] is True

    def test_retained_abandoned_value_fails_correction_but_not_hallucination(self):
        record = make_record()
        m = score_record(record, "Send it to John on Tuesday at 3 PM.")
        assert m["correction_resolution"] is False
        # "john" is the abandoned value: correction failure, NOT hallucination
        assert m["hallucination_flag"] is False

    def test_invented_entity_flags_hallucination(self):
        m = score_record(make_record(), "Send it to Sarah on Friday at 3 PM.")
        assert m["hallucination_flag"] is True
        assert "friday" in m["hallucinated"]

    def test_filler_retention_fails_compositional(self):
        record = make_record(transformations=["filler"], corrections=[])
        m = score_record(record, "Uh, send it to Sarah on Tuesday at 3 PM.")
        assert m["compositional_pass"] is False

    def test_verbosity_flag(self):
        m = score_record(make_record(), "Here is the cleaned version: " + PERFECT + " Have a nice day!")
        assert m["verbosity_flag"] is True

    def test_exact_match_is_case_sensitive_but_normalized_is_not(self):
        m = score_record(make_record(), "send it to sarah on tuesday at 3 pm.")
        assert m["exact_match"] is False
        assert m["normalized_match"] is True

    def test_chained_correction_intermediate_must_not_survive(self):
        record = make_record(
            corrections=[
                {"abandoned": "Tuesday", "kept": "Thursday", "kind": "date", "marker": "no"},
                {"abandoned": "Thursday", "kept": "Monday", "kind": "date", "marker": "wait"},
            ],
            output="Let's meet Monday.",
            input="let's meet Tuesday no Thursday no wait Monday",
        )
        assert score_record(record, "Let's meet Monday.")["correction_resolution"] is True
        assert score_record(record, "Let's meet Thursday.")["correction_resolution"] is False


class TestWerCer:
    def test_levenshtein_wer(self):
        assert abs(levenshtein_wer("a b c", "a c") - 1 / 3) < 1e-9
        assert levenshtein_wer("a b", "a b") == 0.0

    def test_wer_and_cer_computed(self):
        m = score_record(make_record(), "Send it to Sarah on Tuesday at 3 pm")
        assert 0.0 <= m["wer"] <= 1.0
        assert 0.0 <= m["cer"] <= 1.0


class TestPreservationAndFormatting:
    def test_preservation_ok(self):
        record = make_record(
            transformations=["filler"], corrections=[],
            preservation_case=True, preserved=["from 3 to 4"],
            output="The meeting is from 3 to 4.",
            input="uh the meeting is from 3 to 4",
        )
        assert score_record(record, "The meeting is from 3 to 4.")["preservation_ok"] is True
        assert score_record(record, "The meeting is at 4.")["preservation_ok"] is False

    def test_formatting_ok(self):
        record = make_record(
            transformations=["formatting"], corrections=[],
            formatting={"type": "bullet_list", "items": ["milk", "eggs"]},
            output="- Milk\n- Eggs",
            input="make a list of milk and eggs",
        )
        assert score_record(record, "- Milk\n- Eggs")["formatting_ok"] is True
        assert score_record(record, "Milk and eggs")["formatting_ok"] is False


class TestAggregate:
    def test_compositional_accuracy_is_mean_over_mixed(self):
        records = [make_record(id="a"), make_record(id="b", combo="filler+corr")]
        scored = [
            {"record": records[0], **score_record(records[0], PERFECT)},
            {"record": records[1], **score_record(records[1], "wrong output")},
        ]
        acc = compositional_accuracy(scored)
        assert acc["overall"] == 0.5

    def test_aggregate_has_sections(self):
        records = [make_record(id="a"), make_record(id="b", combo="rep+corr", category="mixed",
                                                    transformations=["repetition", "self_correction"])]
        scored = [{"record": r, **score_record(r, PERFECT)} for r in records]
        agg = aggregate(scored)
        for section in ("overall", "per_category", "per_difficulty", "per_combo", "compositional"):
            assert section in agg, section
