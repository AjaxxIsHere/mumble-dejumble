"""Tests for evaluation.failure_analysis — bucketing failures with reasons."""

from evaluation.failure_analysis import analyze, bucket
from evaluation.metrics import score_record


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


def bucket_of(record, actual):
    scored = score_record(record, actual)
    return bucket(record, actual, scored), scored


PERFECT = "Send it to Sarah on Tuesday at 3 PM."


class TestBuckets:
    def test_perfect_output_has_no_bucket(self):
        b, _ = bucket_of(make_record(), PERFECT)
        assert b is None

    def test_retained_abandoned_is_correction_resolution(self):
        b, scored = bucket_of(make_record(), "Send it to John on Tuesday at 3 PM.")
        assert b == "correction_resolution"
        assert scored["correction_resolution"] is False

    def test_priority_correction_beats_hallucination(self):
        # retains abandoned AND invents something: correction bucket wins
        record = make_record()
        actual = "Send it to John on Friday at 3 PM."
        b, _ = bucket_of(record, actual)
        assert b == "correction_resolution"

    def test_repetition_retained(self):
        record = make_record(
            transformations=["repetition"], corrections=[],
            input="the the report is ready", output="The report is ready.",
        )
        b, _ = bucket_of(record, "The the report is ready.")
        assert b == "repetition_removal"

    def test_preservation_deletion(self):
        record = make_record(
            transformations=["filler"], corrections=[],
            preservation_case=True, preserved=["from 3 to 4"],
            input="uh the meeting is from 3 to 4", output="The meeting is from 3 to 4.",
        )
        b, _ = bucket_of(record, "The meeting is at 4.")
        assert b == "repetition_removal"  # "deleted meaningful repetition"

    def test_hallucination(self):
        record = make_record(corrections=[])
        b, scored = bucket_of(record, "Send it to Sarah on Friday at 3 PM.")
        assert b == "hallucination"
        assert scored["hallucination_flag"] is True

    def test_information_deletion(self):
        record = make_record(corrections=[], transformations=["filler"],
                             entities={"names": [], "dates": [], "times": [], "numbers": [],
                                       "locations": [], "tech": [], "items": []})
        b, _ = bucket_of(record, "Send it.")
        assert b == "information_deletion"

    def test_formatting_failure(self):
        record = make_record(
            transformations=["formatting"], corrections=[],
            formatting={"type": "bullet_list", "items": ["milk", "eggs"]},
            output="- Milk\n- Eggs", input="make a list of milk and eggs",
        )
        b, _ = bucket_of(record, "Milk and eggs")
        assert b == "formatting_failure"

    def test_verbosity(self):
        b, _ = bucket_of(make_record(corrections=[]), "Here is the cleaned version: " + PERFECT)
        assert b == "verbosity"

    def test_partial_rewrite(self):
        b, _ = bucket_of(make_record(corrections=[]), "Send it to Sarah on Tuesday at 3 PM thanks")
        assert b == "partial_rewrite"

    def test_punctuation_only(self):
        b, _ = bucket_of(make_record(corrections=[]), "Send it to Sarah on Tuesday at 3 PM")
        assert b == "punctuation_only"


class TestAnalyze:
    def test_analyze_returns_buckets_and_examples(self):
        records = [
            make_record(id="a"),
            make_record(id="b", combo="filler+corr"),
        ]
        actuals = [PERFECT, "wrong output"]
        rows = [{"record": r, "actual": a} for r, a in zip(records, actuals)]
        out = analyze(rows)
        assert "bucket_counts" in out
        assert out["bucket_counts"]["correction_resolution"] >= 1
        assert "examples" in out
        assert "suggestions" in out
        assert len(out["examples"]["correction_resolution"]) == 1
