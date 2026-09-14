"""Tests for data.generators.injectors — the noise engine.

Core invariants pinned here:
- Every injector changes the text and is deterministic for a fixed rng.
- Correction: abandoned value present in input, kept value (== the clean
  text's verbatim span) present in input, abandoned != kept.
- Repetition never touches protected spans.
- Filler never splits a digit/entity span.
- Formatting produces structurally valid artifacts whose content matches
  the spoken command.
"""

import random
import re

import pytest

from data.generators.entities import SlotError, entity_spans
from data.generators.injectors import (
    FILLERS,
    _render_correction,
    apply_correction,
    apply_filler,
    apply_reformulation,
    apply_repetition,
    make_formatting,
)

CLEAN = "Send the project files to Sarah on Tuesday at 3 PM."


def filler_count(text: str) -> int:
    # Longest-first alternation so "you know what" is counted once, not also as "you know".
    alternation = "|".join(sorted((re.escape(f) for f in FILLERS), key=len, reverse=True))
    return len(re.findall(r"\b(?:" + alternation + r")\b", text.lower()))


class TestFiller:
    def test_changes_text_and_reports_count(self):
        rng = random.Random(1)
        res = apply_filler(CLEAN, rng, "easy")
        assert res.text != CLEAN
        assert res.delta["transformation"] == "filler"
        assert res.delta["filler_count"] == filler_count(res.text) >= 1

    def test_count_scales_with_difficulty(self):
        for diff, lo, hi in (("easy", 1, 1), ("medium", 1, 2), ("hard", 2, 4)):
            for i in range(10):
                res = apply_filler(CLEAN, random.Random(i), diff)
                n = filler_count(res.text)
                assert lo <= n <= hi, (diff, res.text)

    def test_never_splits_entity_spans(self):
        for i in range(30):
            res = apply_filler(CLEAN, random.Random(i), "hard")
            for raw in ("3 PM", "Tuesday", "Sarah", "project files"):
                assert raw in res.text, (i, res.text)

    def test_deterministic(self):
        a = apply_filler(CLEAN, random.Random(7), "hard")
        b = apply_filler(CLEAN, random.Random(7), "hard")
        assert a.text == b.text and a.delta == b.delta


class TestRepetition:
    def test_changes_text_and_reports_spans(self):
        for i in range(15):
            res = apply_repetition(CLEAN, random.Random(i), "medium")
            assert res.text != CLEAN
            assert res.delta["transformation"] == "repetition"
            assert res.delta["repeated_spans"], res.text

    def test_never_touches_protected_spans(self):
        text = "Send the project files to Sarah on Tuesday at 3 PM."
        protected = [(27, 32)]  # "Sarah"
        for i in range(20):
            res = apply_repetition(text, random.Random(i), "hard", protected=protected)
            assert res.text.count("Sarah") == 1, res.text
            assert "project files" in res.text or res.text.count("project files") >= 1

    def test_entity_values_survive(self):
        for i in range(20):
            res = apply_repetition(CLEAN, random.Random(i), "hard")
            for raw in ("3 PM", "Tuesday", "Sarah"):
                assert raw in res.text, res.text

    def test_deterministic(self):
        a = apply_repetition(CLEAN, random.Random(5), "hard", protected=[(27, 32)])
        b = apply_repetition(CLEAN, random.Random(5), "hard", protected=[(27, 32)])
        assert a.text == b.text


class TestCorrection:
    def test_abandoned_and_kept_both_in_input(self):
        rng = random.Random(3)
        res = apply_correction(CLEAN, rng, "easy", kinds=("date",))
        corr = res.delta["corrections"][0]
        assert corr["abandoned"] in res.text
        assert corr["kept"] in res.text
        assert corr["abandoned"] != corr["kept"]
        assert corr["kept"] == "Tuesday"
        assert corr["kind"] == "date"

    def test_kept_is_verbatim_clean_span(self):
        # Whatever the span kind, the kept value must be the clean text's value.
        for i in range(20):
            res = apply_correction(CLEAN, random.Random(i), "medium")
            corr = res.delta["corrections"][0]
            assert corr["kept"] in CLEAN, corr

    def test_kind_filter_is_respected(self):
        rng = random.Random(9)
        for i in range(10):
            res = apply_correction(CLEAN, random.Random(i), "easy", kinds=("time",))
            assert res.delta["corrections"][0]["kind"] == "time"

    def test_raises_slot_error_when_kind_absent(self):
        with pytest.raises(SlotError):
            apply_correction("Buy milk and eggs.", random.Random(1), "easy", kinds=("tech",))

    def test_marker_variety(self):
        markers = set()
        for i in range(200):
            res = apply_correction(CLEAN, random.Random(i), "hard")
            markers.add(res.delta["corrections"][0]["marker"])
        assert len(markers) >= 8

    def test_deterministic(self):
        a = apply_correction(CLEAN, random.Random(2), "hard", kinds=("name",))
        b = apply_correction(CLEAN, random.Random(2), "hard", kinds=("name",))
        assert a.text == b.text and a.delta == b.delta

    def test_comma_markers_are_comma_rendered(self):
        # Regression: "butter actually bike lock" needs commas around the marker.
        for marker in ("actually", "I mean", "sorry", "wait", "oh hold on"):
            rendered = _render_correction("butter", "bike lock", marker)
            assert f", {marker}, " in rendered, (marker, rendered)
        assert _render_correction("Tuesday", "Thursday", "no wait") == "Tuesday no wait Thursday"


class TestReformulation:
    def test_restart_mode_duplicates_prefix(self):
        for i in range(10):
            res = apply_reformulation(CLEAN, random.Random(i), "easy")
            ref = res.delta["reformulations"][0]
            if ref["mode"] == "restart":
                assert ref["kept"] == CLEAN
                assert ref["abandoned"] in CLEAN  # abandoned is a prefix of kept
                assert res.text.count(ref["abandoned"]) >= 2

    def test_distract_mode_drops_distractor(self):
        for i in range(30):
            res = apply_reformulation(CLEAN, random.Random(i), "easy")
            ref = res.delta["reformulations"][0]
            if ref["mode"] == "distract":
                assert ref["abandoned"] in res.text
                # kept is the clean output; the input restates it (lowercased first word)
                assert ref["kept"].rstrip(".").lower() in res.text.lower()
                # the distractor core must NOT be part of the kept text
                assert ref["abandoned"] not in ref["kept"]

    def test_both_modes_appear_across_seeds(self):
        modes = set()
        for i in range(60):
            res = apply_reformulation(CLEAN, random.Random(i), "easy")
            modes.add(res.delta["reformulations"][0]["mode"])
        assert modes == {"restart", "distract"}

    def test_deterministic(self):
        a = apply_reformulation(CLEAN, random.Random(4), "hard")
        b = apply_reformulation(CLEAN, random.Random(4), "hard")
        assert a.text == b.text and a.delta == b.delta


class TestFormatting:
    def test_bullet_list_structure_and_content(self):
        rng = random.Random(1)
        out = make_formatting(rng, "bullet_list")
        assert out["type"] == "bullet_list"
        lines = [l for l in out["output"].split("\n") if l.strip()]
        assert 2 <= len(lines) <= 6
        assert all(re.match(r"^(- |\d+\. )", l) for l in lines), out["output"]
        for item in out["items"]:
            assert item in out["input"]

    def test_heading_structure(self):
        rng = random.Random(2)
        out = make_formatting(rng, "heading")
        assert out["type"] == "heading"
        assert out["output"].startswith("# ")
        assert len(out["output"].splitlines()) == 1
        phrase = out["output"][2:]
        assert phrase.lower() in out["input"].lower()

    def test_polite_message_preserves_base_content(self):
        rng = random.Random(3)
        for i in range(20):
            out = make_formatting(random.Random(i), "polite_message")
            assert out["type"] == "polite_message"
            base = out["items"][0]
            # every entity value in the base must survive into the polite output
            for span in entity_spans(base):
                assert span.raw.lower() in out["output"].lower(), (base, out["output"])

    def test_all_types_deterministic(self):
        for kind in ("bullet_list", "heading", "polite_message"):
            a = make_formatting(random.Random(8), kind)
            b = make_formatting(random.Random(8), kind)
            assert a == b

    def test_mind_wrappers_use_gerunds(self):
        # Regression: "Do you mind confirm the slot?" needs "confirming".
        for i in range(100):
            out = make_formatting(random.Random(i), "polite_message")
            if out["output"].startswith(("Do you mind ", "Would you mind ")):
                verb = out["output"].split()[3]
                assert verb.endswith("ing"), out["output"]

    def test_bullet_items_are_capitalized(self):
        # Uniform artifacts: list items start with a capital (matches the
        # hand-written stress-test convention "- Milk").
        for i in range(30):
            out = make_formatting(random.Random(i), "bullet_list")
            for line in out["output"].split("\n"):
                assert re.match(r"^(- |\d+\. )[A-Z0-9]", line), line

    def test_reformulation_strips_any_terminator(self):
        # Robust terminator stripping: questions and exclamations must not
        # collide with the distractor marker ("...?", actually..." is invalid).
        texts = ["Can you send the files to Sarah?",
                 "Call Omar back!",
                 "Send the invoice to Maria."]
        for base in texts:
            for i in range(20):
                res = apply_reformulation(base, random.Random(i), "easy")
                assert re.search(r"[.!?],", res.text) is None, res.text
                assert "??" not in res.text and "!!" not in res.text and ".." not in res.text, res.text

    def test_random_type_is_one_of_three(self):
        for i in range(30):
            out = make_formatting(random.Random(i))
            assert out["type"] in ("bullet_list", "heading", "polite_message")
