"""Tests for data.generators.longform — realistic 50-300 word monologues."""

import random
import re

from data.generators.entities import entity_spans
from data.generators.longform import compose_longform

BUCKETS = {"easy": (45, 95), "medium": (80, 165), "hard": (140, 330)}


def word_count(text: str) -> int:
    return len(text.split())


class TestComposeLongform:
    def test_word_counts_land_in_buckets(self):
        for difficulty, (lo, hi) in BUCKETS.items():
            for i in range(15):
                result = compose_longform("work", random.Random(i), difficulty)
                wc = word_count(result.text)
                assert lo <= wc <= hi, (difficulty, wc, result.text[:120])

    def test_offsets_reference_unit_text(self):
        result = compose_longform("travel", random.Random(3), "medium")
        for unit in result.units:
            assert result.text[unit["offsets"][0]:unit["offsets"][1]] == unit["text"], unit

    def test_units_cover_the_whole_text(self):
        result = compose_longform("shopping", random.Random(4), "hard")
        covered = [u["offsets"] for u in result.units]
        covered.sort()
        assert covered[0][0] == 0
        assert covered[-1][1] == len(result.text)

    def test_no_repeated_connectives(self):
        for i in range(10):
            result = compose_longform("work", random.Random(i), "hard")
            for unit in result.units[1:]:
                connective = unit["connective"]
                if connective:
                    count = sum(1 for u in result.units[1:] if u["connective"] == connective)
                    assert count == 1, (connective, result.text)

    def test_entities_are_present(self):
        for i in range(10):
            result = compose_longform("work", random.Random(i), "medium")
            kinds = {s.kind for s in entity_spans(result.text)}
            assert len(kinds) >= 2, (kinds, result.text[:120])

    def test_deterministic(self):
        a = compose_longform("work", random.Random(9), "hard")
        b = compose_longform("work", random.Random(9), "hard")
        assert a.text == b.text
        assert a.units == b.units

    def test_extensions_never_follow_a_period(self):
        # Regression: extensions used to attach AFTER the sentence terminator
        # ("...to the inventory., even though Austin is far").
        for i in range(10):
            text = compose_longform("work", random.Random(i), "hard").text
            assert re.search(r"\w\.\s*,", text) is None, text

    def test_extensions_respect_domain(self):
        # Technical clauses must not attach to non-technical monologues and
        # vice versa (no non-sequitur entity combinations).
        for i in range(15):
            technical = compose_longform("technical", random.Random(i), "hard").text
            assert "weather in" not in technical, technical
            assert "before {location}" not in technical
            everyday = compose_longform("everyday", random.Random(i), "hard").text
            assert "goes down" not in everyday, everyday
            assert "keeps failing" not in everyday, everyday
            travel = compose_longform("travel", random.Random(i), "hard").text
            assert "goes down" not in travel, travel

    def test_no_plural_verb_disagreement(self):
        # "so the {item} stays fresh" was plural-unsafe; the replacement
        # predicate must agree with any item, plural or singular.
        from data.generators.entities import PLURAL_ITEMS

        plural = {p.lower() for p in PLURAL_ITEMS}
        for i in range(15):
            text = compose_longform("shopping", random.Random(i), "hard").text
            assert re.search(r"\bstays?\b", text) is None, text

    def test_no_comma_after_conjunct_connectives(self):
        # "Oh and add paint..." reads better than "Oh and, add paint..."
        from data.generators.longform import NO_COMMA_CONNECTIVES

        for i in range(10):
            result = compose_longform("work", random.Random(i), "hard")
            for unit in result.units:
                conn = unit["connective"]
                if conn and conn in NO_COMMA_CONNECTIVES:
                    assert unit["text"].startswith(conn.capitalize() + " "), unit["text"]
