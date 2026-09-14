"""Tests for data.generators.entities — vocab pools, span extraction,
same-kind variants, and the without-replacement pool sampler."""

import random
import re

import pytest

from data.generators.entities import (
    ENTITY_KINDS,
    FIRST_NAMES,
    ITEMS,
    LAST_NAMES,
    LOCATIONS,
    PLURAL_ITEMS,
    SINGULAR_ITEMS,
    SlotError,
    TECH_TERMS,
    WEEKDAYS,
    EntitySpan,
    PoolSampler,
    entity_spans,
    entity_variant,
    extract_entities,
    make_date,
    make_duration,
    make_id,
    make_name,
    make_number,
    make_port,
    make_time,
    pick_entity_span,
)


def kind_of(text: str) -> str | None:
    spans = entity_spans(text)
    return spans[0].kind if spans else None


class TestPools:
    def test_pool_sizes_meet_minimums(self):
        assert len(FIRST_NAMES) >= 150
        assert len(LAST_NAMES) >= 50
        assert len(ITEMS) >= 180
        assert len(LOCATIONS) >= 90
        assert len(TECH_TERMS) >= 110

    @pytest.mark.parametrize("pool", [FIRST_NAMES, LAST_NAMES, ITEMS, LOCATIONS, TECH_TERMS])
    def test_pools_are_unique(self, pool):
        assert len(pool) == len(set(pool))

    @pytest.mark.parametrize("pool", [FIRST_NAMES, LAST_NAMES, ITEMS, LOCATIONS, TECH_TERMS])
    def test_pools_have_unique_canonical_forms(self, pool):
        # No two entries may normalize to the same string (dedup/canonical safety).
        canonicals = {p.lower() for p in pool}
        assert len(canonicals) == len(pool)

    def test_weekdays_present(self):
        assert len(WEEKDAYS) == 7

    def test_no_cross_pool_embedding(self):
        # No location entry may contain an item/tech entry as a whole-word
        # sequence (or vice versa): extraction resolves overlaps longest-first,
        # so a shorter embedded entry would be permanently shadowed
        # (e.g. "the library" vs "library"). Non-word substring coincidences
        # like "Rome" inside "Prometheus" are harmless (case + word boundaries).
        for a in LOCATIONS:
            for b in ITEMS + TECH_TERMS:
                assert re.search(r"(?<![A-Za-z])" + re.escape(b) + r"(?![A-Za-z])", a, re.IGNORECASE) is None, (a, b)
                assert re.search(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])", b, re.IGNORECASE) is None, (a, b)


class TestMakers:
    def test_make_number_produces_a_number(self):
        rng = random.Random(7)
        for _ in range(30):
            assert kind_of(make_number(rng)) == "number"

    def test_generic_make_number_stays_small(self):
        # Regression: port-style 4-5 digit values ("60944 minutes late") must
        # never leak into generic number slots.
        for i in range(100):
            v = make_number(random.Random(i))
            if v.isdigit():
                assert 1 <= int(v) <= 999, v

    def test_make_time_produces_a_time(self):
        rng = random.Random(7)
        for _ in range(30):
            assert kind_of(make_time(rng)) == "time"

    def test_make_date_produces_a_date(self):
        rng = random.Random(7)
        for _ in range(30):
            assert kind_of(make_date(rng)) == "date"

    def test_make_name_produces_a_name(self):
        rng = random.Random(7)
        for _ in range(30):
            assert kind_of(make_name(rng)) == "name"

    def test_makers_have_variety(self):
        rng = random.Random(11)
        times = {make_time(rng) for _ in range(60)}
        dates = {make_date(rng) for _ in range(60)}
        assert len(times) > 15
        assert len(dates) > 15


class TestSpecializedMakers:
    def test_make_duration_is_realistic(self):
        for i in range(50):
            v = make_duration(random.Random(i))
            m = re.fullmatch(r"(\d+) minutes", v)
            assert m and int(m.group(1)) in {5, 10, 15, 20, 30, 45, 60, 90}, v
            assert kind_of(v) == "number"

    def test_make_port_is_four_to_five_digit_port_range(self):
        for i in range(50):
            v = make_port(random.Random(i))
            assert re.fullmatch(r"\d{4,5}", v), v
            assert 1024 <= int(v) <= 65535, v

    def test_make_id_is_four_to_five_digits(self):
        for i in range(50):
            v = make_id(random.Random(i))
            assert re.fullmatch(r"\d{4,5}", v), v
            assert 1000 <= int(v) <= 99999, v


class TestItemPartition:
    def test_items_is_exact_partition(self):
        assert set(ITEMS) == set(SINGULAR_ITEMS) | set(PLURAL_ITEMS)
        assert not (set(SINGULAR_ITEMS) & set(PLURAL_ITEMS))

    def test_plural_items_look_plural(self):
        assert all(p.lower().endswith("s") for p in PLURAL_ITEMS)
        assert "scissors" in PLURAL_ITEMS
        assert "laptop" in SINGULAR_ITEMS
        assert "cooler" in SINGULAR_ITEMS

    def test_number_variant_of_large_number_stays_large(self):
        # Corrections on ports stay port-like: "8080 no 3000", not "8080 no 42".
        rng = random.Random(3)
        for i in range(20):
            v = entity_variant(entity_spans("8080")[0], random.Random(i))
            assert re.fullmatch(r"\d{4,5}", v), v
            assert 1000 <= int(v) <= 65535, v

    def test_number_variant_of_five_digit_id_terminates(self):
        # Regression: variants of 5-digit ids (up to 99999) must return —
        # an upper bound of 65535 made this loop forever.
        for i in range(10):
            v = entity_variant(entity_spans("90234")[0], random.Random(i))
            assert re.fullmatch(r"\d{4,5}", v), v
            assert 1000 <= int(v) <= 99999, v


class TestEntitySpans:
    def test_finds_known_entities_with_kinds(self):
        text = "Send the report to Sarah Chen on Tuesday at 3 PM from Dubai, port 8080."
        spans = entity_spans(text)
        kinds = {s.kind: s.raw for s in spans}
        assert kinds["name"] == "Sarah Chen"
        assert kinds["date"] == "Tuesday"
        assert kinds["time"] == "3 PM"
        assert kinds["location"] == "Dubai"
        assert kinds["number"] == "8080"

    def test_finds_items_and_tech(self):
        text = "Buy milk and bread for the Kubernetes deployment."
        spans = entity_spans(text)
        assert any(s.kind == "item" and s.raw == "milk" for s in spans)
        assert any(s.kind == "tech" and s.raw == "Kubernetes" for s in spans)

    def test_spans_do_not_overlap(self):
        text = "Meet Sarah Chen at 3 PM on March 4th in Dubai."
        spans = entity_spans(text)
        for i, a in enumerate(spans):
            for b in spans[i + 1:]:
                assert a.end <= b.start or b.end <= a.start

    def test_canonical_is_normalized_raw(self):
        spans = entity_spans("Ship to DUBAI by Tuesday.")
        assert any(s.kind == "location" and s.canonical == "dubai" for s in spans)

    def test_extract_entities_returns_canonical_set(self):
        text = "Send the file to Sarah tomorrow at 4:30 PM."
        ents = extract_entities(text)
        assert "sarah" in ents
        assert "tomorrow" in ents
        assert "4:30 pm" in ents


class TestEntityVariant:
    @pytest.mark.parametrize("kind", ENTITY_KINDS)
    def test_variant_never_equals_raw(self, kind):
        rng = random.Random(3)
        # Build a span for each kind from a guaranteed source value.
        source = {
            "name": "Sarah Chen",
            "item": "milk",
            "location": "Dubai",
            "tech": "Kubernetes",
            "number": "42",
            "time": "3 PM",
            "date": "Tuesday",
        }[kind]
        spans = entity_spans(source)
        assert spans, f"source {source!r} should extract as {kind}"
        for _ in range(15):
            variant = entity_variant(spans[0], rng)
            assert variant.lower() != spans[0].raw.lower()

    @pytest.mark.parametrize("kind", ENTITY_KINDS)
    def test_variant_is_same_kind(self, kind):
        rng = random.Random(5)
        source = {
            "name": "Sarah Chen",
            "item": "milk",
            "location": "Dubai",
            "tech": "Kubernetes",
            "number": "42",
            "time": "3 PM",
            "date": "Tuesday",
        }[kind]
        span = entity_spans(source)[0]
        for _ in range(10):
            variant = entity_variant(span, rng)
            assert kind_of(variant) == kind, f"{variant!r} should extract as {kind}"

    def test_number_format_family_preserved(self):
        rng = random.Random(9)
        assert re.fullmatch(r"\d+", entity_variant(entity_spans("42")[0], rng))
        assert re.fullmatch(r"\$\d+\.\d+", entity_variant(entity_spans("$12.50")[0], rng))

    def test_time_format_family_preserved(self):
        rng = random.Random(9)
        assert re.fullmatch(r"\d+ (am|pm)", entity_variant(entity_spans("3 PM")[0], rng), re.IGNORECASE)
        assert re.fullmatch(
            r"\d+:\d+ (am|pm)", entity_variant(entity_spans("4:30 PM")[0], rng), re.IGNORECASE
        )
        assert entity_variant(entity_spans("noon")[0], rng) == "midnight"

    def test_date_family_preserved(self):
        rng = random.Random(9)
        assert entity_variant(entity_spans("Tuesday")[0], rng) in WEEKDAYS
        assert entity_variant(entity_spans("tomorrow")[0], rng) in {"today", "yesterday"}
        assert re.fullmatch(
            r"march \d+(st|nd|rd|th)?", entity_variant(entity_spans("March 4th")[0], rng), re.IGNORECASE
        )

    def test_name_surname_preserved(self):
        rng = random.Random(9)
        variant = entity_variant(entity_spans("Sarah Chen")[0], rng)
        assert variant.endswith(" Chen")
        assert not variant.startswith("Sarah")


class TestPickEntitySpan:
    def test_raises_slot_error_when_kind_missing(self):
        with pytest.raises(SlotError):
            pick_entity_span("Buy milk and eggs", random.Random(1), kinds=("date",))

    def test_returns_requested_kind(self):
        span = pick_entity_span("Send it on Tuesday please", random.Random(1), kinds=("date", "time"))
        assert span.kind == "date"

    def test_returns_span_with_valid_offsets(self):
        text = "Send it to Sarah tomorrow"
        span = pick_entity_span(text, random.Random(1), kinds=("name",))
        assert text[span.start:span.end] == span.raw
        assert isinstance(span, EntitySpan)


class TestPoolSampler:
    def test_no_repeats_within_session(self):
        sampler = PoolSampler(random.Random(4))
        draws = [sampler.sample("item") for _ in range(15)]
        assert len(draws) == len(set(draws))

    def test_raises_when_pool_exhausted(self):
        sampler = PoolSampler(random.Random(4))
        pool = sampler.pools["item"]
        with pytest.raises(SlotError):
            for _ in range(len(pool) + 1):
                sampler.sample("item")

    def test_sessions_are_independent(self):
        a, b = PoolSampler(random.Random(1)), PoolSampler(random.Random(1))
        assert [a.sample("item") for _ in range(10)] == [b.sample("item") for _ in range(10)]

    def test_unknown_pool_raises(self):
        with pytest.raises(KeyError):
            PoolSampler(random.Random(1)).sample("nonsense")

    def test_item_s_only_singular(self):
        sampler = PoolSampler(random.Random(1))
        for _ in range(30):
            assert sampler.sample("item_s") in SINGULAR_ITEMS

    def test_item_p_only_plural(self):
        sampler = PoolSampler(random.Random(1))
        for _ in range(30):
            assert sampler.sample("item_p") in PLURAL_ITEMS

    def test_no_cross_kind_item_repeat(self):
        # One record never repeats an item across item/item_s/item_p slots.
        sampler = PoolSampler(random.Random(2))
        seen = {sampler.sample("item").lower()}
        for _ in range(20):
            for kind in ("item_s", "item_p", "item"):
                value = sampler.sample(kind)
                assert value.lower() not in seen, (kind, value)
                seen.add(value.lower())
