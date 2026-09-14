"""Tests for data.generators.templates — the clean-output template bank."""

import random
import re

import pytest

from data.generators.entities import PLURAL_ITEMS, PoolSampler, entity_spans
from data.generators.templates import (
    ALL_TEMPLATES,
    DOMAINS,
    KNOWN_SLOT_KINDS,
    PLURAL_TECH,
    PRESERVATION_TEMPLATES,
    TEMPLATES,
    Template,
    fill_pattern,
    fill_slots,
    templates_for,
)

SLOT_TO_KIND = {
    "name": "name",
    "name2": "name",
    "item": "item",
    "item_s": "item",
    "item_p": "item",
    "items": "item",
    "location": "location",
    "tech": "tech",
    "tech_s": "tech",
    "tech_p": "tech",
    "number": "number",
    "price": "number",
    "percent": "number",
    "version": "number",
    "duration": "number",
    "port": "number",
    "id": "number",
    "time": "time",
    "date": "date",
}


class TestBank:
    def test_at_least_540_templates_across_12_domains(self):
        assert len(ALL_TEMPLATES) >= 540
        assert len(DOMAINS) == 12

    def test_each_domain_has_at_least_40_templates(self):
        for domain in DOMAINS:
            assert len(TEMPLATES[domain]) >= 40, domain

    def test_ids_are_unique(self):
        ids = [t.id for t in ALL_TEMPLATES]
        assert len(ids) == len(set(ids))

    def test_texts_are_unique(self):
        texts = [t.text for t in ALL_TEMPLATES]
        assert len(texts) == len(set(texts))

    def test_slots_are_known_kinds(self):
        for t in ALL_TEMPLATES:
            assert set(t.slots) <= set(KNOWN_SLOT_KINDS), t.id

    def test_tech_slots_are_always_the_tech(self):
        # Pool entries are bare ("CI pipeline"), so every template must write
        # "the {tech}" — this guards against "the the CI pipeline" double-articles.
        # Sentence-initial "The {tech}" is legitimate, so compare case-insensitively.
        for t in ALL_TEMPLATES:
            for slot in ("{tech}", "{tech_s}"):
                for m in re.finditer(re.escape(slot), t.text):
                    assert t.text[max(0, m.start() - 4) : m.start()].lower() == "the ", t.id

    def test_no_double_articles_after_fill(self):
        rng = random.Random(2)
        for i, t in enumerate(ALL_TEMPLATES):
            filled, _ = fill_slots(t, PoolSampler(random.Random(i)), random.Random(i))
            assert re.search(r"\bthe\s+the\b", filled) is None, t.id
            assert re.search(r"\ba\s+a\b", filled) is None, t.id

    def test_slot_order_matches_text(self):
        # Sequential filling depends on slot kinds appearing in text order.
        for t in ALL_TEMPLATES:
            kinds_in_text = re.findall(r"\{(\w+)\}", t.text)
            assert kinds_in_text == list(t.slots), t.id


class TestFill:
    def test_fill_leaves_no_braces(self):
        for i, t in enumerate(ALL_TEMPLATES):
            filled, _ = fill_slots(t, PoolSampler(random.Random(i)), random.Random(i))
            assert "{" not in filled and "}" not in filled, t.id

    def test_filled_slots_extract_as_entities(self):
        for i, t in enumerate(ALL_TEMPLATES):
            filled, _ = fill_slots(t, PoolSampler(random.Random(i)), random.Random(i))
            kinds = {span.kind for span in entity_spans(filled)}
            for slot in t.slots:
                assert SLOT_TO_KIND[slot] in kinds, f"{t.id}: slot {slot} -> {filled!r}"

    def test_repeat_slots_are_distinct_by_default(self):
        t = Template(
            id="t_range", domain="scheduling",
            text="The meeting runs from {time} to {time}.", slots=("time", "time"),
        )
        for i in range(20):
            filled, _ = fill_slots(t, PoolSampler(random.Random(i)), random.Random(i))
            times = [s.canonical for s in entity_spans(filled) if s.kind == "time"]
            assert len(times) == 2 and times[0] != times[1], filled

    def test_repeat_same_keeps_one_value(self):
        t = Template(
            id="t_same", domain="notes",
            text="Keep {number} as {number} until Friday.", slots=("number", "number"),
        )
        filled, _ = fill_slots(t, PoolSampler(random.Random(1)), random.Random(1), repeat_same=True)
        numbers = [s.canonical for s in entity_spans(filled) if s.kind == "number"]
        assert len(numbers) == 2 and numbers[0] == numbers[1], filled

    def test_mapping_can_fill_patterns(self):
        t = Template(
            id="t_pat", domain="scheduling",
            text="The meeting runs from {time} to {time}.",
            slots=("time", "time"), preserved_patterns=("from {time} to {time}",),
        )
        filled, mapping = fill_slots(t, PoolSampler(random.Random(3)), random.Random(3))
        assert fill_pattern(t.preserved_patterns[0], mapping) in filled

    def test_vocab_coverage_is_broad(self):
        names, items, locations, techs = set(), set(), set(), set()
        for i, t in enumerate(ALL_TEMPLATES * 6):
            filled, _ = fill_slots(t, PoolSampler(random.Random(i)), random.Random(i))
            for span in entity_spans(filled):
                if span.kind == "name":
                    names.add(span.canonical)
                elif span.kind == "item":
                    items.add(span.canonical)
                elif span.kind == "location":
                    locations.add(span.canonical)
                elif span.kind == "tech":
                    techs.add(span.canonical)
        assert len(names) >= 100, len(names)
        assert len(items) >= 100, len(items)
        assert len(locations) >= 50, len(locations)
        assert len(techs) >= 80, len(techs)


class TestAgreement:
    """Subject-verb agreement: {item_s}/{tech_s} slots must never produce plural
    nouns before singular verbs, and {item_p} slots must always produce plurals."""

    def test_item_s_never_plural(self):
        plural = {p.lower() for p in PLURAL_ITEMS}
        templates = [t for t in ALL_TEMPLATES if "item_s" in t.slots]
        assert templates, "no item_s templates found"
        for i in range(200):
            t = templates[i % len(templates)]
            filled, _ = fill_slots(t, PoolSampler(random.Random(i)), random.Random(i))
            for span in entity_spans(filled):
                if span.kind == "item":
                    assert span.canonical not in plural, (t.id, filled)

    def test_item_p_always_plural(self):
        plural = {p.lower() for p in PLURAL_ITEMS}
        templates = [t for t in ALL_TEMPLATES if "item_p" in t.slots]
        assert templates, "no item_p templates found"
        for i in range(100):
            t = templates[i % len(templates)]
            filled, _ = fill_slots(t, PoolSampler(random.Random(i)), random.Random(i))
            item_spans = [s for s in entity_spans(filled) if s.kind == "item"]
            assert item_spans, (t.id, filled)
            for span in item_spans:
                assert span.canonical in plural, (t.id, filled)

    def test_tech_s_never_plural_tech(self):
        plural_tech = {t.lower() for t in PLURAL_TECH}
        templates = [t for t in ALL_TEMPLATES if "tech_s" in t.slots]
        assert templates, "no tech_s templates found"
        for i in range(200):
            t = templates[i % len(templates)]
            filled, _ = fill_slots(t, PoolSampler(random.Random(i)), random.Random(i))
            for span in entity_spans(filled):
                if span.kind == "tech":
                    assert span.canonical not in plural_tech, (t.id, filled)

    def test_name2_is_a_different_person(self):
        templates = [t for t in ALL_TEMPLATES if "name2" in t.slots]
        assert templates, "no name2 templates found"
        for i in range(100):
            t = templates[i % len(templates)]
            filled, _ = fill_slots(t, PoolSampler(random.Random(i)), random.Random(i))
            names = [s.canonical for s in entity_spans(filled) if s.kind == "name"]
            assert len(names) == 2, (t.id, filled)
            assert names[0] != names[1], (t.id, filled)

    def test_bank_has_grown(self):
        # The 50+ template expansion requirement.
        assert len(ALL_TEMPLATES) >= 590, len(ALL_TEMPLATES)


class TestTemplatesFor:
    def test_required_kinds_filter(self):
        filtered = templates_for("work", required_kinds={"date"})
        assert len(filtered) > 5
        for t in filtered:
            assert "date" in t.slots

    def test_unknown_domain_raises(self):
        with pytest.raises(KeyError):
            templates_for("nonsense")


class TestPreservationTemplates:
    def test_at_least_10_with_valid_patterns(self):
        assert len(PRESERVATION_TEMPLATES) >= 10
        for t in PRESERVATION_TEMPLATES:
            assert t.preserved_patterns, t.id
            for pattern in t.preserved_patterns:
                for m in re.finditer(r"\{(\w+)\}", pattern):
                    assert m.group(1) in t.slots, f"{t.id}: {pattern!r}"

    def test_patterns_appear_verbatim_in_filled_text(self):
        for i, t in enumerate(PRESERVATION_TEMPLATES):
            filled, mapping = fill_slots(
                t, PoolSampler(random.Random(i)), random.Random(i), repeat_same=t.repeat_same
            )
            for pattern in t.preserved_patterns:
                assert fill_pattern(pattern, mapping) in filled, f"{t.id}: {pattern!r} -> {filled!r}"
