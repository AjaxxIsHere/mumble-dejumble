"""Tests for the transformation recipe system."""

import json
import pytest

from data.augmentation.recipes import (
    Recipe,
    Transformation,
    TransformationType,
    make_noop_recipe,
    make_filler_recipe,
    make_repetition_recipe,
    make_correction_recipe,
    make_compositional_recipe,
    validate_composition,
    recipe_summary,
)


class TestRecipeSerialization:
    """Tests for recipe JSON round-trip."""

    def test_noop_recipe_roundtrip(self):
        recipe = make_noop_recipe()
        json_str = recipe.to_json()
        restored = Recipe.from_json(json_str)
        assert len(restored.transformations) == 0
        assert restored.composition_label == "noop"

    def test_filler_recipe_roundtrip(self):
        recipe = make_filler_recipe(location="sentence_start")
        json_str = recipe.to_json()
        restored = Recipe.from_json(json_str)
        assert len(restored.transformations) == 1
        assert restored.transformations[0].type == "filler"
        assert restored.transformations[0].location == "sentence_start"

    def test_correction_recipe_roundtrip(self):
        recipe = make_correction_recipe("Tuesday", "Thursday", "date", "no wait")
        json_str = recipe.to_json()
        restored = Recipe.from_json(json_str)
        assert restored.transformations[0].abandoned == "Tuesday"
        assert restored.transformations[0].retained == "Thursday"

    def test_compositional_recipe_roundtrip(self):
        recipe = make_compositional_recipe(["filler", "repetition"])
        json_str = recipe.to_json()
        restored = Recipe.from_json(json_str)
        types = {t.type for t in restored.transformations}
        assert "filler" in types
        assert "repetition" in types


class TestRecipeComposition:
    """Tests for composition labels and validation."""

    def test_noop_composition(self):
        recipe = make_noop_recipe()
        assert recipe.composition_label == "noop"

    def test_single_filler(self):
        recipe = make_filler_recipe()
        assert recipe.composition_label == "single"

    def test_filler_plus_repetition(self):
        recipe = make_compositional_recipe(["filler", "repetition"])
        assert recipe.composition_label == "filler+rep"

    def test_filler_plus_correction(self):
        recipe = make_compositional_recipe(
            ["filler", "self_correction"],
            corrections=[("Tuesday", "Thursday", "date")],
        )
        assert recipe.composition_label == "filler+corr"

    def test_triple_composition(self):
        recipe = make_compositional_recipe(
            ["filler", "repetition", "self_correction"],
            corrections=[("Tuesday", "Thursday", "date")],
        )
        assert recipe.composition_label == "filler+rep+corr"

    def test_validate_valid_composition(self):
        recipe = make_filler_recipe()
        is_valid, label = validate_composition(recipe)
        assert is_valid
        assert label == "single"

    def test_validate_noop(self):
        recipe = make_noop_recipe()
        is_valid, label = validate_composition(recipe)
        assert is_valid
        assert label == "noop"


class TestRecipeSummary:
    """Tests for human-readable recipe summaries."""

    def test_noop_summary(self):
        recipe = make_noop_recipe()
        assert "noop" in recipe_summary(recipe)

    def test_filler_summary(self):
        recipe = make_filler_recipe(location="sentence_start")
        summary = recipe_summary(recipe)
        assert "filler" in summary
        assert "sentence_start" in summary

    def test_correction_summary(self):
        recipe = make_correction_recipe("Tuesday", "Thursday", "date")
        summary = recipe_summary(recipe)
        assert "Tuesday" in summary
        assert "Thursday" in summary
        assert "→" in summary


class TestTransformationTypes:
    """Tests for transformation type enum."""

    def test_all_types_exist(self):
        assert TransformationType.FILLER.value == "filler"
        assert TransformationType.REPETITION.value == "repetition"
        assert TransformationType.SELF_CORRECTION.value == "self_correction"
        assert TransformationType.FORMATTING_COMMAND.value == "formatting_command"

    def test_transformation_dataclass(self):
        t = Transformation(type="filler", location="mid_sentence")
        assert t.type == "filler"
        assert t.location == "mid_sentence"
        assert t.count == 1
