"""Tests for config.training_config — loads/validates without torch installed."""

import pytest

from config.training_config import TrainingConfig, asdict_spec, load_config


class TestDefaults:
    def test_defaults_are_the_experiment_2_recipe(self):
        cfg = TrainingConfig()
        assert cfg.model_name == "Qwen/Qwen3.5-0.8B"
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.learning_rate == 2e-4
        assert cfg.max_length == 1024
        assert cfg.num_train_epochs == 3
        assert cfg.per_device_train_batch_size == 4
        assert cfg.gradient_accumulation_steps == 8
        assert cfg.warmup_ratio == 0.1
        assert cfg.seed == 42

    def test_defaults_validate(self):
        TrainingConfig().validate()


class TestValidate:
    def test_rejects_bad_learning_rate(self):
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0.0).validate()

    def test_rejects_bad_batch_size(self):
        with pytest.raises(ValueError):
            TrainingConfig(per_device_train_batch_size=0).validate()

    def test_rejects_unknown_ladder_rung(self):
        with pytest.raises(ValueError):
            TrainingConfig(bnb_fallback_ladder=("4bit", "nonsense")).validate()

    def test_rejects_bad_dataset_version(self):
        with pytest.raises(ValueError):
            TrainingConfig(dataset_version="not-a-version").validate()


class TestLoadConfig:
    def test_load_from_file_and_overrides(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text('{"learning_rate": 1e-4, "num_train_epochs": 2}')
        cfg = load_config(path=p, overrides={"lora_r": 8})
        assert cfg.learning_rate == 1e-4
        assert cfg.num_train_epochs == 2
        assert cfg.lora_r == 8
        assert cfg.model_name == "Qwen/Qwen3.5-0.8B"

    def test_unknown_override_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            load_config(overrides={"not_a_field": 1})

    def test_unknown_json_field_rejected(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text('{"not_a_field": 1}')
        with pytest.raises(ValueError):
            load_config(path=p)


class TestSpecNames:
    def test_spec_names_mapping(self):
        d = asdict_spec(TrainingConfig())
        assert d["MODEL_NAME"] == "Qwen/Qwen3.5-0.8B"
        assert d["MAX_SEQ_LENGTH"] == 1024
        assert d["LEARNING_RATE"] == 2e-4
        assert d["NUM_EPOCHS"] == 3
        assert d["BATCH_SIZE"] == 4
        assert d["GRADIENT_ACCUMULATION"] == 8
        assert d["LORA_R"] == 16
        assert d["LORA_ALPHA"] == 32
        assert d["LORA_DROPOUT"] == 0.05
        assert d["WARMUP_RATIO"] == 0.1
        assert d["WEIGHT_DECAY"] == 0.01
