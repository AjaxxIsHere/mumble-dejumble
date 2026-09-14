"""Unit tests for training.train._parse_args (no torch required).

Pins the override syntax: values are parsed as JSON, and values that are
not valid JSON are taken literally as strings.
"""

from training.train import _parse_args


class TestParseArgs:
    def test_bare_string_override(self):
        _, overrides = _parse_args(["train.py", "--override=experiment_name:exp001"])
        assert overrides == {"experiment_name": "exp001"}

    def test_json_typed_overrides(self):
        _, overrides = _parse_args([
            "train.py",
            "--override=lora_r:8",
            "--override=learning_rate:2e-4",
            "--override=load_in_4bit:false",
        ])
        assert overrides == {"lora_r": 8, "learning_rate": 2e-4, "load_in_4bit": False}

    def test_quoted_json_string_value(self):
        _, overrides = _parse_args(["train.py", '--override=experiment_name:"exp002"'])
        assert overrides == {"experiment_name": "exp002"}

    def test_config_path(self):
        config_path, overrides = _parse_args([
            "train.py",
            "--config=configs/foo.json",
            "--override=num_train_epochs:5",
        ])
        assert config_path == "configs/foo.json"
        assert overrides == {"num_train_epochs": 5}

    def test_validate_flag(self):
        _, overrides = _parse_args(["train.py", "--validate-config-only"])
        assert overrides == {"__validate_only": True}