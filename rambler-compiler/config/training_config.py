"""Training configuration — the single, clearly-defined configuration block
(spec §16). Loads and validates WITHOUT torch (works on the CPU laptop and in
Colab alike).

Field names map to the spec §16 identifiers (see asdict_spec): e.g.
max_seq_length -> MAX_SEQ_LENGTH, learning_rate -> LEARNING_RATE.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from data.common import parse_dataset_version

_SPEC_NAMES = {
    "model_name": "MODEL_NAME",
    "output_root": "OUTPUT_DIR",
    "max_seq_length": "MAX_SEQ_LENGTH",
    "learning_rate": "LEARNING_RATE",
    "num_train_epochs": "NUM_EPOCHS",
    "per_device_train_batch_size": "BATCH_SIZE",
    "gradient_accumulation_steps": "GRADIENT_ACCUMULATION",
    "lora_r": "LORA_R",
    "lora_alpha": "LORA_ALPHA",
    "lora_dropout": "LORA_DROPOUT",
    "warmup_ratio": "WARMUP_RATIO",
    "weight_decay": "WEIGHT_DECAY",
}


@dataclass
class TrainingConfig:
    # model / data (the V4 hybrid dataset is the default training source)
    model_name: str = "Qwen/Qwen3.5-0.8B"
    train_file: str = "data/splits/20260904.1/train.jsonl"
    validation_file: str = "data/splits/20260904.1/validation.jsonl"
    dataset_version: str = "20260904.1"
    output_root: str = "outputs"
    experiment_name: str = "exp001"

    # QLoRA / LoRA
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_fallback_ladder: tuple[str, ...] = ("4bit", "8bit", "fp16")
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    lora_exclude_modules: tuple[str, ...] = (".*vision.*", ".*visual.*", ".*nextn.*")

    # optimizer / schedule
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 1.0

    # batching / hardware (T4: fp16, bf16 is slow on sm_75)
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_length: int = 1024
    packing: bool = True
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    num_train_epochs: int = 3

    # eval / checkpointing
    train_on_responses_only: bool = True
    eval_steps: int = 100
    eval_metric_batch: int = 32
    eval_max_new_tokens: int = 256
    save_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 10
    resume_from_checkpoint: bool = False

    seed: int = 42

    def validate(self) -> None:
        if not (0.0 < self.learning_rate <= 1e-2):
            raise ValueError(f"learning_rate {self.learning_rate} out of range")
        if self.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.max_length < 64:
            raise ValueError("max_length must be >= 64")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.lora_r < 1 or self.lora_alpha < 1:
            raise ValueError("lora_r / lora_alpha must be >= 1")
        if not (0.0 <= self.warmup_ratio < 1.0):
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.num_train_epochs < 1:
            raise ValueError("num_train_epochs must be >= 1")
        allowed_ladder = {"4bit", "8bit", "fp16", "bf16"}
        if not set(self.bnb_fallback_ladder) <= allowed_ladder or not self.bnb_fallback_ladder:
            raise ValueError(f"fallback ladder {self.bnb_fallback_ladder} invalid")
        parse_dataset_version(self.dataset_version)


def load_config(
    path: str | Path | None = None,
    overrides: dict | None = None,
) -> TrainingConfig:
    """Load defaults, apply a JSON file, then CLI overrides (unknown keys rejected)."""
    values: dict = {}
    if path is not None:
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(TrainingConfig)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        values.update(raw)
    if overrides:
        known = {f.name for f in fields(TrainingConfig)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown overrides: {sorted(unknown)}")
        values.update(overrides)
    cfg = TrainingConfig(**values)
    cfg.validate()
    return cfg


def asdict_spec(cfg: TrainingConfig) -> dict:
    """Resolved config with spec §16-style keys (for config.json artifacts)."""
    d = asdict(cfg)
    out = {}
    for key, value in d.items():
        out[_SPEC_NAMES.get(key, key.upper())] = value
    return out
