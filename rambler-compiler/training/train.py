"""QLoRA SFT training driver (runs on Colab).

Flow: config -> tokenizer -> model (bnb fallback ladder) -> LoRA ->
SFTTrainer with a MetricEvalCallback (real greedy-generation metrics during
training, not just loss) -> adapter + artifacts in outputs/experiments/<name>/.

Usage: python -m training.train [--config path.json] [--override key:value ...]

Override values are parsed as JSON (so 8 -> int, 2e-4 -> float, true -> bool);
values that are not valid JSON are taken literally as strings
(e.g. --override=experiment_name:exp001).
"""

import json
import random
import sys
from pathlib import Path

from config.training_config import TrainingConfig, asdict_spec, load_config
from data.common import INSTRUCTION, write_json
from training.model_utils import (
    build_lora_config,
    generate,
    get_response_template,
    inspect_linear_modules,
    load_model_with_fallback,
    make_formatting_func,
    setup_tokenizer,
)


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _env_block() -> dict:
    import platform

    import torch
    import transformers
    import peft
    import trl

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "trl": trl.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
    }


class MetricEvalCallback:
    """On every evaluation step: sample a deterministic slice of the validation
    set, greedy-generate, and log real metrics (exact match, correction
    resolution) — the Experiment 1 blind spot was watching only loss."""

    def __init__(self, eval_records, tokenizer, batch: int, max_new_tokens: int,
                 seed: int, out_dir: Path):
        self.eval_records = eval_records
        self.tokenizer = tokenizer
        self.batch = batch
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.out_dir = out_dir
        self.history: list[dict] = []

    def __call__(self, trainer, args, state, control, **kwargs):
        model = kwargs.get("model") or trainer.model
        rng = random.Random(self.seed + state.global_step)
        sample = rng.sample(self.eval_records, min(self.batch, len(self.eval_records)))

        import evaluation.metrics as M

        scored = []
        for rec in sample:
            actual = generate(model, self.tokenizer, rec["input"], self.max_new_tokens)
            metrics = M.score_record(rec, actual)
            scored.append({"id": rec.get("id"), "actual": actual, **metrics})
        exact = sum(r["exact_match"] for r in scored) / len(scored)
        corr = sum(r["correction_resolution"] for r in scored) / len(scored)
        ent = sum(r["entity_preservation"] for r in scored) / len(scored)

        state.log_history[-1]["eval_exact"] = exact
        state.log_history[-1]["eval_corr_resolution"] = corr
        state.log_history[-1]["eval_entity_preservation"] = ent
        self.history.append({
            "step": state.global_step, "eval_exact": exact,
            "eval_corr_resolution": corr, "eval_entity_preservation": ent,
        })
        return control


def _parse_args(argv):
    config_path, overrides = None, {}
    for arg in argv[1:]:
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
        elif arg.startswith("--override="):
            key, value = arg.split("=", 1)[1].split(":", 1)
            try:
                overrides[key] = json.loads(value)
            except json.JSONDecodeError:
                overrides[key] = value  # bare string override, e.g. exp001
        elif arg == "--validate-config-only":
            overrides["__validate_only"] = True  # handled below
    return config_path, overrides


def main(argv) -> int:
    config_path, overrides = _parse_args(argv)
    validate_only = overrides.pop("__validate_only", False)
    cfg = load_config(path=config_path, overrides=overrides or None)
    if validate_only:
        print("config OK:", cfg.experiment_name)
        return 0

    out_dir = Path(cfg.output_root) / "experiments" / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. tokenizer + model (ladder) + LoRA
    tokenizer = setup_tokenizer(cfg.model_name)
    model, resolved_quantization = load_model_with_fallback(cfg)
    linear_names = inspect_linear_modules(model)
    lora_config = build_lora_config(cfg)

    resolved = asdict_spec(cfg)
    resolved["RESOLVED_QUANTIZATION"] = resolved_quantization
    resolved["ENV"] = _env_block()
    resolved["LINEAR_MODULE_COUNT"] = len(linear_names)
    write_json(out_dir / "config.json", resolved)

    # 2. data
    train_records = _load_jsonl(cfg.train_file)
    eval_records = _load_jsonl(cfg.validation_file)

    # 3. trainer
    from trl import SFTConfig, SFTTrainer

    packing = cfg.packing and not cfg.train_on_responses_only  # TRL: mutually exclusive
    args = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        max_length=cfg.max_length,
        packing=packing,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        optim=cfg.optim,
        max_grad_norm=cfg.max_grad_norm,
        num_train_epochs=cfg.num_train_epochs,
        gradient_checkpointing=cfg.gradient_checkpointing,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        report_to="none",
        seed=cfg.seed,
    )

    metric_callback = MetricEvalCallback(
        eval_records, tokenizer, cfg.eval_metric_batch, cfg.eval_max_new_tokens,
        cfg.seed, out_dir,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_records,
        eval_dataset=eval_records,
        processing_class=tokenizer,
        formatting_func=make_formatting_func(tokenizer),
        peft_config=lora_config,
        train_on_responses_only=cfg.train_on_responses_only,
        response_template=get_response_template(tokenizer) if cfg.train_on_responses_only else None,
        callbacks=[metric_callback],
    )

    # 4. train + save
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    # 5. training metrics + overfit watch
    log_history = list(trainer.state.log_history)
    write_json(out_dir / "training_metrics.json", {
        "log_history": log_history,
        "metric_evals": metric_callback.history,
    })
    exacts = [h["eval_exact"] for h in metric_callback.history]
    overfit_warning = None
    if len(exacts) >= 3:
        peak = max(exacts)
        if exacts[-1] < peak - 0.05:
            overfit_warning = (
                f"eval_exact peaked at {peak:.3f} then fell to {exacts[-1]:.3f} "
                "while training continued — possible overfitting"
            )
    readme = out_dir / "README.md"
    readme.write_text(
        f"# {cfg.experiment_name}\n\n"
        f"model: {cfg.model_name}\ndataset_version: {cfg.dataset_version}\n"
        f"quantization: {resolved_quantization}\n"
        f"train records: {len(train_records)}\n"
        f"overfit_warning: {overfit_warning or 'none'}\n"
    )
    if overfit_warning:
        print("WARNING:", overfit_warning)
    print(f"done: {out_dir} (quantization={resolved_quantization})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
