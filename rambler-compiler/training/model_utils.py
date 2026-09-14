"""Model / tokenizer utilities for the Qwen3.5-0.8B QLoRA pipeline.

Torch imports are LAZY (inside functions) so this module stays importable
without a GPU — pinned by tests/test_no_torch_imports.py.
"""

from config.training_config import TrainingConfig
from data.common import INSTRUCTION


def setup_tokenizer(model_name: str):
    """Qwen2Tokenizer with explicit pad/eos (do not trust checkpoint defaults)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|im_end|>"
    return tokenizer


def make_formatting_func(tokenizer):
    """SFT formatting_func: system INSTRUCTION + user transcript -> prompt, with
    the chat template and thinking mode explicitly off (non-thinking is the
    default, the flag is belt-and-braces)."""

    def formatting_func(example):
        messages = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": example["input"]},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except (TypeError, ValueError):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt + example["output"] + tokenizer.eos_token

    return formatting_func


def get_response_template(tokenizer):
    """Token ids of the assistant turn opener (drives train_on_responses_only)."""
    return tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)


def inspect_linear_modules(model) -> list[str]:
    """All nn.Linear module names — the diagnostic that makes the novel
    Gated DeltaNet architecture visible before LoRA construction."""
    import torch.nn as nn

    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def _attempt_load(cfg: TrainingConfig, rung: str):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if rung in ("4bit", "8bit"):
        compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(rung == "4bit"),
            load_in_8bit=(rung == "8bit"),
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        return AutoModelForCausalLM.from_pretrained(
            cfg.model_name, quantization_config=bnb_config, device_map="auto"
        )
    dtype = torch.bfloat16 if rung == "bf16" else torch.float16
    return AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype, device_map="auto"
    )


def load_model_with_fallback(cfg: TrainingConfig):
    """The bnb fallback ladder (4-bit -> 8-bit -> fp16/bf16). Each rung is a
    fresh load; the resolved rung is returned so the run's config.json records
    it — a fallback is never silent."""
    ladder = cfg.bnb_fallback_ladder
    if not cfg.load_in_4bit and "4bit" in ladder:
        ladder = tuple(r for r in ladder if r != "4bit")
    errors = []
    for rung in ladder:
        try:
            model = _attempt_load(cfg, rung)
            return model, rung
        except Exception as exc:  # noqa: BLE001 — fallback ladder must catch everything
            errors.append(f"{rung}: {exc!r}")
    raise RuntimeError(
        f"all quantization rungs failed for {cfg.model_name}: " + "; ".join(errors)
    )


def build_lora_config(cfg: TrainingConfig):
    """LoRA over all linear modules minus vision-tower and MTP (nextn) modules,
    so adapters stay text-backbone-only and merge prefix-safe for GGUF."""
    from peft import LoraConfig

    try:
        return LoraConfig(
            task_type="CAUSAL_LM",
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=cfg.lora_target_modules,
            exclude_modules=list(cfg.lora_exclude_modules),
        )
    except TypeError:  # older PEFT without exclude_modules
        return LoraConfig(
            task_type="CAUSAL_LM",
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=cfg.lora_target_modules,
        )


def generate(model, tokenizer, transcript: str, max_new_tokens: int = 256) -> str:
    """Greedy generation with the production chat template (thinking off)."""
    import torch

    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": transcript},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except (TypeError, ValueError):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
