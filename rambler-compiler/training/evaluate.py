"""Inference-only evaluation: model -> greedy predictions.jsonl.

Rows are the input records plus an "actual" field; scoring happens on the
CPU-only machine via evaluation.metrics / evaluation.failure_analysis /
evaluation.stress_test (the torch-free tier).

Usage:
  python -m training.evaluate <records.jsonl> --out predictions.jsonl
      [--adapter outputs/experiments/exp001/adapter]
      [--max-new-tokens 256]
"""

import json
import sys
from pathlib import Path

from config.training_config import load_config
from training.model_utils import generate, load_model_with_fallback, setup_tokenizer


def main(argv) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    records_path = args[0] if args else None
    out_arg = next((a.split("=", 1)[1] for a in argv[1:] if a.startswith("--out=")), None)
    adapter = next((a.split("=", 1)[1] for a in argv[1:] if a.startswith("--adapter=")), None)
    max_new = int(next(
        (a.split("=", 1)[1] for a in argv[1:] if a.startswith("--max-new-tokens=")), "256"))
    if not records_path or not out_arg:
        print("usage: python -m training.evaluate <records.jsonl> --out=pred.jsonl "
              "[--adapter=dir] [--max-new-tokens=N]", file=sys.stderr)
        return 2

    cfg = load_config()
    tokenizer = setup_tokenizer(cfg.model_name)
    model, resolved = load_model_with_fallback(cfg)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"loaded adapter from {adapter}")

    with open(records_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    out_path = Path(out_arg)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            actual = generate(model, tokenizer, record["input"], max_new)
            f.write(json.dumps({**record, "actual": actual}, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(records)}")
    print(f"wrote {out_path} ({len(records)} predictions, quantization={resolved})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
