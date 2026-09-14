"""Export: merge the LoRA adapter -> fp16 safetensors -> GGUF (llama.cpp)
f16 / Q8_0 / Q4_K_M -> optional 3-prompt llama-cli smoke test.

Qwen3.5 is natively supported by llama.cpp (LLM_ARCH_QWEN35); the merged
checkpoint is the ORIGINAL Qwen/Qwen3.5-0.8B + adapter, so the official
convert_hf_to_gguf.py path applies (community text-only checkpoints are not
llama.cpp-supported for export — that's why training uses the original).

Usage:
  python -m training.export --adapter outputs/experiments/exp001/adapter \
      --llama-cpp-dir /content/llama.cpp [--smoke]
"""

import subprocess
import sys
from pathlib import Path

from config.training_config import load_config
from training.model_utils import load_model_with_fallback, setup_tokenizer


def main(argv) -> int:
    adapter = next((a.split("=", 1)[1] for a in argv[1:] if a.startswith("--adapter=")), None)
    llama_dir = next((a.split("=", 1)[1] for a in argv[1:] if a.startswith("--llama-cpp-dir=")), None)
    smoke = any(a == "--smoke" for a in argv[1:])
    if not adapter or not llama_dir:
        print("usage: python -m training.export --adapter=<dir> --llama-cpp-dir=<dir> [--smoke]",
              file=sys.stderr)
        return 2

    cfg = load_config()
    out_dir = Path(adapter).parent
    merged_dir = out_dir / "model" / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # 1. merge adapter into the base model, save fp16
    import torch
    from peft import PeftModel

    tokenizer = setup_tokenizer(cfg.model_name)
    base, resolved = load_model_with_fallback(cfg)
    model = PeftModel.from_pretrained(base, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))
    print(f"merged model saved to {merged_dir}")

    # 2. convert + quantize with llama.cpp
    llama_dir = Path(llama_dir)
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_dir / "llama-quantize"
    f16_gguf = merged_dir / "qwen3.5-0.8b-speech-compiler-f16.gguf"

    subprocess.run(
        [sys.executable, str(convert_script), str(merged_dir),
         "--outfile", str(f16_gguf), "--outtype", "f16"],
        check=True,
    )
    print(f"converted to {f16_gguf}")

    for quant, name in (("Q8_0", "qwen3.5-0.8b-speech-compiler-Q8_0.gguf"),
                        ("Q4_K_M", "qwen3.5-0.8b-speech-compiler-Q4_K_M.gguf")):
        subprocess.run(
            [str(quantize_bin), str(f16_gguf), str(merged_dir / name), quant], check=True
        )
        print(f"quantized {quant} -> {merged_dir / name}")

    # 3. smoke test: 3 prompts through llama-cli (results go into the README)
    if smoke:
        from data.common import INSTRUCTION

        cli = llama_dir / "llama-cli"
        prompts = [
            "uh i i wanted to send sarah the project files tomorrow actually no wednesday morning",
            "make a list of milk eggs and bread",
            "the meeting is from 3 to 4",
        ]
        results = []
        for p in prompts:
            proc = subprocess.run(
                [str(cli), "-m", str(merged_dir / "qwen3.5-0.8b-speech-compiler-Q4_K_M.gguf"),
                 "-p", f"<|im_start|>system\n{INSTRUCTION}<|im_end|>\n"
                       f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
                 "-n", "128", "--no-display-prompt"],
                capture_output=True, text=True, timeout=300,
            )
            results.append({"input": p, "output": proc.stdout.strip()})
        (out_dir / "gguf_smoke_test.json").write_text(
            __import__("json").dumps(results, indent=2)
        )
        print("smoke test results written to gguf_smoke_test.json")

    print(f"export complete: {merged_dir} (base quantization for merge: {resolved})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
