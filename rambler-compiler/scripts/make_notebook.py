"""Generates notebooks/rambler_training.ipynb from the cell definitions below.

Run: python scripts/make_notebook.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


CELLS = [
    md(
        "# Rambler Compiler — Experiment 2 (Qwen3.5-0.8B QLoRA)",
        "",
        "Full pipeline: dataset → baseline → fine-tune → evaluate → failure analysis → GGUF export.",
        "Run the cells top to bottom. The dataset engine, evaluation tier, and stress test are",
        "pure Python (no torch) and identical to the local machine's code.",
    ),
    code(
        "# ---- 1. Setup: pinned training deps ----",
        "!pip install -q transformers==5.16.1 peft==0.20.0 trl==1.12.0 accelerate==1.14.0 datasets==5.0.1 bitsandbytes jiwer",
        "# torch: use Colab's preinstalled CUDA build (do not reinstall)",
        "import torch, transformers, peft, trl",
        "print(torch.__version__, transformers.__version__, peft.__version__, trl.__version__)",
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')",
    ),
    code(
        "# ---- 2. Hugging Face auth (authenticated = faster, higher rate limits) ----",
        "# Create a READ token at https://huggingface.co/settings/tokens and add it as a",
        "# Colab secret named HF_TOKEN (🔑 sidebar → Secrets → + New secret → paste token).",
        "from huggingface_hub import login",
        "",
        "try:",
        "    from google.colab import userdata",
        "    token = userdata.get('HF_TOKEN')",
        "except Exception:",
        "    token = None",
        "",
        "if token:",
        "    login(token=token, add_to_git_credential=False)",
        "    print('HF authenticated via HF_TOKEN secret — downloads use your account rate limits.')",
        "else:",
        "    print('No HF_TOKEN secret found: downloads run unauthenticated (slower, rate-limited).')",
        "    print('Fix: add a secret named HF_TOKEN with a read token from https://huggingface.co/settings/tokens')",
    ),
    code(
        "# ---- 3. Project root: mount Drive and point at the repo copy ----",
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "import sys, os",
        "REPO_PATH = '/content/drive/MyDrive/mumble_jumble/rambler-compiler'  # <-- adjust",
        "# If the repo is missing, upload rambler-compiler/ to that Drive folder first.",
        "assert os.path.exists(os.path.join(REPO_PATH, 'data', 'dataset_builder.py')), 'repo not found'",
        "sys.path.insert(0, REPO_PATH)",
        "os.chdir(REPO_PATH)",
        "from data.common import INSTRUCTION",
        "import evaluation.metrics as M",
        "print('repo OK; INSTRUCTION:', INSTRUCTION[:60], '...')",
    ),
    md(
        "## Phase 2 — Dataset (skip if data/splits already built)",
    ),
    code(
        "# ---- 4. Build the 15k V4 hybrid dataset (only if data/splits/<VERSION> is missing) ----",
        "import json, os",
        "from data.dataset_builder import V4BuildConfig, build_v4",
        "VERSION = '20260904.1'",
        "if not os.path.exists(f'data/splits/{VERSION}/train.jsonl'):",
        "    summary = build_v4(V4BuildConfig(total=15000), 'data')",
        "    print(summary)",
        "else:",
        "    print('dataset already built')",
        "stats = json.load(open('data/dataset_stats.json'))",
        "print('total records:', stats['total'])",
        "print('per bucket:', stats['per_bucket'])",
    ),
    md(
        "## Phase 3 — Baseline: the untrained model (the bar to beat)",
        "",
        "Run the BASE Qwen3.5-0.8B on a validation sample and the full 200-record stress set",
        "BEFORE training.",
    ),
    code(
        "# ---- 5. Baseline predictions (validation sample + FULL stress set) ----",
        "from training.model_utils import load_model_with_fallback, setup_tokenizer, generate",
        "from config.training_config import load_config",
        "from data.common import load_jsonl, write_json",
        "cfg = load_config()",
        "tokenizer = setup_tokenizer(cfg.model_name)",
        "model, resolved = load_model_with_fallback(cfg)",
        "print('loaded with quantization:', resolved)",
        "",
        "val_records = load_jsonl(cfg.validation_file)[:300]  # sample for speed",
        "for name, records, out in [('baseline_val', val_records, 'outputs/baseline_val.jsonl'),",
        "                           ('baseline_stress', load_jsonl('data/stress_test.jsonl'), 'outputs/baseline_stress.jsonl')]:",
        "    import os",
        "    os.makedirs('outputs', exist_ok=True)",
        "    with open(out, 'w') as f:",
        "        for r in records:",
        "            f.write(json.dumps({**r, 'actual': generate(model, tokenizer, r['input'], 256)}) + '\\n')",
        "    print('wrote', out, len(records), 'predictions')",
    ),
    code(
        "# ---- 6. Score the baseline ----",
        "import json",
        "from data.common import write_json",
        "import evaluation.metrics as M",
        "rows = [json.loads(l) for l in open('outputs/baseline_val.jsonl')]",
        "scored = [{'record': {k: v for k, v in r.items() if k != 'actual'}, **M.score_record({k: v for k, v in r.items() if k != 'actual'}, r['actual'])} for r in rows]",
        "write_json('outputs/baseline_validation_metrics.json', M.aggregate(scored))",
        "o = M.aggregate(scored)['overall']",
        "print(f\"BASELINE val sample: EM {o['exact_match']:.1%}  WER {o['wer']:.3f}  corr_res {o['correction_resolution']:.1%}  compositional {M.compositional_accuracy(scored)['overall']:.1%}\")",
    ),
    md(
        "## Phase 4 — Fine-tuning (QLoRA, T4-friendly)",
        "",
        "Defaults: r=16/α=32, lr 2e-4 cosine, 3 epochs, eff. batch 32, seq 1024 packing,",
        "bnb 4-bit NF4 with a fallback ladder (4bit→8bit→fp16) recorded in config.json.",
    ),
    code(
        "# ---- 7. Train ----",
        "import sys",
        "sys.argv = ['train.py', '--override=experiment_name:exp001']",
        "from training.train import main as train_main",
        "train_main(sys.argv)",
    ),
    code(
        "# ---- 8. Training curves (loss + eval metrics) ----",
        "import json, matplotlib.pyplot as plt",
        "tm = json.load(open('outputs/experiments/exp001/training_metrics.json'))",
        "loss = [(h['step'], h['loss']) for h in tm['log_history'] if 'loss' in h]",
        "evals = [(h['step'], h['eval_exact']) for h in tm['metric_evals']]",
        "plt.figure(figsize=(10, 4))",
        "if loss: plt.plot([s for s, _ in loss], [v for _, v in loss], label='train loss')",
        "if evals: plt.plot([s for s, _ in evals], [v for _, v in evals], 'o-', label='eval exact match')",
        "plt.xlabel('step'); plt.legend(); plt.grid(alpha=.3); plt.show()",
    ),
    md(
        "## Phase 5 — Evaluation of the fine-tuned model",
    ),
    code(
        "# ---- 9. Generate predictions (validation + test + stress) ----",
        "!python -m training.evaluate data/splits/20260904.1/validation.jsonl --out outputs/predictions_val.jsonl --adapter outputs/experiments/exp001/adapter",
        "!python -m training.evaluate data/splits/20260904.1/test.jsonl --out outputs/predictions_test.jsonl --adapter outputs/experiments/exp001/adapter",
        "!python -m training.evaluate data/stress_test.jsonl --out outputs/predictions_stress.jsonl --adapter outputs/experiments/exp001/adapter",
    ),
    code(
        "# ---- 10. Score: full category report + COMPOSITIONAL ACCURACY ----",
        "!python -m evaluation.metrics outputs/predictions_val.jsonl --out outputs/experiments/exp001/validation_metrics.json",
        "!python -m evaluation.stress_test outputs/predictions_stress.jsonl --out outputs/experiments/exp001/stress_test_results.json",
        "import json",
        "vm = json.load(open('outputs/experiments/exp001/validation_metrics.json'))",
        "print('OVERALL:', {k: round(v, 3) for k, v in vm['overall'].items() if isinstance(v, float)})",
        "print('COMPOSITIONAL ACCURACY:', vm['compositional']['overall'])",
        "print('per combo:', {k: round(v, 3) for k, v in vm['compositional']['per_combo'].items() if v is not None})",
        "for cat, m in vm['per_category'].items():",
        "    print(f\"  {cat:14s} EM {m['exact_match']:.1%}  corr_res {m['correction_resolution']:.1%}  comp {m['compositional_pass']:.1%}\")",
    ),
    code(
        "# ---- 11. Compare against the baseline ----",
        "import json",
        "base = json.load(open('outputs/baseline_validation_metrics.json'))['overall']",
        "fine = json.load(open('outputs/experiments/exp001/validation_metrics.json'))['overall']",
        "print(f\"{'metric':18s} {'base':>8s} {'finetuned':>10s}\")",
        "for m in ('exact_match', 'correction_resolution', 'entity_preservation'):",
        "    print(f'{m:18s} {base[m]:8.1%} {fine[m]:10.1%}')",
    ),
    md(
        "## Phase 6 — Failure analysis (do not retrain yet — read the buckets first)",
    ),
    code(
        "# ---- 12. Bucket failures with reasons ----",
        "!python -m evaluation.failure_analysis outputs/predictions_val.jsonl --out outputs/experiments/exp001/failure_analysis.json",
        "import json",
        "fa = json.load(open('outputs/experiments/exp001/failure_analysis.json'))",
        "print('failures:', fa['total_failures'], '/', fa['total_records'])",
        "for bucket, n in fa['bucket_counts'].items():",
        "    if n: print(f'  {bucket:26s} {n:5d}  <- {fa[\"suggestions\"][bucket]}')",
    ),
    md(
        "## Phase 7 — Export to GGUF (Android via llama.cpp; ≤1.5 GB budget)",
    ),
    code(
        "# ---- 13. Merge + convert + quantize (Q8_0 / Q4_K_M) + smoke test ----",
        "!git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp",
        "!cmake -B /content/llama.cpp/build -S /content/llama.cpp -DGGML_CUDA=OFF && cmake --build /content/llama.cpp/build -j4 --target llama-quantize llama-cli",
        "# put the binaries on the expected path",
        "!ln -sf /content/llama.cpp/build/bin/llama-quantize /content/llama.cpp/llama-quantize",
        "!ln -sf /content/llama.cpp/build/bin/llama-cli /content/llama.cpp/llama-cli",
        "!python -m training.export --adapter outputs/experiments/exp001/adapter --llama-cpp-dir /content/llama.cpp --smoke",
    ),
    md(
        "## Done — record the experiment",
        "",
        "Copy outputs/experiments/exp001/ back to the project folder. The stress test results",
        "are produced exactly once, post-export — never inside a tuning loop.",
    ),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    out = REPO / "notebooks" / "rambler_training.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NOTEBOOK, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
