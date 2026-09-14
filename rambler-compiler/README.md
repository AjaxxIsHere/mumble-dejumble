# rambler-compiler — V4: Free Rule-Based Hybrid Dataset

Fine-tuning and evaluation pipeline for a **local speech-to-clean-text compiler**
(Rambler / Wispr Flow style): Whisper transcript in → clean intended text out.
Model: **Qwen3.5-0.8B (instruct)**, QLoRA fine-tuned, targeting Android
deployment under **1.5 GB RAM** via llama.cpp GGUF.

## Why V4?

Earlier experiments evolved through two generations of data strategy:

- **V1/V2 (fully synthetic templates):** the template + entity engine produced
  clean text and noisy input entirely from scratch. Coverage was great but the
  *clean targets* were machine-written and drifted from how people actually
  dictate.
- **The paid-API experiment (real-data-first + LLM augmentation):** the right
  idea — start from real dictated sentences — but it paid a commercial LLM
  per example (rate limits, latency, cache, an API key in the repo, cost
  tracking, a second paid LLM pass as a judge). Every rebuild spent money.

**V4 keeps the goals and drops the bill.** It keeps the real-data-first idea
and the target dataset distribution, but *realizes* every transformation with
the deterministic, rule-based noise engine (`data/generators/injectors.py`)
that V1/V2 already proved out. Result:

- **100% free** — no external APIs, no keys, no tokens, no cost tracking.
- **Fully deterministic** — same config ⇒ byte-identical splits.
- **Reproducible offline** — the real speech source (Nyra) is optional and
  cached; without it the build degrades to templates only, with a warning.
- **Real clean targets** — a large share of clean text is real dictated
  speech (`nyralabs/disfluency_speech_english` intended transcripts), so the
  model learns to clean the way people actually speak; templates fill the
  buckets real speech can't cover (formatting commands, structured lists).

## Repository Layout

```
rambler-compiler/
├── config/training_config.py      # TrainingConfig (points at the V4 split)
├── data/
│   ├── common.py                  # INSTRUCTION, normalize, V4 constants
│   ├── dataset_builder.py         # V2 legacy builder + V4 hybrid builder (CLI)
│   ├── augmentation/
│   │   ├── recipes.py             # Typed transformation recipes
│   │   └── realizer.py            # V4 rule engine: recipe -> messy input
│   ├── generators/                # Template clean text + noise injectors
│   ├── sources/
│   │   └── nyra.py                # Real clean-text source (free, cached)
│   ├── validators/                # V1-V10 gates, coverage, dedup
│   ├── stress_test.py             # 200 hand-written hold-out records
│   ├── stress_test_v4.py          # +200 hand-written conversational records
│   └── splits/                    # Generated splits (gitignored)
├── training/                      # QLoRA SFT (torch-free until Colab)
├── evaluation/                    # Metrics, stress runner (torch-free)
├── tests/                         # pytest suite, CPU-only
├── notebooks/rambler_training.ipynb   # Colab training notebook (generated)
└── docs/design.md                 # Full design document
```

## Quick Start

### 1. Run all tests (CPU machine)
```bash
cd rambler-compiler
python3 -m venv .venv && .venv/bin/pip install pytest datasets
.venv/bin/python -m pytest  # 300+ tests, ~15s
```

### 2. Build the V4 dataset (15k, hybrid)
```bash
python -m data.dataset_builder --v4 --total 15000 --out data   # real-first by default
# Writes data/splits/20260904.1/{train,validation,test}.jsonl + manifest
# and refreshes data/dataset_stats.json
```

The build never touches the network unless the Nyra cache is missing; it
prints a warning and uses templates only when the real source can't load.

### 3. Inspect the result
```python
from data.common import load_json
stats = load_json("data/dataset_stats.json")
print(stats["per_bucket"], stats["per_source"])
```

### 4. Train (Google Colab — T4)
Copy this folder to Drive, open `notebooks/rambler_training.ipynb`
(generated from `scripts/make_notebook.py`), and run it top to bottom.

### 5. Evaluate
```bash
python -m evaluation.metrics predictions.jsonl --out validation_metrics.json
python -m evaluation.stress_test predictions.jsonl --out stress_results.json
python -m evaluation.failure_analysis predictions.jsonl --out failure_analysis.json
```

### 6. Export GGUF
```bash
python -m training.export --adapter outputs/experiments/exp001/adapter \
    --llama-cpp-dir /path/to/llama.cpp --smoke
```

## V4 Data Architecture

### Hybrid clean-text sources
Every training example pairs a **messy input** with a **clean target**. The
clean target always comes from one of two sources:

1. **Real speech (Nyra)** — `nyralabs/disfluency_speech_english` intended
   transcripts: sentences a person actually dictated. Filtered to sentence-
   case texts of 3–60 words. Real clean text is the default source for the
   no-op, simple, correction, and compositional buckets: a sentence may back
   up to `real_max_uses` (3) DIFFERENT recipes — each recipe realizes it with
   its own seed, so the (input, output) pairs stay distinct — but only once
   as an identity no-op, so identity records never repeat a sentence.
2. **Template engine** — used where real speech cannot serve: formatting
   (spoken-command → artifact pairs, which real dictation rarely produces
   verbatim) and topping up any bucket the real pool can't fill (entity
   corrections need an extractable entity, so the pool gates them).

`real_ratio` (default 0.85) is an upper bound on the real share of the
non-formatting mix; the manifest reports the actual `per_source` counts
(capacity is capped by pool size × reuse and by eligibility — only ~26% of
real sentences contain an entity the correction injector can use, so
correction-heavy plans top up from templates when the pool is exhausted).

### Dataset Distribution (target)
The whole dataset is allocated to five buckets, mirroring the plan that
defined the failed paid experiment — now met without paying anything:

| Bucket | Share | What the model sees |
|---|---|---|
| No-op | 20% | clean input == clean output (learns not to over-clean) |
| Simple disfluencies | 15% | fillers (`um`, `like`, …) or repetitions (`the the`) |
| Corrections / restarts | 25% | self-corrections (`Tuesday no wait Thursday`), reformulations & restarts |
| Compositional | 30% | 2–3 transformations at once (`filler + repetition + correction`) |
| Formatting | 10% | spoken formatting command → bullet list / heading / polite message |

Within each bucket, difficulties (easy/medium/hard) and sub-kinds are drawn
from a seeded RNG, so the mix is stable across runs.

### Realization — the rule engine (`data/augmentation/realizer.py`)
Recipes describe *what* to add; the rule engine decides *how*, deterministically:

| Transformation | Rule (injector) |
|---|---|
| `filler` | insert from a 19-word filler bank at token boundaries, never inside an entity |
| `repetition` | word / phrase / partial-word stutter / clause echo |
| `self_correction` | swap a real value for a same-kind alternative, then correct back with a natural marker |
| `reformulation` | restart (repeat the abandoned prefix) or distract (hedge + drop the distractor) |
| `formatting` | `make_formatting` — bullet lists, headings, polite messages |

Corrections require an extractable entity (name/date/time/number/location/…)
in the clean text; the allocator only assigns them to texts that have one and
falls back to another kind otherwise. The clean target is never invented and
never modified — metadata is derived from the injector deltas by construction.

### Quality gates (all free, all deterministic)
1. **Per-record V1–V10 checks** (`data/validators/quality.py`): structure,
   input ≠ output unless identity, capitalization/whitespace, the correction
   invariant (abandoned in input, kept in output, kept ≠ abandoned, no leak),
   entity preservation, formatting validity, length bounds.
2. **Dedup** — exact pairs + near-duplicates via 3-gram Jaccard within
   (template-id / entity-signature) groups; offenders are regenerated.
3. **V9 vocab coverage** — every entity-pool value appears as a kept value and
   (soft) as an abandoned correction value, so no word is ever “the correction
   word”. Hard-fails the real build when the floors are unmet.

There is no LLM judge: the gates above are exactly the properties that used to
be double-checked by a paid second API call, minus the cost.

### Reproducibility
Same config + same (optionally cached) real pool ⇒ byte-identical JSONL,
pinned by `tests/test_v4_builder.py::TestReproducibility`.

## Real-data source (free)

- `nyralabs/disfluency_speech_english` (HuggingFace, ~4,960 records) —
  verbatim transcripts of real speech paired with intended clean text.
- The loader strips bracket tags (`[UH]`, `[UM]`, `[laughter]`) and cutoff
  asterisks so verbatim resembles Whisper output; the intended transcript is
  preserved exactly and becomes the clean target.
- HuggingFace `datasets` is the only data dependency. The dataset is cached
  under `~/.cache/huggingface`, so builds run fully offline after the first
  download.

## Training Pipeline

QLoRA SFT on Qwen3.5-0.8B (instruct):
- LoRA: r=16, α=32, all-linear (minus vision/MTP)
- Optimizer: paged_adamw_8bit, lr 2e-4, cosine, 3 epochs
- Batch: 4 × 8 gradient accumulation = effective 32
- Max seq length: 1024
- Training instruction: `data/common.py INSTRUCTION` (single source of truth —
  the app must use this exact prompt at inference)

`config/training_config.py` points at the V4 split by default
(`data/splits/20260904.1/`).

## Android Deployment

- Target: GGUF Q4_K_M (~533 MB) under 1.5 GB total
- Export: merge adapter → convert_hf_to_gguf.py → llama-quantize
- App prompt must equal `data/common.py INSTRUCTION`
- Flutter bindings: `llama_cpp_flutter` or `llamadart`

## CLI

```bash
# Build the V4 dataset (defaults: 15k, real_ratio 0.5, out data/)
python -m data.dataset_builder --v4

# Tune it
python -m data.dataset_builder --v4 --total 5000 --real-ratio 0.9 --real-max-uses 4 --seed 7 \
    --out outputs/experiments/exp002/splits --dataset-version 20260904.2

# Load Nyra clean texts directly
python -c "from data.sources.nyra import load_nyra; print(len(load_nyra()))"

# Regenerate the training notebook after editing scripts/make_notebook.py
python scripts/make_notebook.py
```

## Configuration

All configuration is centralized:
- `config/training_config.py` — training hyper-parameters + dataset pointers
- `data/dataset_builder.py` — `V4BuildConfig` (totals, distribution ratios,
  real_ratio, Nyra source, gates) and the legacy `BuildConfig` for the V2 path
- `data/common.py` — `INSTRUCTION`, categories, V4 constants
