# Experiment 2 — Rambler-Style Speech Compiler: Qwen3.5-0.8B Fine-Tuning & Evaluation Pipeline

## Context

The Flutter app (`mumble_jumble`) transcribes speech with Whisper and needs a local "speech compiler" that rewrites messy spoken transcripts into clean intended text (Rambler/Wispr Flow-style). Experiment 1 (Gemma-3-270M-it LoRA) went 5.2% → 45.0% exact match, but repetition EM was only 14.3% and mid-sentence corrections ("...tomorrow actually no wednesday...") failed — the model learned isolated transformations, not **compositional** ones.

Experiment 2 builds a ground-up pipeline (dataset engine → hold-out stress test → QLoRA fine-tuning → multi-metric evaluation → failure analysis) around **Qwen3.5-0.8B**, targeting Android deployment under **1.5 GB RAM**. The user's 28-section brief is the requirements document; this plan fixes concrete technical decisions and execution order. Priority: problem definition → dataset → evaluation → fine-tuning → failure analysis → Android. **No training until the dataset engine and stress test are validated.** The pipeline lives at workspace root: `rambler-compiler/` (sibling to the Flutter app; Experiment 1's `google colab stuff/rambler-training` stays untouched as reference).

## Verified Research Findings (FIRST TASK items 1–5)

### 1. Model — Qwen3.5-0.8B exists and is confirmed

- **Qwen/Qwen3.5-0.8B** (instruct; Apache-2.0) + **-Base** variant. **Choose instruct**: post-trained for instruction-following, non-thinking by default, and the GGUF ecosystem (unsloth) quantizes exactly this variant.
- Architecture (config.json verified): `Qwen3_5ForConditionalGeneration`, `model_type qwen3_5`; 24 layers via `layer_types` — 18 Gated DeltaNet (linear attention) + 6 full attention (indices 3,7,11,15,19,23); **dense** FFN; hidden 1024, FFN 3584, vocab 248,320, head_dim 256; native context 262k; MTP (1 nextn layer); early-fusion **multimodal** (vision tower in checkpoint, unused for our text-only task).
- Tokenizer: `Qwen2Tokenizer`, ChatML-style (`<|im_start|>/<|im_end|>`, eos `<|im_end|>`, pad `<|endoftext|>`). Chat template has `enable_thinking` param; **default = non-thinking** — we pass `enable_thinking=False` explicitly.

### 2. Loading requirements

- Pin versions Experiment 1 verified on Colab (Aug 2026): **transformers 5.16.1, peft 0.20.0, trl 1.12.0, accelerate 1.14.0, datasets 5.0.1, bitsandbytes latest**, torch = Colab's CUDA build. No `trust_remote_code`.
- Load original checkpoint via `AutoModelForCausalLM` → `Qwen3_5ForConditionalGeneration`; vision tower loads but is never called. `use_kernels=True` is inference-only — NOT used in training.

### 3. LoRA/QLoRA support — yes

- PEFT `target_modules="all-linear"` is architecture-agnostic (all nn.Linear/Conv1D minus lm_head) — covers DeltaNet projections + dense FFN. We filter resolved names to **exclude vision-tower and `nextn` (MTP) modules** → text-backbone-only adapters → prefix-safe merge onto the original checkpoint for GGUF.
- TRL v1.x: `SFTTrainer(model, args=SFTConfig, processing_class=tokenizer, peft_config=..., quantization_config=BitsAndBytesConfig(...), formatting_func=...)` (verified current API).
- QLoRA: 4-bit NF4 + double quant, compute dtype **fp16 on T4** (bf16 slow there), bf16 on Ampere+ — configurable.

### 4. Chat template / inference instruction

- `formatting_func`: `[{"role":"system","content":INSTRUCTION},{"role":"user","content":transcript}]` → `apply_chat_template(tokenize=False, add_generation_prompt=True, enable_thinking=False)`. `INSTRUCTION` lives in `data/common.py` as single source of truth (the exact prompt the Flutter app must use later):
  > "You are a speech-to-text compiler. Rewrite the spoken transcript into the text the user intended to type. Remove disfluencies, stutters, repetitions, abandoned phrases, and incorrect earlier wording. Resolve self-corrections using the user's final intended wording. Preserve meaningful information, names, numbers, dates, times, and technical terms. Apply requested formatting. Output only the final text."

### 5. Android inference / export

- **Primary path: GGUF via llama.cpp** — native `LLM_ARCH_QWEN35` incl. Gated DeltaNet kernels (verified in llama.cpp source `src/models/qwen35.cpp`); unsloth ships Qwen3.5-0.8B GGUFs (Q4_K_M ≈ 533 MB, Q8_0 ≈ 812 MB — well under 1.5 GB with headroom for KV cache + Whisper). Export: merge adapter onto original checkpoint → `convert_hf_to_gguf.py` → `llama-quantize` Q4_K_M + Q8_0.
- Flutter bindings mature: `llama_cpp_flutter`, `llamadart` (Qwen3.5 MTP speculative decoding, LiteRT/NPU options), `llama_cpp_dart`. App currently hardcodes `flutter_gemma` (Gemma-only, `lib/services/gemma_cleanup_service.dart:53`) — switching is a **later phase**, out of pipeline scope; constraint recorded: app prompt == training INSTRUCTION, GGUF asset ~500 MB.
- Alternatives noted, not locked: ExecuTorch, MediaPipe/LiteRT, Qualcomm AI Hub (has qwen3_5_0_8b). Deferred until quality is proven (spec §20).

## Key Decisions

| Decision | Choice | Why |
|---|---|---|
| Variant | Instruct | Instruction-shaped task; base needs far more training; GGUF ecosystem matches instruct |
| Training checkpoint | Original multimodal (text-only inputs) | Adapter merge onto same checkpoint = prefix-safe GGUF via official converter; community text-only checkpoints aren't llama.cpp-supported for export |
| Dataset size | 15,000 records | Spec range 10–20k; balances diversity vs Colab time |
| Seq length | 1024, packing (train only) | Long-form ≤ ~300 words ≈ ≤600 tokens; packing amortizes padding; eval stays unpacked for comparable loss |
| LoRA | r=16, α=32, dropout 0.05, all-linear minus vision/nextn | Exp 1's proven recipe; all-linear covers DeltaNet layers |
| Optimizer | paged_adamw_8bit, lr 2e-4 cosine, warmup 10%, 3 epochs, eff. batch 32 (4×8), fp16 on T4 | Exp 1's working config; ~400 steps/epoch ≈ 1,200 steps total |
| Loss | `train_on_responses_only=True` (response template `<\|im_start\|>assistant\n`), flag-controlled | Exp 1 pattern; plain completion-loss is fallback |
| Eval generation | Greedy (`do_sample=False`), max_new_tokens 256 | Deterministic; outputs are short |
| bnb fallback ladder | 4-bit NF4 → 8-bit → fp16 LoRA; `resolved_quantization` written to config.json (never silent) | Novel DeltaNet arch may hit bnb edge cases |
| Metrics | Deterministic rule-judge (no GPU/LLM judge) + jiwer WER/CER with stdlib fallback | Exp 1 pattern; reproducible, zero cost; Python 3.14 wheel safety |
| Repo tiers | `data/`, `evaluation/`, `tests/` never import torch (grep-tested); `training/` imports torch lazily inside functions | Local machine is CPU-only Python 3.14; dataset + metrics run anywhere |

## Repository Layout (new project at workspace root)

```
rambler-compiler/
├── config/training_config.py      # TrainingConfig dataclass (spec §16 names) + load/override, no-torch validation
├── data/
│   ├── common.py                  # INSTRUCTION, normalize(), load/write_jsonl, md5_rng() (stable across runs), CATEGORIES
│   ├── generators/
│   │   ├── entities.py            # entity vocab pools + span extraction + same-kind entity_variant()
│   │   ├── templates.py           # ~540 clean-output templates (12 domains, typed slots, unit roles)
│   │   ├── injectors.py           # filler / repetition / correction / reformulation / formatting
│   │   ├── longform.py            # 50–300-word monologue composer (units + connectives + offsets)
│   │   └── composer.py            # Plan → record; canonical injector order; metadata by construction
│   ├── validators/quality.py      # gate checks V1–V10 + dataset stats collectors
│   ├── dataset_builder.py         # BuildConfig → plans → compose → gate → dedup → stratified 85/10/5 → JSONL + manifest
│   ├── stress_test.jsonl          # 200 hand-written records (HOLD-OUT, never in training)
│   └── splits/<version>/train.jsonl, validation.jsonl, test.jsonl
├── training/
│   ├── model_utils.py             # tokenizer setup, chat-template formatting, bnb ladder, module inspection, LoRA build
│   ├── train.py                   # QLoRA SFT driver + MetricEvalCallback + experiment artifacts
│   ├── evaluate.py                # inference-only: model → predictions.jsonl (torch)
│   └── export.py                  # merge → convert_hf_to_gguf.py → quantize Q4_K_M/Q8_0 → llama_cpp smoke test
├── evaluation/                    # torch-free; runs on the CPU laptop
│   ├── metrics.py                 # score_record + aggregate + compositional_accuracy
│   ├── failure_analysis.py        # 10 failure buckets + "why" heuristics → JSON + report
│   └── stress_test.py             # predictions over stress set → stress_test_results.json (per level + per combination)
├── notebooks/rambler_training.ipynb
├── outputs/experiments/<exp>/     # config.json, dataset_stats.json, training_metrics.json, validation_metrics.json,
│                                  # stress_test_results.json, failure_analysis.json, predictions/, model/, README.md
├── tests/                         # ~200 pytest, CPU-only
├── requirements-data.txt          # stdlib only
├── requirements-eval.txt          # jiwer (with stdlib fallback path)
├── requirements-dev.txt           # pytest
├── requirements-training.txt      # Colab pins (verified versions)
└── README.md                      # phases, run instructions, Android notes
```

## Dataset Design

### Record schema (JSONL; metadata never shown to the model)

```json
{
  "id": "mix_000123",
  "input": "<messy transcript>",
  "output": "<clean target>",
  "category": "basic|filler|repetition|correction|reformulation|formatting|long|mixed",
  "transformations": ["filler", "repetition", "self_correction"],
  "num_transformations": 3,
  "difficulty": "easy|medium|hard",
  "domain": "<12 domains>",
  "entities": {"names": [], "dates": [], "times": [], "numbers": [], "locations": [], "tech": [], "items": []},
  "corrections": [{"abandoned": "tomorrow", "kept": "wednesday morning", "kind": "date", "marker": "no wait"}],
  "reformulations": [{"abandoned": "...", "kept": "..."}],
  "formatting": null | {"type": "bullet_list|heading|polite_message", "items": []},
  "preservation_case": false,
  "preserved": [],
  "seed": 123,
  "generator_version": "v1.0.0"
}
```

Additive schema fields (`reformulations`, `preserved`) make reformulation and preservation checks metadata-driven instead of regex-guessed. Identity records allowed: `transformations: []`, `input == output`, only inside `basic` (~15% of basic ≈ 340) — teaches "don't over-edit" (spec §7/§11).

### Category quotas (15,000 total — spec §4 percentages, exactly)

| Category | % | Count | Notes |
|---|---|---|---|
| basic | 15 | 2,250 | minimal noise; ~340 identity records |
| filler | 10 | 1,500 | filler insertion only |
| repetition | 10 | 1,500 | word/phrase/partial-word stutter, dynamic |
| correction | 20 | 3,000 | 7 entity kinds × ~14 markers |
| reformulation | 10 | 1,500 | abandoned phrases / false starts |
| formatting | 5 | 750 | lists 45% / headings 25% / polite 30% |
| long | 15 | 2,250 | buckets 30–60 / 60–100 / 100–150 / 150–300 tokens |
| mixed | 15 | 2,250 | **explicit composition matrix** |

**Preservation (spec §7)** is a flag, not a 9th category: 600 records (10% carved from repetition/correction/reformulation budgets) are drawn from `PRESERVATION_TEMPLATES` ("the meeting runs from {time} to {time}", "check the files twice, once now and once on {date}", "it was really really important") with `preservation_case: true` and `preserved: [substrings]` that must survive verbatim. Reports slice by the flag; spec's category list stays exact.

### Composition matrix (mixed — guaranteed per-combination counts, 225 each, spec §5)

filler+rep · filler+corr · rep+corr · rep+reform · corr+fmt · filler+rep+corr · filler+corr+fmt · rep+corr+long · filler+rep+corr+fmt · multi-corrections-different-entities.

### Generator architecture — clean output FIRST, then inject noise (spec §6–§9)

1. **entities.py**: pools sized for diversity — 150 names, 50 surnames, 180 items, 90 locations, 110 tech terms, ~2000 numbers (digits+word forms), ~1500 times, ~2400 dates (combinatorial generators); ~18 fillers; ~14 correction markers. **Anti-association rule (mechanical):** abandoned-value pool == kept-value pool — every enumerated entry must appear ≥5× as a kept value and (names/items/locations/tech) ≥3× as an abandoned value; enforced by validator V9 (build fails below floor). No word is ever "the correction word".
2. **templates.py**: ~540 templates, 12 domains × ~45, varied grammatical subjects (I/we/the report/the order/…), imperative/declarative/question forms, `role` tag (any/opener/middle/closer) for longform. Slots filled without-replacement within a record.
3. **injectors.py** — uniform API `(text, rng, difficulty) → InjectorResult(text, delta)`; deterministic; guarantees a change; never mutates entity-internal characters:
   - **filler**: insertion at token boundaries only (never inside "5:30" or "five thirty"); counts by difficulty 1/1–2/2–4; at hard, filler *inside* a correction marker cluster allowed ("no, um, wait") — tracked anyway.
   - **repetition**: weighted flavors — word 40% ("the the"), 2–3-word phrase 25%, partial stutter 20% ("w-w-went"), clause echo 15%; **protected spans (correction clusters + preserved spans) never repeated**.
   - **correction**: pick span of required kind → `wrong = entity_variant(span)` (same-kind alternative, never equal, format-preserving: digits↔digits, words↔words) → replace with `"{wrong} {marker} {right}"` where right == clean text verbatim (the invariant). Markers: "no wait", "actually", "I mean", "sorry", "make that", "change that to", "wait", "not X, Y", "I meant Y", "rather", "oh hold on", "I was going to say X, but Y works better" (hard). Multi-attribute: swap name+date+time in one utterance on distinct spans.
   - **reformulation**: (a) identical restart 60% (abandon 1–4-word prefix, restart verbatim), (b) reworded 40% (abandoned = paraphrase from a rule bank, kept = clean text). Markers: "I mean", "what I meant is", "let me rephrase that", "so basically", "in other words". Metadata: `reformulations`.
   - **formatting**: wraps, doesn't noise — spoken command as input ("make a list of …", "make this a heading: …", "make it polite: …"), output = derived artifact (`- item` lines / `# Heading` / polite wrapper from ~10 patterns). **Composition rule:** when formatting + other transforms combine, the noise applies to the *command text* and the output is the artifact with the corrected value — invariant survives by construction.
4. **longform.py**: 2–4 same-domain units (opener/middle/closer roles — the seam-prevention mechanism), ~30 connectives ("and then", "oh and", "by the way", "so basically", …) never repeated within a record, unit offsets tracked so injectors target specific units; difficulty = length (50–80 / 80–150 / 150–300 words).
5. **composer.py**: `Plan(category, transforms, domain, difficulty, key)` → `compose()`:
   - `plan_rng = Random(md5(f"{seed}:{plan.key}"))` — md5, not `hash()`, so regeneration is byte-identical across processes/runs (Exp 1 pattern; pinned by a test).
   - Template selection filtered by `required_kinds` (the kinds the plan's corrections need) — structurally eliminates the "correction with no suitable slot" edge case; slot-miss after 10 retries → `None`, builder reallocates and counts gaps.
   - Canonical injector order: **correction → reformulation → repetition (with protected spans) → filler** — surface noise last so it never corrupts marker clusters; preservation plans forbid correction and allow repetition only outside `preserved` spans.
   - Metadata by construction, never re-extracted (except entity lists computed from the well-formed clean output).
6. **dataset_builder.py**: exact per-category/combo allocation → ordered Plan list (deterministic) → compose → gate validators (discard + regenerate, bounded) → grouped near-dup pass → **stratified 85/10/5 split by category** (dedicated `random.Random(seed)`; every category ≥25 in test) → `data/splits/<version>/` + `manifest.json` + `dataset_stats.json`.

### Validators (quality.py) — spec §24

V1 structure/enums; V2 input≠output unless identity; V3 non-empty ≥3 tokens; **V4 correction consistency (core invariant): abandoned present in input, kept present in output, abandoned absent from output, kept≠abandoned**; V5 entity preservation (output entities == metadata); V6 formatting validity (list/heading/polite structure; no markdown where `formatting: null`); V7 length bounds; V8 dedup: exact pairs + near-dup via 3-gram Jaccard ≥0.85 **within template_id+entity-signature groups** (O(n), not O(n²)); **V9 vocab coverage (anti-association)**; V10 preservation (every `preserved[]` substring verbatim). Stats collectors: marker histogram, filler/repetition densities, template usage, vocab coverage, token-length percentiles, realized combos vs matrix targets, gap count.

## Rambler Stress Test (spec §10–§11, §23)

- `data/stress_test.jsonl`: **200 hand-written records**, ids `stress_0001..0200`, 50 per level: easy (simple cleanup) / medium (multiple disfluencies) / hard (long + multiple corrections) / extreme (long + corrections + formatting + many entities). Schema = data schema + `level`.
- Includes all spec §11 cases verbatim (the "tomorrow actually no wednesday morning" smoke test, "remove eggs and add paratha", server/port + professional, …), no-change cases, adversarial preservation cases ("I was going to say Tuesday, but Wednesday works better"), and **combinations NOT in the training matrix** (filler+reformulation, triple same-kind correction, formatting+correction+repetition) — the extreme level is the compositional-generalization canary.
- Never used for training/tuning: disjoint id prefix; builder never reads the file; `tests/test_stress_schema.py` asserts no stress id in any split; README rule — stress results produced once, after a checkpoint is frozen.
- Baseline (untrained model) run against it BEFORE training (Phase 3) — the bar to beat.

## Evaluation Design (spec §12–§14)

`evaluation/metrics.py` (torch-free, deterministic, prediction-file driven — scoring happens on the CPU laptop):

- `exact_match` (raw) + `normalized_match` (Exp 1 normalize semantics).
- `wer` / `cer` via jiwer, with **pure-stdlib Levenshtein fallback** (both paths tested).
- `entity_preservation` — per-kind canonical recall of `entities` from actual.
- `correction_resolution` — per `corrections[]` entry: kept canonical present AND abandoned canonical absent (all must pass).
- `hallucination` — entity canonical in actual absent from expected (abandoned-retained counts as correction failure, not hallucination — non-overlap pinned by test).
- `verbosity_flag` — token count outside 0.5–1.5× expected; `artifact_flag` — "Here is…", "Sure!", "cleaned version" regexes (Exp 1 `ARTIFACT_RE`).
- `formatting_ok` — structural checks (V6 applied to actual).
- `preservation_ok` — every `preserved[]` substring verbatim.
- **`compositional_pass` — the headline for mixed**: per-transformation checks (filler tokens absent unless in expected; no new adjacent duplication; correction resolved; no abandoned n-grams; formatting structure ok; long: unit entities present + ≥50% expected tokens). Record passes iff all listed transformations pass.
- Aggregation: overall, per-category, per-difficulty, **per-combination** (which composition collapses, exactly) + **COMPOSITIONAL ACCURACY** as its own report row (spec §13 format).

`evaluation/failure_analysis.py` — 10 buckets, first-match priority: correction_resolution (sub-reasons: kept-missing / abandoned-retained / both-retained / kept-mangled) → repetition_removal (incl. "deleted meaningful repetition" for preservation records) → hallucination (nearest-expected-entity edit-distance hint) → entity_corruption → information_deletion (content-token recall <0.8) → formatting_failure → long_context_failure → verbosity → partial_rewrite / punctuation_only (recall ≥0.9, differ only by punctuation). Output: counts + rates + dominant sub-reasons + 20 examples/bucket (INPUT/EXPECTED/MODEL/TRANSFORMATIONS/FAILURE TYPE/WHY) + priority suggestions + `review.csv`.

## Training Design (spec §15–§16)

`config/training_config.py`: single `TrainingConfig` dataclass with spec §16 field names — MODEL_NAME="Qwen/Qwen3.5-0.8B", MAX_SEQ_LENGTH=1024, LEARNING_RATE=2e-4, NUM_EPOCHS=3, BATCH_SIZE=4, GRADIENT_ACCUMULATION=8, LORA_R=16, LORA_ALPHA=32, LORA_DROPOUT=0.05, LORA_TARGET_MODULES="all-linear", LORA_EXCLUDE_MODULES=(".*vision.*", ".*visual.*", ".*nextn.*"), WARMUP_RATIO=0.1, WEIGHT_DECAY=0.01, cosine, seed 42, + bnb ladder + eval/save cadence. Loads/validates without torch (`--validate-config-only`).

`training/model_utils.py`:
- `setup_tokenizer` — Qwen2Tokenizer; explicitly set pad `<|endoftext|>`, eos `<|im_end|>`.
- `make_formatting_func` — system+user chat template, `enable_thinking=False` with TypeError fallback (template may not accept the kwarg on this checkpoint).
- `inspect_linear_modules` — logs actual nn.Linear names before LoRA (makes the novel-arch risk visible), asserts lm_head excluded, logs vision/nextn exclusions.
- `load_model_with_fallback` — 4-bit NF4 → 8-bit → fp16 ladder; each rung re-creates from scratch; `resolved_quantization` persisted to config.json.
- `build_lora_config` — all-linear + exclude regexes; fallback to explicit name list if PEFT lacks `exclude_modules`.

`training/train.py`:
1. Load tokenizer + model (ladder), attach LoRA, log final target-module list.
2. `SFTTrainer` with `processing_class=tokenizer`, `formatting_func`, `peft_config`, `train_on_responses_only=True` + response template, `SFTConfig`: packing (train only), fp16 on T4, gradient_checkpointing, eval_strategy=steps (100), save_total_limit=3, report_to="none".
3. **`MetricEvalCallback`** — on each eval: deterministic 32-record val slice (rng seeded by global_step), greedy generation, torch-free metrics via `evaluation.metrics` → logs `eval_exact`, `eval_corr_resolution`, `eval_entity_preservation` to `training_metrics.json` (fixes Exp 1's loss-only blind spot).
4. Overfit watch: val metric peak then degradation while train loss falls → warning (spec §16).
5. Artifacts to `outputs/experiments/<exp>/`: resolved config.json (incl. `resolved_quantization`, env block with library versions + GPU), training_metrics.json, adapter, README stub.

`training/evaluate.py` — inference-only: base-or-adapter (same ladder) → greedy predictions.jsonl over validation/test/stress (ids, inputs, expected, actual, metadata). `training/export.py` — `merge_and_unload` → fp16 safetensors → `convert_hf_to_gguf.py` → quantize Q8_0 + Q4_K_M → 3-example llama_cpp smoke test (recorded in README). RAM math in README: Q4_K_M ~533 MB + KV/compute headroom < 1.5 GB.

## Notebook & Experiment Tracking (spec §21–§22)

- `notebooks/rambler_training.ipynb`: self-contained Colab flow — setup (pinned deps) → build dataset (or fetch splits) → **baseline on validation+stress (before training)** → train → evaluate → compare vs baseline → export GGUF → stress once, record.
- `outputs/experiments/<exp>/`: config.json, dataset_stats.json, training_metrics.json, validation_metrics.json, stress_test_results.json, failure_analysis.json, predictions/, model/ (adapter + merged + GGUF), auto-generated README.md (model, dataset version/size, hyperparameters, duration, GPU, results, known failure modes).
- Reproducibility: global seed 42; per-record `md5_rng(seed:plan.key)`; split rng dedicated; greedy inference; **byte-identical regeneration test**; dataset version `YYYYMMDD.N` stamped everywhere.
- Latency/RAM benchmarks (spec §21) are a later phase once a GGUF exists — README stub only (spec: quality → quantization → Android).

## Implementation Phases (execution order)

1. **Scaffold** `rambler-compiler/` + requirements + tests skeleton.
2. **Dataset engine** (entities → templates → injectors → longform → composer → validators → builder) with tests; build the full 15k dataset locally (CPU) and inspect stats + samples.
3. **Stress test**: hand-write `data/stress_test.jsonl` (200, 50/50/50/50); schema-validate + assert non-overlap with splits.
4. **Evaluation harness** (metrics.py, failure_analysis.py, stress_test.py) + fixture tests (prediction-file driven, no GPU).
5. **Training + export code** (training_config.py, model_utils.py, train.py, evaluate.py, export.py) — written + import-checked locally, executed on Colab.
6. **Colab notebook** mirroring the pipeline.
7. **README + docs** (design doc, run instructions, Android follow-up notes: flutter_gemma → GGUF package switch; app prompt == INSTRUCTION).

Local verification (no GPU): `pytest` (~200 tests: generator invariants, correction consistency, dedup, split stratification, byte-identical regeneration, metric correctness on fixtures incl. WER fallback, stress schema + non-overlap, torch-import guard over data/evaluation) + full dataset build + stats report. Colab verification: baseline stress run → train → eval → failure analysis → export smoke test (user runs training; pipeline writes everything to `outputs/experiments/...`).

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| bnb/QLoRA incompatibility with Gated DeltaNet ops | Fallback ladder 4-bit → 8-bit → fp16 LoRA; `resolved_quantization` recorded (never silent); module inspection logged before LoRA |
| Vision tower / MTP polluting LoRA targets or VRAM | Exclude regexes (vision/visual/nextn) after all-linear resolution; frozen vision tower in 4-bit is small VRAM cost |
| llama.cpp `convert_hf_to_gguf.py` gaps for qwen3_5 | Verified LLM_ARCH_QWEN35 + qwen35.cpp tensor loading; unsloth GGUF exists as ecosystem evidence; fallback: unsloth converter on merged checkpoint |
| Template burnout / word↔transformation association (the Exp 1 failure) | ~540 templates × combinatorial slots; shared abandoned/kept pools + V9 coverage gate; near-dup rejection; longform frames; extreme stress level (unseen combos) as canary |
| Dataset self-deception (spec §23) | Stress test hand-written, no template overlap, disjoint ids, overlap-asserted, never tuned against; no benchmark-specific post-processing |
| Overfitting | Val-eval during training with real metrics (MetricEvalCallback), best-checkpoint tracking, overfit warning |
| Python 3.14 CPU machine | data/evaluation/tests torch-free by construction (grep guard test); jiwer optional with stdlib fallback; training runs on Colab |
