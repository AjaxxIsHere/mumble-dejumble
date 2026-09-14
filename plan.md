# Plan: On-Device Rambler-Style Voice Transcriber

> Brainstormed 2026-09-06. Two-stage architecture, English-only v1, GGUF export,
> per-utterance cleanup. Goal: match the **core dictation-cleanup experience** of
> Google Rambler with a <1B-parameter model running under ~1GB RAM.

---

## 1. Goal & Scope

### What we're building
A two-stage voice-to-text pipeline where spoken, disfluent dictation ("um I I think
we should uh— no wait, we must go tomorrow") becomes clean formatted text
("We must go tomorrow.") — all on-device, under 1GB RAM.

### v1 Feature checklist (Rambler parity targets)
1. ✅ Filler-word removal (um, uh, like, you know…) — context-aware, not regex
2. ✅ Mid-sentence self-correction resolution ("I went to— no, sorry, I took the…")
3. ✅ Stutter / repetition removal ("I I", "the- the")
4. ✅ Punctuation, capitalization, paragraphing
5. ⏸ Inline dictation commands ("period", "comma", "new paragraph") — **deferred** (v1 dataset has no spoken-command rows; v1.1 data pass)
6. ✅ Graceful fallback (cleanup failure → return raw ASR text, Rambler-style)
7. ⏸ Cleanup modes — **v1 ships a single standard cleanup**; mode tokens deferred to v2
8. ❌ Streaming cleanup — deferred to v2
9. ❌ Multilingual — out of scope, no plans

### Non-goals for v1
- Streaming/chunked cleanup while user speaks (v2)
- Speaker diversity in audio (irrelevant — cleanup is text-only)
- Languages other than English

---

## 2. Architecture

**Decision: two-stage pipeline.** Neither candidate cleanup model has audio input,
and this matches Rambler's own design (Gemini 3.5 Transcribe + LLM cleanup).

> ⚠️ **Kernel-compatibility warning:** Qwen3.5-0.8B uses a hybrid **Gated DeltaNet
> linear-attention** architecture (3:1 linear:full-attention layout, inherited from
> Qwen3-Next). Linear-attention kernels are a known tripwire in Unsloth and in
> third-party GGUF runtimes (llamadart / Flutter GGUF bindings): training can fail
> or silently misbehave, and on-device inference can crash or kernel-panic.
> **Decision: Qwen3.5-0.8B is the only model to train** (accepted risk — capability
> over caution). Risk is managed, not ignored: if Unsloth or GGUF export misbehaves,
> training/export follows the §5.2 fallback path (HF `peft` + native converter).

```
Mic audio (Flutter)
  → [Stage 1: ASR]  whisper.cpp tiny.en  (locked for Phases 1–3, ~40 MB)
  → verbatim text (lowercase, spotty punctuation, ASR quirks)
  → [Stage 2: Cleanup LLM]  Qwen3.5-0.8B fine-tune, GGUF Q4_K_M  (~550 MB)
  → clean formatted text
```

Because Stage 2 is text-in/text-out:
- The DisfluencySpeech **audio** is mostly irrelevant to us; we use its **transcript pairs**.
- Cleanup quality is speaker-independent → the dataset's single-speaker limitation
  is mitigated at the text level via synthetic diversity (§4).

### Model choices

| Component | Choice | Size (quantized) | Notes |
|---|---|---|---|
| Cleanup LLM | **Qwen3.5-0.8B** (instruct, Apache 2.0) — the only model trained | ~550 MB @ Q4_K_M | Hybrid Gated DeltaNet — kernel tripwire acknowledged (warning above); §5.2 fallback path applies |
| ASR | **whisper.cpp `tiny.en`** — locked for Phases 1–3 | ~40 MB | Stable Dart FFI bindings (`whisper_flutter`), proven mobile NEON/Metal acceleration |
| Runtime target | llama.cpp / GGUF on-device | — | `n_ctx` hard-capped at **1024**; Flutter integration later |

### RAM budget (must stay < 1GB)

| Item | Estimate |
|---|---|
| Cleanup LLM Q4_K_M (Qwen3.5-0.8B) | ~550 MB |
| ASR model (whisper.cpp tiny.en) | ~40 MB |
| KV cache @ **n_ctx = 1024** + compute buffers | **~30–60 MB** cache + ~100–140 MB buffers |
| **Total (cleanup resident, ASR transient)** | **< 800 MB** ✅ |

Rules:
- `n_ctx` hard cap = **1024 tokens** in llama.cpp (dictation utterances rarely
  exceed ~80 tokens; 1024 gives ~12× headroom for utterance + template). Most
  Qwen3.5 layers are linear-attention with fixed-size recurrent state rather than
  growing KV, so the cache stays small even at 1024 (~30–60 MB).
- Ignore the model's 256K context marketing entirely.
- Never load ASR + cleanup weights simultaneously: run ASR → release → cleanup.
  Peak memory is bounded by cleanup model + its cache.

---

## 3. Cleanup behavior (v1: single standard mode)

**Decision (post-audit): v1 trains ONE cleanup behavior** — the `<mode:standard>`
semantics from the original design. No mode tokens, no settings toggle in v1.
The system prompt still leaves room for a mode prefix so v2 can add modes
without changing the runtime architecture.

Behavior: remove fillers/stutters, resolve self-corrections and distractor
reformulations, fix punctuation/capitalization. Preserve all critical entities.

Example: *"You kn-know if they ever, what about that as as, basically, an
invasion, kind of, of privacy?"* → *"You know if they ever, what about that as
an invasion of privacy?"*

Deferred (v1.1/v2): spoken punctuation commands ("period" → "."),
`<mode:verbatim>` / `<mode:concise>` tokens. The runtime change is one prompt
string; the cost of adding them later is a data pass + retrain, not an
architecture change.

**Entity guardrail (retained from the original design):** cleanup may compress
*style*, never *facts*. Dropping dates, times, locations, proper names, or
numbers that belong to the KEPT content is a hard eval failure (EPS, §4.5).
Note the audited dataset distinguishes abandoned clauses ("by Thursday, sorry,
Saturday" → "by Saturday" is CORRECT) from kept content — EPS must score
against declared kept-entities, not raw input text.

---

## 4. Data plan

### 4.1 Primary dataset — user-authored generator v4.0.0 (audited 2026-09-06)

**Decision: the user-built dataset in `Model Training Pipeline Final/` is the
primary (and only) training set for v1.** It audited clean and covers more
features, with better provenance, than the composer prototype. One target per
input (standard mode), splits provided.

| Split | Rows | Buckets |
|---|---|---|
| train | 12,750 | simple 1,913 · correction 3,187 · compositional 3,825 · formatting 1,275 · noop (identity) 2,550 |
| validation | 1,500 | same stratification |
| test | 750 | same stratification |

Audit results: 12,747/12,750 unique inputs · 0 empty/short rows · 0 Switchboard
tag leftovers · noop bucket 100% `input==output` · split leakage 1 row
(train∩val; dedupe optional) · transformation provenance consistent
(`corrections` count == `self_correction` count) · mixtures present
(filler+corr 1,169 / filler+rep 1,141 / rep+corr 923 / filler+rep+corr 592) ·
**EPS vs declared kept-entities: 0.988** (all residual "misses" are
tokenizer artifacts like `alma` vs `Alma's`; zero genuine entity corruption).

Per-row metadata (entities, difficulty, domain, transformations,
reformulation provenance, seed) is the eval harness's ground truth — EPS
scores against the **declared kept-entities**, never inferred from raw input.

Known gaps (accepted for v1, deferred to v1.1): no spoken-command rows, no
ASR-styled inputs, no per-input mode targets, `everyday` domain is 67% of rows.

### 4.2 Legacy synthetic pipeline (retired)

The `data_pipeline/` composer prototype (tag normalization, corruption ops,
3-mode targets) was **removed intentionally** after the user dataset landed.
Lessons that survive it:

- Content diversity, not op math, is the binding constraint: 15K composer
  samples over ~150 builtin bases deduped to 3.7K unique inputs.
- Switchboard tag normalization (`[UH]`→"uh", drop `[laughter]`) is required
  if DisfluencySpeech raw text is ever reused — the user generator already
  handles this (0 tag leftovers).
- The audit script (`data_pipeline/audit_user_dataset.py`) remains in use.

### 4.3 Optional extension layer (v1.1, NOT critical path)

If post-training eval shows weak buckets: an Ollama-on-Colab session (Qwen3.5-4B
Q4, pilot gate of 50 samples, negative constraints: no fact alteration, no
deletion of dates/times/locations/names/numbers) can generate:

1. spoken-command rows ("confirm the slot period" → "Confirm the slot."),
2. ASR-styled variants of existing inputs (lowercase, stripped punctuation),
3. verbatim/concise mode targets if mode tokens are revived in v2.

Nothing in v1 depends on this. Fallback provider: OpenRouter free tier.

### 4.4 Eval harness (built on the v4 dataset)

Use the dataset's own splits: **validation.jsonl** for checkpoint selection,
**test.jsonl** touched once at the end. Bucketed reporting by `bucket`,
`difficulty`, and `category`; domain tails reported but not gated.

Metrics: exact match, normalized edit distance, content-word WER vs the
reference output, and **EPS (Entity Preservation Score) computed against the
row's DECLARED kept-entities** (`entities` field), not entities inferred from
the raw input. Dropped entities from abandoned clauses ("by Thursday, sorry,
Saturday") are correct behavior and must NOT count against EPS. Any pruned
declared kept-entity is a hard failure and gates model selection.

---

## 5. Training plan (Google Colab, Python notebooks)

- **Notebook 1 — Data** (`Model Training Pipeline Final/dataset_creator.ipynb`):
  the user's generator (v4.0.0), emits train/validation/test JSONL.
  Companion audit: `data_pipeline/audit_user_dataset.py` (self-contained;
  re-run after any regeneration). No separate data-generation GPU session in
  v1 — the optional §4.3 layer is the only GPU generation, and only if needed.
- **Notebook 2 — Training**: Unsloth + QLoRA on free T4 on **Qwen3.5-0.8B** (the
  only model trained). Phase 2 gate: the shipped model must pass per-bucket eval +
  EPS **and** train and export cleanly; §5.2 fallback path applies on any failure.
  - Seq len 256 **with packing** (utterances are short; packing eliminates padding waste).
    v4 dataset ≈ 12,750 rows ≈ **~1.5M tokens/epoch** → a T4 epoch is minutes,
    so the full multi-epoch run fits ONE Colab session with margin.
  - LoRA r=16–32, 1–3 epochs, eval after each epoch
  - Possibly full fine-tune instead (0.8B may fit) — decide by eval results
  - Monitor per-bucket eval; top up composer data for weak buckets and retrain
  - **Data-scaling check:** save a checkpoint after every epoch and eval each one.
    Flat epoch-to-epoch eval → data saturated, downsample next run. Still
    improving → data volume justified. (12,750 rows ≈ ~1.5M tokens/epoch ≈
    0.0001% of the base model's pretraining tokens — overfit risk is managed by
    diversity, dedup, LoRA capacity limits, and 1–3 epochs, not by shrinking
    the dataset.)

### 5.1 Runtime budget vs Colab free-tier 3h session limit

- Token diet: compact chat template (minimal system prompt, no mode token in
  v1) + seq 256 + packing → **~1.5M tokens/epoch** on the v4 dataset.
- Benchmark-first: first cell of every training session times 200 steps and
  extrapolates the real epoch duration on the assigned GPU before committing.
- Resume-aware training: checkpoints to Google Drive every ~500 steps; any new
  session resumes from Drive. Runtime disconnects cost wall-clock time only,
  never progress. Expected shape: ~50–130 min/epoch on T4 → full run = 2–3
  sessions of <2h each.
- Session split: no data-generation GPU session exists in v1 (§4.3 is optional);
  training sessions only load the v4 JSONL from Drive and train.
- Escape hatches for one-session training: Kaggle Notebooks (free 2×T4,
  30 GPU-h/week, 12h sessions) or Colab Pro L4 (~3–4× T4 → whole run ≈ 1.5–2h).

### 5.2 Fallback path for the 0.8B (if Unsloth / GGUF export misbehaves)

If the hybrid Gated DeltaNet kernels fail in Unsloth (training) or
`save_pretrained_gguf` produces a broken/incompatible GGUF for Qwen3.5-0.8B:

1. Train with vanilla Hugging Face `transformers` + `peft` (QLoRA) — same
   hyper-parameters, slower per step (no Unsloth fast kernels), still fits a T4
   at 0.8B.
2. Export with native **`llama.cpp/convert_hf_to_gguf.py`** on the merged weights —
   the upstream converter is the reference implementation and tracks architecture
   support most closely.
3. Validate in llama.cpp CLI (n_ctx = 1024) before touching any Flutter runtime.
4. **Hard blocker:** if the 0.8B still fails there, stop and escalate upstream
   (file a minimal repro against llama.cpp/Unsloth) — no in-project fallback model
   exists by decision; do not hack around it.

---

## 6. Export & validation

1. Merge LoRA → base weights
2. Export **GGUF** via Unsloth native export (`save_pretrained_gguf`); if export
   misbehaves for Qwen3.5, use the §5.2 path (HF `peft` training + native
   `llama.cpp/convert_hf_to_gguf.py`)
3. Quantize Q4_K_M (primary) and Q5_K_M (quality comparison)
4. Smoke-test in llama.cpp with **`n_ctx = 1024`**: real RAM usage (<800 MB
   target), latency per utterance, output parity with the HF checkpoint on the
   eval set
5. Fallback logic check: malformed output / timeout → return raw ASR text

---

## 7. Phases & milestones

| Phase | Deliverable | Done when |
|---|---|---|
| 0 — Decisions | This document | ✅ agreed |
| 1 — Data | v4.0.0 dataset generated + audited | ✅ DONE (12,750/1,500/750, audit clean, EPS 0.988) |
| 2 — Training | Fine-tuned Qwen3.5-0.8B adapter | Beats heuristics-only baseline on per-bucket eval + EPS; identity & traps pass; GGUF export path proven |
| 3 — Export | GGUF (Q4_K_M) validated in llama.cpp | RAM + latency measured under budget on target-class device |
| 4 — Robustness | Fallback + edge-case pass | Cleanup failure never loses user text |
| 5 — Flutter integration | Replace/upgrade Gemma cleanup + speech_service | (deferred, design later) |

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Synthetic data distribution mismatch vs real speech | Anchor with real DisfluencySpeech pairs; validate composer stats against its transcript statistics |
| 0.8B too weak for cleanup nuance (hard corrections, reformulations) | Difficulty/category-bucketed eval exposes it; top up hard rows via §4.3 layer; fallback: accept and document |
| Cleanup prunes kept entities (dates, names, numbers) | EPS gate vs DECLARED kept-entities counts pruning as hard failure; entity-audited dataset (0.988 pre-training) |
| Hybrid Gated DeltaNet kernels fail in Unsloth / Flutter GGUF runtime | Accepted risk (0.8B is the only model); §5.2 fallback path (HF peft + convert_hf_to_gguf.py) tried first; if both fail → hard blocker, escalate upstream |
| Model edits already-clean text | Identity samples in training + identity bucket in eval |
| v1.1 data passes (commands/ASR-style/modes) if needed | Ollama on Colab GPU = no limits; §4.3 optional layer with pilot gate; OpenRouter fallback |
| RAM overruns on low-end devices | Hard context cap, Q4_K_M, benchmark before Flutter phase; ASR/cleanup never loaded simultaneously if needed |
| Colab 3h runtime limit mid-run | Resume-aware Drive checkpoints every ~500 steps; benchmark-first; token diet (packing, seq 256); Kaggle/Colab Pro fallback |
| OpenRouter free model deprecation | Composer is deterministic and provider-independent; LLM layer is a nice-to-have, not critical path |

---

## 9. Open questions / parking lot

- [ ] Moonshine as alternative ASR — post-MVP benchmark only (whisper.cpp tiny.en is locked for Phases 1–3)
- [ ] Qwen3.5-0.8B kernel verdict (Unsloth train + GGUF inference) — resolved in Phase 2; §5.2 fallback path on failure
- [ ] v1.1 optional: Ollama rewrite generation (4B pilot pass-rate) + prompt format for command/ASR-style rows
- [ ] Flutter runtime integration details (Phase 5, deferred)
- [ ] v2: streaming cleanup strategy (chunking, re-cleanup of partial text)
