# VLM Prompt Evaluation Harness

Measures VLM answer quality for the auto-triggered detection asks against the
**live** demo stack. The vision backend on :8081 now runs weapon and fall
channels concurrently (one 8-channel layout), and there are two backend_vlm
processes: **weapon on :5000, fall on :5001**. One eval run targets one
category / one port. Used to iterate on the prompt bundles, the user-message
template, and `backend_vlm/src/generation_config.json`.

## Setup

The demo stack must be up before running the harness. The vision backend
listens on **:8081** and streams weapon + fall channels concurrently on a
single 8-channel layout. The two backend_vlm processes listen on
**:5000 (weapon)** and **:5001 (fall)**. One eval run targets one category
and therefore one VLM port.

```bash
cd backend_vlm/eval
uv venv .venv
uv pip install --python .venv/bin/python "python-socketio[client]" pillow requests
```

## NPU Smoke: dual-VLM co-tenancy

`npu_smoke_dual_vlm.py` is a standalone Wave-1 verification for the dual-VLM
refactor: it proves that ARIES can hold two full Qwen3-VL-2B instances at the
same time on the disjoint NPU core slots defined in `core_allocation.yaml`
(vision encoder + text decoder pinned per instance) and that they can generate
concurrently without the runtime context-switching between overlapping cores.
If this fails, the two-process backend_vlm plan needs to pivot to a
one-process ensemble before downstream cards commit code to the split.

The smoke uses `backend_vlm`'s own venv (it imports `mblt_model_zoo` and
`mblt_tracker`), not the eval subdir's:

```bash
cd backend_vlm
uv venv .venv           # if not created
uv pip install --python .venv/bin/python -e .
.venv/bin/python eval/npu_smoke_dual_vlm.py
```

Options: `--model-id`, `--core-allocation-path`, `--no-sequential`,
`--no-concurrent`. Sequential and concurrent phases both run by default.

Exit criteria (human operator moves this card to `done` only if all hold):

- Both instances load without OOM or core-allocation errors, respecting the
  four core slots from `core_allocation.yaml` (verifiable via mblt-tracker
  per-core utilization during the run).
- `sequential_ok` and `concurrent_ok` both `true` in the JSON summary.
- Concurrent per-thread decode ms is not more than ~2x the sequential
  per-instance decode ms (same for image-encode ms). A large gap means the
  runtime is context-switching between overlapping cores — either the
  disjoint slot layout in `core_allocation.yaml` is wrong or the pinning
  kwargs are not being respected. The script flags this as "apparent NPU
  contention (vision side)" or "(text side)" in `notes`.

If the smoke fails, leave the card in review with the failing `notes` and the
four core lists so the plan can be replanned before B/D start — do not move
it to `done`.

## Workflow

1. **Capture real trigger snapshots** from the running vision backend.
   Channels are filtered by their `category` field on `/detections`; no
   mode switching is needed (both categories stream in parallel now):

   ```bash
   .venv/bin/python capture_cases.py --category weapon --count 10
   .venv/bin/python capture_cases.py --category fall --count 10
   ```

2. **Run an evaluation** against the live VLM server. `--category` picks
   the target port automatically (weapon → :5000, fall → :5001); override
   with `--vlm-port` if needed. The harness rebuilds the annotated image
   (red boxes) and the user message exactly like the frontend, streams
   the answer over Socket.IO, and scores it:

   ```bash
   .venv/bin/python run_eval.py --category weapon --label my-run \
       --bundle ../../frontend/public/prompt-bundles/weapon_detection/en
   ```

   `--bundle` defaults to the live `en` bundle for the category (its dir
   name is still `weapon_detection` / `fall_detection`); point it at any
   directory containing a `system.txt` (e.g. a scratch dir with a prompt
   variant) to test a candidate before editing the live bundle. The user
   message is always the numeric-free `build_detection_prompt` that
   mirrors the shipped frontend `buildDetectionPrompt` byte-for-byte.

3. **Compare runs** (re-scores saved answers with the current checks):

   ```bash
   .venv/bin/python rescore.py
   ```

## Checks per answer

- `coordinate_leak` — numeric tuples like `(168, 52)` / `[168, 52, 59, 46]`
  or words like "bounding box", "roi", "coordinates", "pixels"
- `metadata_leak` — echoes of detection fields (label_name, channel_index, ...)
- `repetition` — runaway loops: any 1-4 word unit repeated 3+ times in a row
  ("falling falling falling"), or the same normalized 4-gram appearing more
  than 3 times anywhere (numbers are normalized so
  "confidence: 0.97 ... confidence: 0.95" loops still match)
- `too_short` / `too_long` — outside the 15-80 word band
- `incomplete` — does not end with sentence-final punctuation (usually the
  `max_new_tokens` cap truncating a too-long answer)

Timing: `ttft_s` (ask → first token), `decode_s` (first token → end).
Note: timings include contention if a browser demo session is asking
concurrently — close the tab for clean numbers.

## Result history (2026-07-15, 10 cases per mode)

Historical tables below predate the dual-VLM split, when a single server
served both categories in sequence via `POST /mode` / `POST /model`. Runs
are now per-category against the matching port (weapon on :5000, fall on
:5001); numbers still show weapon vs fall side by side for continuity.


| run | weapon pass | fall pass | notes |
|---|---|---|---|
| baseline | 1/10 | 4/10 | median 2 words, 1-2 word answers, one 180s runaway timeout |
| v1-prompt | 5/10 | 2/10 | strict rules alone; temp 0.9 ignores length limits |
| v1-prompt-gencfg | 6/10 | 6/10 | temp 0.6, rep 1.2, no_repeat_ngram 5, max_new_tokens 120 |
| v2 | 7/10 | 7/10 | example-driven prompt; model stopped copying "Sentence 1:" scaffolding only partially |
| v3-minimal | 8/10 | 9/10 | numeric-free user message: numeric coordinate leaks eliminated |
| v4 (shipped) | 9/10 | 6/10* | shorter example, word bans; *fall failures are 81-82 words vs the 80-word flag — borderline only |

Shipped configuration: the v4 system prompts live in
`frontend/public/prompt-bundles/` (all four languages; fall/en later gained
the keyword-anchor sentence), the `minimal` user template is implemented in
`frontend/app/page.tsx`, and the v2 sampling params live in
`backend_vlm/src/generation_config.json`. Intermediate candidate files are
not kept in the tree — see the git history of this directory for the exact
variants each run used.

## Model sweep (2026-07-15, shipped prompts, 10 cases per mode)

`./sweep_models.sh` — switches each VLM server model via `POST /model` per
run, looping over both categories (weapon on :5000, fall on :5001).

| model | weapon | fall | decode med | notes |
|---|---|---|---|---|
| Qwen2-VL-2B (default) | 8/10 | 9/10 | ~5s | shipped tuning target |
| Qwen3-VL-2B | 10/10 | 9/10 | 3-4s | |
| Qwen3-VL-4B | 10/10 | 10/10 | ~4s | most consistent (16-26 words) |
| Qwen3-VL-8B | 9/10 | 10/10 | 6-7s | concise (14-27 words) |
| aya-vision-8b (before fixes) | 0/10 | 0/10 | 15-16s | see below |
| aya-vision-8b (after fixes) | 9/10 | 9/10* | ~5s | *fall from the aya-soft run |

## Repetition stress test (2026-07-15, 100 uncapped trials per run)

`stress_repetition.py --category weapon|fall` — 4 concurrent Socket.IO
sessions against the matching VLM port (:5000 or :5001), token cap lifted
to 2048, flags exact loops, near-duplicate sentences, token runaway (>300),
slow decode (>30s). Reproduces the intermittent runaway that single browser
sessions rarely surface.

| configuration (Qwen2-VL-2B unless noted) | flagged | worst case |
|---|---|---|
| pre-branch baseline (master prompts/config, roi numbers in input) | **22/100** | 2049-token runaway, 189s |
| shipped v4 stack (rep 1.2 + no_repeat_ngram 5 + temp 0.5) | 0/100 | max 144 tokens |
| shipped stack on Qwen3-VL-2B | 1/100 | 2049-token "synonym chain" (see below) |
| shipped stack on Qwen3-VL-4B / 8B / aya | 0/100 each | max 40 / 35 / 70 tokens |
| **Qwen-recipe (presence 1.5, rep 1.0, temp 0.7/top_p 0.8/top_k 20)** | **0/100** | max 86 tokens |
| Qwen-recipe on Qwen3-VL-2B | **0/100** | max 129 tokens |

The residual failure of the v4 stack is a documented mechanism: banning
exact repeats (repetition_penalty + no_repeat_ngram_size) makes the model
detour through near-synonyms while the penalties also suppress EOS — an
endless "settling dropping descending plunging..." chain. Research
consensus: degenerate repetition is mitigable, never fully preventable
(Holtzman 2020; DITTO 2022; LZ Penalty 2025 measured up to 4% residual for
industry-standard penalties), so a hard `max_new_tokens` cap stays
mandatory as the blast-radius bound.

The shipped shared config is the Qwen-card recipe: additive
`presence_penalty 1.5` via a custom LogitsProcessor (HF `generate()` has no
native one; unlike repetition_penalty it never penalizes EOS), plus
`repetition_penalty 1.0`, temp 0.7 / top_p 0.8 / top_k 20,
`max_new_tokens 150`. It scored best on English (Qwen2 10/10+8/10,
Qwen3-VL-2B 10/10+10/10), killed the synonym chain (0/100 stress on both),
and is 10/10 in Korean on Qwen3-VL-2B. aya-vision-8b keeps its own override
(the recipe is untested there).

Known limitation — Qwen2-VL-2B (default model): the branch eliminated its
runaway repetition (baseline 22/100 → 0/100) but its other quality gains
are limited. Non-English output is weak at the model level regardless of
config (Korean is incoherent under every configuration tested; under the
presence-penalty recipe it also tends to bail out into one-liners, ko fall
5/10 vs 7/10 with the interim rep-1.2 stack — we accepted this rather than
maintaining a Qwen2-specific config). Use Qwen3 models when answer quality
matters, especially outside English.

## Multilingual (ko/ja/zh) results (2026-07-15, 10 cases per mode)

`run_eval.py --category ... --language ko|ja|zh` scores with char-count length bands, CJK
banned words, a character-level repetition pass, and a wrong_language check.

| model | ko (weapon/fall) | ja | zh | verdict |
|---|---|---|---|---|
| Qwen2-VL-2B | 7/4 | 4/5 | 7/6 | non-English unreliable: incoherent Korean, ja mixes zh/en |
| Qwen3-VL-2B | 9/9 | 10/9 | 10/10 | recommended for non-English demos |

Fixes that got ko/ja fall from 4-6/10 to 9/10:

- CJK prompts state an explicit char budget, ban greetings/preambles
  ("물론입니다..."), and demand language purity; fall prompts carry the
  keyword-anchor sentence in every language.
- `max_new_tokens` 90 → 150: Hangul/kana are token-dense (~1.5 tokens per
  char), so 100-char answers were hitting the old cap mid-sentence. English
  regression at the new cap: 10/10 (answers still end naturally; the cap
  stays a runaway bound, now ~10s worst case).

The prompt-quality ceiling for non-English is model-bound: the same prompts
score 2-7/10 on Qwen2-VL-2B and 9-10/10 on Qwen3-VL-2B. Pick the model per
deployment language rather than tuning prompts further.

aya-vision-8b needed two fixes (both shipped):

1. `src/generation_config.json` hardcoded Qwen token ids and was applied to
   every model, so aya never emitted its EOS — every answer ran to the token
   cap and degenerated into mixed-language output. The pipeline now keeps the
   active model's own bos/eos/pad ids.
2. The strong anti-repetition penalties the Qwen models require
   (repetition_penalty 1.2 + no_repeat_ngram_size 5) collapse aya's fluency
   into run-on word salad. `generation_config.<model-name>.json` now
   overrides the shared config per model;
   `generation_config.aya-vision-8b.json` ships softer sampling
   (repetition_penalty 1.05, temperature 0.3, no ngram ban). Trade-off: an
   occasional repetitive answer (~1/10), bounded by max_new_tokens.
