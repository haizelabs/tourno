# Tulu IF DPO Data Generation

Generate completions across multiple models, then judge them pairwise with
Claude Sonnet 4.5 to construct DPO preference pairs.

## Pipeline

1. **`generate.py`** — instruction-tuned models via OpenRouter:
   `qwen/qwen3-14b`, `google/gemma-4-31b-it`, `google/gemini-3-flash-preview`,
   `openai/gpt-5.1`.
2. **`generate_tinker.py`** — pretrained-only base models via Tinker:
   `meta-llama/Llama-3.1-8B`, `deepseek-ai/DeepSeek-V3.1-Base`. (OpenRouter
   does not host pretrained-only models.)
3. **`build_dpo.py`** — for each prompt, runs a round-robin tournament with
   Sonnet 4.5 as judge over **all** candidates: the 6 generated completions
   plus Tulu's existing gpt-4o `chosen` and gpt-4o `rejected` (the latter was
   constructed to deliberately violate one constraint, so it functions as a
   strong negative). Picks highest win-rate as `chosen`, lowest as `rejected`.

   Output schema is a superset of `tourno.training.types.PreferenceSample` for
   direct use with `train_dpo.py`.

Both generate scripts stream to disk and are resumable — re-running skips
`(id, model)` pairs already on disk.

## Candidates per prompt

```
qwen/qwen3-14b                          # ~14B, instruct
google/gemma-4-31b-it                   # ~31B, instruct
google/gemini-3-flash-preview           # hosted, instruct
openai/gpt-5.1                          # hosted, instruct
meta-llama/Llama-3.1-8B                 # ~8B, base/pretrained (Tinker)
deepseek-ai/DeepSeek-V3.1-Base          # base/pretrained (Tinker)
openai/gpt-4o (tulu chosen)             # already in dataset, satisfies all constraints
openai/gpt-4o (tulu rejected)           # already in dataset, violates one constraint
```

→ 8 tournament candidates per prompt, 28 pairwise comparisons per round-robin.

## Usage

```bash
export OPENROUTER_API_KEY=sk-...
export TINKER_API_KEY=...

# 1. Instruct models on OpenRouter (run once per split)
uv run --env-file .env scripts/tulu-experiment/generate.py \
    --input  datasets/tulu_if_train.jsonl \
    --output datasets/dpo/train_completions.jsonl

# 2. Base models on Tinker (writes to the same JSONL)
uv run --env-file .env scripts/tulu-experiment/generate_tinker.py \
    --input  datasets/tulu_if_train.jsonl \
    --output datasets/dpo/train_completions.jsonl \
    --models meta-llama/Llama-3.1-8B deepseek-ai/DeepSeek-V3.1-Base

# 3. Tournament + DPO pairs
uv run --env-file .env scripts/tulu-experiment/build_dpo.py \
    --completions datasets/dpo/train_completions.jsonl \
    --samples     datasets/tulu_if_train.jsonl \
    --output      datasets/dpo/train.jsonl
```

Repeat for `val` and `test` splits with the corresponding paths.

## Concurrency

- `generate.py`: 256 global / 64 per-model concurrent OpenRouter calls.
- `generate_tinker.py`: 32 concurrent Tinker `sample_async` calls per model.
- `build_dpo.py`: 256 concurrent Sonnet judge calls; tournaments across all
  prompts run in parallel.

Tune via `--global-concurrency` / `--per-model-concurrency` /
`--judge-concurrency` / `--concurrency` flags if you hit rate limits.
