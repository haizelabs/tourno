"""Build per-model 'Style Profile' word clouds for cwbench v3 evals.

For each of 3 models we sample up to 50 completions, ask Sonnet 4.5 (via
OpenRouter) for 5-10 single-word stylistic descriptors of the prose, aggregate
the descriptor frequencies, and render a wordcloud PNG.

Outputs:
  /Users/bhavesh/Developer/tourno/cwbench-rl/eval/figures/style_<model>.png
  /Users/bhavesh/Developer/tourno/cwbench-rl/eval/style_descriptors.json  (cache)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import AsyncOpenAI
from wordcloud import WordCloud

load_dotenv()

EVAL_DIR = Path("/Users/bhavesh/Developer/tourno/cwbench-rl/eval")
FIG_DIR = EVAL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = EVAL_DIR / "style_descriptors.json"

MODELS: dict[str, str] = {
    "base": "Qwen3-8B (base)",
    "pointwise": "Pointwise GRPO",
    "pairwise": "Pairwise GRPO",
    "tourno": "TournO mixture",
}

EVAL_FILENAMES: dict[str, str] = {
    "base": "base.json",
    "pointwise": "pointwise.json",
    "pairwise": "pairwise.json",
    "tourno": "tourno-step120.json",
}

JUDGE_MODEL = "anthropic/claude-sonnet-4.5"
SAMPLES_PER_MODEL = 50
MAX_CONCURRENCY = 12
SEED = 7
MAX_COMPLETION_CHARS = 6000  # truncate very long completions for the judge

JUDGE_SYSTEM = (
    "You are a literary stylometer. Your sole job is to read a writing sample "
    "and label its stylistic register with single-word adjectives."
)

JUDGE_USER_TEMPLATE = """Read the following creative-writing sample and pick \
5-10 single-word stylistic descriptors that capture its prose register. Use \
common adjectives a critic would use about style/tone (e.g. lyrical, sparse, \
vivid, vibrant, moody, formulaic, restrained, ornate, breathless, plain, \
chaotic, playful, grim, melodramatic, terse, rambling, mannered).

Constraints:
- Single English words only, lowercase, no spaces or punctuation.
- 5 to 10 words total.
- Describe HOW it is written, not WHAT happens.
- Output ONLY a JSON array of strings, e.g. ["lyrical","moody","ornate"]

WRITING SAMPLE:
\"\"\"
{sample}
\"\"\"

JSON array:"""


def load_completions(model_key: str) -> list[str]:
    with open(EVAL_DIR / EVAL_FILENAMES[model_key], "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for row in data["rows"]:
        for it in row["iterations"]:
            c = (it.get("completion") or "").strip()
            if len(c) >= 200:
                out.append(c)
    return out


def sample_completions(completions: list[str], n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    if len(completions) <= n:
        return completions
    return rng.sample(completions, n)


def parse_descriptors(text: str) -> list[str]:
    """Extract a JSON array of words from a possibly chatty model response."""
    if not text:
        return []
    # Try fenced JSON first
    m = re.search(r"\[[^\[\]]*\]", text, flags=re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for w in arr:
        if not isinstance(w, str):
            continue
        w = w.strip().lower()
        # Keep only single-word a-z descriptors
        if re.fullmatch(r"[a-z][a-z\-]{1,24}", w):
            out.append(w)
    return out[:10]


async def judge_one(client: AsyncOpenAI, sample: str,
                    sem: asyncio.Semaphore) -> list[str]:
    sample = sample[:MAX_COMPLETION_CHARS]
    prompt = JUDGE_USER_TEMPLATE.format(sample=sample)
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=120,
                    timeout=90,
                )
                txt = resp.choices[0].message.content or ""
                desc = parse_descriptors(txt)
                if desc:
                    return desc
            except Exception as e:
                if attempt == 2:
                    print(f"  judge_one failed after retries: {e!r}")
                    return []
                await asyncio.sleep(1.5 * (attempt + 1))
        return []


async def gather_descriptors_for_model(
    client: AsyncOpenAI, samples: list[str]
) -> list[list[str]]:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [judge_one(client, s, sem) for s in samples]
    results: list[list[str]] = []
    done = 0
    for fut in asyncio.as_completed(tasks):
        desc = await fut
        results.append(desc)
        done += 1
        if done % 10 == 0:
            print(f"  ...{done}/{len(tasks)} judged")
    return results


async def collect_all(cache: dict) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (check .env)")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    for key in MODELS:
        if cache.get(key, {}).get("counts"):
            print(f"[{key}] cached -> {len(cache[key]['counts'])} unique words")
            continue
        print(f"[{key}] sampling completions...")
        completions = load_completions(key)
        samples = sample_completions(completions, SAMPLES_PER_MODEL, SEED)
        print(f"[{key}] judging {len(samples)} samples with {JUDGE_MODEL}")
        descs = await gather_descriptors_for_model(client, samples)
        flat = [w for ws in descs for w in ws]
        counter = Counter(flat)
        cache[key] = {
            "model": key,
            "n_samples": len(samples),
            "raw": descs,
            "counts": dict(counter.most_common()),
        }
        # Persist after each model so partial progress survives crashes
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        print(f"[{key}] {len(counter)} unique words; "
              f"top: {counter.most_common(8)}")
    return cache


# Per-model accent palette to tint the style cloud — matches the rest of the paper.
# Per-model accent palette derived from the canonical 4-color paper scheme.
# Each list = darkened shades of the model's pastel for word-cloud contrast on white.
MODEL_ACCENTS = {
    "base":      ["#a8a59c", "#8f8c83", "#6e6b62", "#bcb9b0"],
    "pointwise": ["#c8b568", "#a89241", "#7e6a26", "#dac98c"],
    "pairwise":  ["#a3a86b", "#828747", "#5d6228", "#b9bd8b"],
    "tourno":    ["#9a7e6c", "#7d6452", "#5a4538", "#b39788"],
}


def make_color_func(model_key: str):
    palette = MODEL_ACCENTS.get(model_key, ["#444444"])

    def color_func(word, font_size, position, orientation, random_state=None,
                   **kwargs):
        rng = random.Random(hash(word) & 0xffffffff)
        return rng.choice(palette)
    return color_func


def render_wordcloud(model_key: str, display_name: str,
                     counts: dict[str, int]) -> Path:
    if not counts:
        print(f"[{model_key}] no descriptors, skipping wordcloud")
        return FIG_DIR / f"cwb_style_{model_key}.png"

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        prefer_horizontal=0.85,
        relative_scaling=0.55,
        min_font_size=12,
        max_font_size=240,
        max_words=120,
        collocations=False,
        random_state=42,
        color_func=make_color_func(model_key),
    ).generate_from_frequencies(counts)

    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor="white")
    ax.imshow(wc, interpolation="bilinear")
    ax.set_axis_off()
    out_png = FIG_DIR / f"cwb_style_{model_key}.png"
    out_pdf = FIG_DIR / f"cwb_style_{model_key}.pdf"
    fig.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    cache: dict = {}
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            cache = {}

    cache = asyncio.run(collect_all(cache))

    out_paths: list[Path] = []
    for key, display in MODELS.items():
        counts = cache.get(key, {}).get("counts", {})
        out = render_wordcloud(key, display, counts)
        out_paths.append(out)
        top = list(counts.items())[:8]
        print(f"wrote {out}  (top: {top})")

    print(f"\n{len(out_paths)} style figures written.")


if __name__ == "__main__":
    main()
