import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from healthbench_types import HealthBenchSample, Rubric
from judges import HealthBenchPointwiseJudge

load_dotenv()

TEST_DATASET_PATH = (
    Path(__file__).resolve().parent.parent.parent / "datasets" / "healthbench_test.jsonl"
)

log = logging.getLogger(__name__)

REPHRASE_PROMPT = """\
Rephrase the following medical response to be approximately {target_words} words.
Keep ALL the same medical information, advice, and content. Do not add any new \
medical facts, recommendations, or caveats. Do not remove any existing medical \
information. Only adjust verbosity, sentence structure, and elaboration level.

Original response:
{completion}

Rephrased response:"""


def load_test_samples(max_samples: int | None = None) -> list[HealthBenchSample]:
    samples: list[HealthBenchSample] = []
    with open(TEST_DATASET_PATH) as f:
        for line in f:
            samples.append(HealthBenchSample.model_validate(json.loads(line)))
    if max_samples:
        samples = samples[:max_samples]
    return samples


def serialize_conversation(prompt: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in prompt)


def normalize_score(raw: float, rubrics: list[Rubric]) -> float:
    pos = sum(r.points for r in rubrics if r.points > 0)
    neg = sum(r.points for r in rubrics if r.points < 0)
    return (raw - neg) / max(1e-4, pos - neg)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Phase 1: sample N completions per question, score each
# ---------------------------------------------------------------------------


async def run_phase1(
    samples: list[HealthBenchSample],
    judge: HealthBenchPointwiseJudge,
    output_dir: Path,
    num_completions: int,
    model: str,
    temperature: float,
    max_tokens: int,
    gen_concurrency: int,
    judge_concurrency: int,
    base_url: str | None,
) -> list[dict]:
    import tinker
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    results_path = output_dir / "phase1_results.jsonl"
    existing = load_jsonl(results_path)
    cached: dict[str, list[dict]] = defaultdict(list)
    for row in existing:
        cached[row["prompt_id"]].append(row)

    needed = [s for s in samples if len(cached[s.prompt_id]) < num_completions]
    if not needed:
        log.info("Phase 1 fully cached")
        return existing

    log.info(f"Phase 1: {len(samples) - len(needed)} cached, {len(needed)} to generate+judge")

    tokenizer = get_tokenizer(model)
    renderer_name = get_recommended_renderer_name(model)
    renderer = get_renderer(renderer_name, tokenizer)

    from eval_common import get_stop_token_ids

    stop_ids = get_stop_token_ids(renderer)

    service = tinker.ServiceClient(base_url=base_url)
    client = await service.create_sampling_client_async(base_model=model)

    gen_sem = asyncio.Semaphore(gen_concurrency)
    judge_sem = asyncio.Semaphore(judge_concurrency)
    gen_done = 0
    judge_done = 0
    total = len(needed)

    async def gen_and_judge(sample: HealthBenchSample) -> None:
        nonlocal gen_done, judge_done

        already = len(cached[sample.prompt_id])
        n = num_completions - already

        async with gen_sem:
            result = await client.sample_async(
                prompt=renderer.build_generation_prompt(sample.prompt),
                num_samples=n,
                sampling_params=tinker.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

        completions: list[tuple[str, int]] = []
        for seq in result.sequences:
            tokens = list(seq.tokens)
            if tokens and tokens[-1] in stop_ids:
                tokens = tokens[:-1]
            completions.append((renderer.tokenizer.decode(tokens), len(tokens)))

        gen_done += 1
        if gen_done % 25 == 0:
            log.info(f"[phase1 gen] {gen_done}/{total}")

        conversation = serialize_conversation(sample.prompt)

        async def judge_one(comp: str, token_count: int) -> None:
            nonlocal judge_done
            async with judge_sem:
                raw_scores = await judge(conversation, [comp], sample.rubrics)
            raw = raw_scores[0]
            norm = normalize_score(raw, sample.rubrics)
            entry = {
                "prompt_id": sample.prompt_id,
                "completion": comp,
                "token_count": token_count,
                "word_count": len(comp.split()),
                "raw_score": raw,
                "normalized_score": norm,
            }
            append_jsonl(results_path, entry)
            judge_done += 1
            if judge_done % 50 == 0:
                log.info(f"[phase1 judge] {judge_done}/{total * num_completions}")

        await asyncio.gather(*[judge_one(c, tc) for c, tc in completions])

    await asyncio.gather(*[gen_and_judge(s) for s in needed])
    log.info(f"Phase 1 done: {gen_done} prompts generated, {judge_done} completions judged")
    return load_jsonl(results_path)


# ---------------------------------------------------------------------------
# Phase 2: rephrase completions to target lengths, re-score
# ---------------------------------------------------------------------------


async def rephrase_completion(
    rephrase_client: AsyncOpenAI,
    rephrase_model: str,
    completion: str,
    target_words: int,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> str:
    for attempt in range(1, max_retries + 2):
        try:
            async with sem:
                resp = await rephrase_client.chat.completions.create(
                    model=rephrase_model,
                    messages=[
                        {
                            "role": "user",
                            "content": REPHRASE_PROMPT.format(
                                target_words=target_words, completion=completion
                            ),
                        }
                    ],
                    timeout=60.0,
                )
                return resp.choices[0].message.content
        except asyncio.CancelledError:
            raise
        except Exception:
            if attempt > max_retries:
                raise
            await asyncio.sleep(min(3 ** (attempt - 1), 60))


async def run_phase2(
    phase1_results: list[dict],
    samples_by_id: dict[str, HealthBenchSample],
    judge: HealthBenchPointwiseJudge,
    output_dir: Path,
    rephrase_model: str,
    length_multipliers: list[float],
    rephrase_concurrency: int,
    judge_concurrency: int,
) -> list[dict]:
    results_path = output_dir / "phase2_results.jsonl"
    existing = load_jsonl(results_path)
    cached_keys: set[tuple[str, float]] = set()
    for row in existing:
        cached_keys.add((row["prompt_id"], row["length_multiplier"]))

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in phase1_results:
        by_prompt[row["prompt_id"]].append(row)

    rephrase_client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    rephrase_sem = asyncio.Semaphore(rephrase_concurrency)
    judge_sem = asyncio.Semaphore(judge_concurrency)

    jobs: list[tuple[str, dict, float]] = []
    for prompt_id, rows in by_prompt.items():
        rows_sorted = sorted(rows, key=lambda r: r["word_count"])
        median_row = rows_sorted[len(rows_sorted) // 2]
        for mult in length_multipliers:
            if (prompt_id, mult) not in cached_keys:
                jobs.append((prompt_id, median_row, mult))

    if not jobs:
        log.info("Phase 2 fully cached")
        return load_jsonl(results_path)

    log.info(f"Phase 2: {len(cached_keys)} cached, {len(jobs)} to rephrase+judge")
    done = 0
    total = len(jobs)

    async def do_one(prompt_id: str, source_row: dict, mult: float) -> None:
        nonlocal done
        original_wc = source_row["word_count"]
        target_words = max(10, int(original_wc * mult))

        if mult == 1.0:
            rephrased = source_row["completion"]
        else:
            rephrased = await rephrase_completion(
                rephrase_client,
                rephrase_model,
                source_row["completion"],
                target_words,
                rephrase_sem,
            )

        sample = samples_by_id[prompt_id]
        conversation = serialize_conversation(sample.prompt)
        async with judge_sem:
            raw_scores = await judge(conversation, [rephrased], sample.rubrics)
        raw = raw_scores[0]
        norm = normalize_score(raw, sample.rubrics)

        entry = {
            "prompt_id": prompt_id,
            "length_multiplier": mult,
            "original_word_count": original_wc,
            "rephrased_word_count": len(rephrased.split()),
            "rephrased_completion": rephrased,
            "raw_score": raw,
            "normalized_score": norm,
        }
        append_jsonl(results_path, entry)
        done += 1
        if done % 25 == 0:
            log.info(f"[phase2] {done}/{total}")

    await asyncio.gather(*[do_one(pid, row, m) for pid, row, m in jobs])
    log.info(f"Phase 2 done: {done} rephrased+judged")
    return load_jsonl(results_path)


# ---------------------------------------------------------------------------
# Phase 3: analysis and plotting
# ---------------------------------------------------------------------------


def analyze_and_plot(
    phase1_results: list[dict],
    phase2_results: list[dict],
    output_dir: Path,
) -> None:
    tokens = np.array([r["token_count"] for r in phase1_results])
    scores = np.array([r["normalized_score"] for r in phase1_results])

    pearson_r, pearson_p = stats.pearsonr(tokens, scores)
    spearman_r, spearman_p = stats.spearmanr(tokens, scores)

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in phase1_results:
        by_prompt[r["prompt_id"]].append(r)

    per_q_spearman = []
    for pid, rows in by_prompt.items():
        if len(rows) < 4:
            continue
        t = [r["token_count"] for r in rows]
        s = [r["normalized_score"] for r in rows]
        if len(set(t)) < 2 or len(set(s)) < 2:
            continue
        rho, _ = stats.spearmanr(t, s)
        if not np.isnan(rho):
            per_q_spearman.append(rho)

    per_q_arr = np.array(per_q_spearman)

    by_mult: dict[float, list[float]] = defaultdict(list)
    for r in phase2_results:
        by_mult[r["length_multiplier"]].append(r["normalized_score"])
    mults_sorted = sorted(by_mult.keys())

    print("\n" + "=" * 60)
    print("LENGTH BIAS ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nPhase 1 — Natural Variation ({len(phase1_results)} completions)")
    print(f"  Pooled Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.2e})")
    print(f"  Pooled Spearman ρ = {spearman_r:.4f}  (p = {spearman_p:.2e})")
    if len(per_q_arr):
        print(f"\n  Per-question Spearman ρ (n={len(per_q_arr)} questions):")
        print(f"    Mean   = {per_q_arr.mean():.4f}")
        print(f"    Median = {np.median(per_q_arr):.4f}")
        print(f"    Std    = {per_q_arr.std():.4f}")
        t_stat, t_p = stats.ttest_1samp(per_q_arr, 0)
        print(f"    t-test vs 0: t={t_stat:.3f}, p={t_p:.2e}")

    if phase2_results:
        print(f"\nPhase 2 — Controlled Rephrasing ({len(phase2_results)} variants)")
        group_scores = []
        for m in mults_sorted:
            arr = np.array(by_mult[m])
            print(
                f"  {m:.2f}x:"
                f" mean={arr.mean():.4f} se={arr.std(ddof=1)/np.sqrt(len(arr)):.4f} (n={len(arr)})"
            )
            group_scores.append(arr)
        if len(group_scores) >= 2:
            kw_stat, kw_p = stats.kruskal(*group_scores)
            print(f"\n  Kruskal-Wallis H={kw_stat:.3f}, p={kw_p:.2e}")
    print("=" * 60 + "\n")

    n_plots = 2 + (1 if phase2_results else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    ax = axes[0]
    ax.scatter(tokens, scores, alpha=0.15, s=8, color="steelblue")
    slope, intercept = np.polyfit(tokens, scores, 1)
    x_fit = np.linspace(tokens.min(), tokens.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color="firebrick", linewidth=2)
    ax.set_xlabel("Token Count")
    ax.set_ylabel("Normalized Score")
    ax.set_title("Length vs Reward (Pooled)")
    ax.text(
        0.05,
        0.95,
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.1e})\nSpearman"
        f" ρ={spearman_r:.3f} (p={spearman_p:.1e})",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if len(per_q_arr):
        ax.hist(per_q_arr, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(
            per_q_arr.mean(),
            color="firebrick",
            linestyle="-",
            linewidth=2,
            label=f"mean={per_q_arr.mean():.3f}",
        )
        ax.set_xlabel("Per-Question Spearman ρ")
        ax.set_ylabel("Count")
        ax.set_title("Within-Question Length-Reward Correlation")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if phase2_results:
        ax = axes[2]
        means = [np.mean(by_mult[m]) for m in mults_sorted]
        ses = [np.std(by_mult[m], ddof=1) / np.sqrt(len(by_mult[m])) for m in mults_sorted]
        ax.errorbar(
            mults_sorted, means, yerr=ses, marker="o", capsize=4, color="steelblue", linewidth=2
        )
        ax.set_xlabel("Length Multiplier")
        ax.set_ylabel("Normalized Score")
        ax.set_title("Rephrasing Experiment\n(same content, different lengths)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "length_bias.png"
    plt.savefig(out_path, dpi=150)
    log.info(f"Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    judge = HealthBenchPointwiseJudge(client=judge_client, model=args.judge_model)

    samples = load_test_samples(args.max_samples)
    samples_by_id = {s.prompt_id: s for s in samples}
    log.info(f"Loaded {len(samples)} test samples")

    phase1_results = await run_phase1(
        samples=samples,
        judge=judge,
        output_dir=output_dir,
        num_completions=args.num_completions,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        gen_concurrency=args.gen_concurrency,
        judge_concurrency=args.judge_concurrency,
        base_url=args.base_url,
    )

    phase2_results = await run_phase2(
        phase1_results=phase1_results,
        samples_by_id=samples_by_id,
        judge=judge,
        output_dir=output_dir,
        rephrase_model=args.rephrase_model,
        length_multipliers=args.length_multipliers,
        rephrase_concurrency=args.rephrase_concurrency,
        judge_concurrency=args.judge_concurrency,
    )

    analyze_and_plot(phase1_results, phase2_results, output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze judge length bias on HealthBench")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-completions", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen-concurrency", type=int, default=32)
    p.add_argument("--judge-concurrency", type=int, default=128)
    p.add_argument("--rephrase-concurrency", type=int, default=32)
    p.add_argument("--judge-model", type=str, default="anthropic/claude-opus-4.5")
    p.add_argument("--rephrase-model", type=str, default="openai/gpt-4.1")
    p.add_argument(
        "--length-multipliers",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0, 1.5, 2.0],
    )
    p.add_argument("--base-url", type=str, default=None, help="Tinker service base URL")
    p.add_argument("--output-dir", type=str, default="healthbench-results/length-bias")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run(args))
