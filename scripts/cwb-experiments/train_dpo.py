import argparse
import asyncio
import logging
import math
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from data import PreferenceDataLoader
from dotenv import load_dotenv
from eval import main as run_eval
from openai import AsyncOpenAI
from tinker import types as tinker_types
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tourno.logger import setup
from tourno.training.dpo import training_loop
from tourno.training.types import DPOConfig, DPOPair, PreferenceSample

load_dotenv()


def build_dpo_pair(sample: PreferenceSample, renderer: Renderer) -> DPOPair:
    obs = renderer.build_generation_prompt(sample.prompt)

    def _response_tokens(text: str) -> list[int]:
        messages = [*sample.prompt, {"role": "assistant", "content": text}]
        tokens, weights = renderer.build_supervised_example(messages)
        return [t for t, w in zip(tokens.tolist(), weights.tolist()) if w > 0]

    return DPOPair(
        obs=obs,
        chosen=tinker_types.SampledSequence(
            stop_reason="stop", tokens=_response_tokens(sample.chosen)
        ),
        rejected=tinker_types.SampledSequence(
            stop_reason="stop", tokens=_response_tokens(sample.rejected)
        ),
    )


def to_dpo_pair_batches(
    dataloader: PreferenceDataLoader,
    renderer: Renderer,
) -> Iterable[tuple[int, Sequence[DPOPair]]]:
    for epoch, samples in dataloader:
        yield epoch, [build_dpo_pair(s, renderer) for s in samples]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO training on a Creative Writing preference dataset")

    ### Dataset settings ###
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a JSONL preference dataset with prompt/chosen/rejected fields",
    )
    parser.add_argument("--max-samples", type=int, default=None)

    ### Logging settings ###
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--log-filter", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="tourno-cwbench")

    ### Model / optimizer settings ###
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--renderer", type=str, default="qwen3_disable_thinking")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)

    ### DPO loss / reference model settings ###
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--reference-model", type=str, default=None)
    parser.add_argument("--reference-model-path", type=str, default=None)

    ### Training schedule settings ###
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument(
        "--n-epochs",
        type=float,
        default=None,
        help="If set, overrides --n-steps based on dataset size",
    )

    ### Checkpoint / runtime settings ###
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-path", type=str, default="./cwbench-dpo")
    parser.add_argument("--base-url", type=str, default=None)

    ### Evaluation settings ###
    parser.add_argument("--eval-judge-model", type=str, default=None)
    parser.add_argument("--eval-judge-base-url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--eval-judge-api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default=Path("datasets/cwbench_val.jsonl").as_posix(),
    )
    parser.add_argument("--eval-max-samples", type=int, default=64)
    parser.add_argument("--eval-num-completions", type=int, default=1)
    parser.add_argument("--eval-max-tokens", type=int, default=4096)
    parser.add_argument("--eval-temperature", type=float, default=0.7)
    parser.add_argument("--eval-judge-temperature", type=float, default=0.0)
    parser.add_argument("--eval-judge-max-tokens", type=int, default=2048)
    parser.add_argument("--eval-gen-concurrency", type=int, default=32)
    parser.add_argument("--eval-judge-concurrency", type=int, default=128)
    parser.add_argument("--no-eval", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    ### Initialize logging ###
    args = parse_args()
    setup(
        level=getattr(logging, args.log_level.upper()),
        filter_pattern=args.log_filter,
    )

    ### Setup tokenizer and renderer ###
    tokenizer = get_tokenizer(args.base_model)
    renderer_name = args.renderer or get_recommended_renderer_name(args.base_model)
    renderer = get_renderer(renderer_name, tokenizer)

    ### Initialize dataloader (yields PreferenceSample batches) ###
    dataloader = PreferenceDataLoader(
        args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_samples,
    )

    ### Initialize training config ###
    n_steps = args.n_steps
    if args.n_epochs is not None:
        n_steps = math.ceil(args.n_epochs * dataloader.num_rows / args.batch_size)

    run_prefix = "dpo"
    model_short = args.base_model.split("/")[-1]
    dataset_short = Path(args.dataset).stem
    name = (
        f"{model_short}_lr{args.learning_rate}_bs{args.batch_size}_lora{args.lora_rank}_"
        f"beta{args.beta}_data{dataset_short}"
    )

    config = DPOConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        reference_model=args.reference_model,
        reference_model_path=args.reference_model_path,
        base_url=args.base_url,
        learning_rate=args.learning_rate,
        beta=args.beta,
        batch_size=args.batch_size,
        n_steps=n_steps,
        save_every=args.save_every,
        log_path=(Path(args.log_path) / f"{run_prefix}_{name}").as_posix(),
        run_name=f"{run_prefix}_{name}",
        wandb_project=args.wandb_project,
    )

    ### Initialize eval judge client ###
    judge_client: AsyncOpenAI | None = None
    if not args.no_eval and args.eval_judge_model is not None:
        api_key = os.environ.get(args.eval_judge_api_key_env)
        if not api_key:
            raise ValueError(f"Missing judge API key in env var {args.eval_judge_api_key_env}")
        judge_client = AsyncOpenAI(api_key=api_key, base_url=args.eval_judge_base_url or None)

    ### Initialize evaluation callback ###
    async def on_checkpoint_save(step: int, _name: str, sampler_path: str) -> None:
        if args.no_eval or args.eval_judge_model is None:
            return

        assert judge_client is not None
        await run_eval(
            label=f"step_{step:06d}",
            dataset_path=Path(args.eval_dataset),
            output_dir=Path(config.log_path) / "evals",
            base_model=args.base_model,
            sampler_path=sampler_path,
            base_url=args.base_url,
            judge_client=judge_client,
            judge_model=args.eval_judge_model,
            renderer_name=args.renderer,
            max_samples=args.eval_max_samples,
            num_completions=args.eval_num_completions,
            max_tokens=args.eval_max_tokens,
            temperature=args.eval_temperature,
            gen_concurrency=args.eval_gen_concurrency,
            judge_temperature=args.eval_judge_temperature,
            judge_max_tokens=args.eval_judge_max_tokens,
            judge_concurrency=args.eval_judge_concurrency,
        )

    ### Run training: render PreferenceSample batches into DPOPair batches on the fly ###
    train_data = to_dpo_pair_batches(dataloader, renderer)
    asyncio.run(training_loop(config, train_data, on_checkpoint_save))
