import asyncio
from typing import Any

import pydantic
from tinker import types as tinker_types


class TrajectoryTurn(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    obs: tinker_types.ModelInput
    ac: tinker_types.SampledSequence


class TrajectoryGroup(pydantic.BaseModel):
    group_size: int
    trajectories: list[list[TrajectoryTurn]]
    rewards: list[float]
    judge_calls: int = 0


class PreferenceSample(pydantic.BaseModel):
    prompt: list[dict[str, str]]
    chosen: str
    rejected: str
    row_id: int | None = None


class DPOPair(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    obs: tinker_types.ModelInput
    chosen: tinker_types.SampledSequence
    rejected: tinker_types.SampledSequence


class GRPOConfig(pydantic.BaseModel):
    base_model: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    judge_type: str = "pointwise"
    judge_model: str = "gpt-4.1-2025-04-14"
    load_checkpoint_path: str | None = None
    kl_reference_model: str | None = None
    base_url: str | None = None

    learning_rate: float = 4e-5
    lr_schedule: str = "none"
    lr_warmup_steps: int = 0
    lr_min: float = 0.0
    batch_size: int = 8
    group_size: int = 8
    num_substeps: int = 1
    loss_fn: str = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    kl_coef: float = 0.0
    kl_discount_factor: float = 0.0
    compute_post_kl: bool = False
    max_steps_off_policy: int = 3
    n_steps: int = 100
    resume_optimizer: bool = False

    save_every: int = 20
    log_path: str = "./grpo-results"
    run_name: str | None = None
    ttl_seconds: int | None = 604800
    wandb_project: str | None = None


class DPOConfig(pydantic.BaseModel):
    base_model: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    load_checkpoint_path: str | None = None
    resume_optimizer: bool = False
    reference_model: str | None = None
    reference_model_path: str | None = None
    base_url: str | None = None

    learning_rate: float = 1e-5
    lr_schedule: str = "linear"
    lr_warmup_steps: int = 0
    lr_min: float = 0.0
    beta: float = 0.1
    batch_size: int = 64
    n_steps: int = 100

    save_every: int = 20
    log_path: str = "./dpo-results"
    run_name: str | None = None
    ttl_seconds: int | None = 604800
    wandb_project: str | None = None


TrainingQueue = asyncio.Queue[tuple[int, TrajectoryGroup]]
