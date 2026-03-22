from typing import Any

import tinker

from pioneer.logger import get_logger


async def get_sampling_client(
    *,
    training_client: tinker.TrainingClient | None = None,
    base_model: str | None = None,
    load_checkpoint_path: str | None = None,
    base_url: str | None = None,
) -> tinker.SamplingClient:
    if training_client:
        return await training_client.save_weights_and_get_sampling_client_async()
    elif base_model:
        service_client = tinker.ServiceClient(base_url=base_url)
        return await service_client.create_sampling_client_async(base_model=base_model)
    elif load_checkpoint_path:
        service_client = tinker.ServiceClient(base_url=base_url)
        return await service_client.create_sampling_client_async(model_path=load_checkpoint_path)
    else:
        raise ValueError(
            "At least one of training_client, base_model, or load_checkpoint_path must be provided"
        )


async def get_training_client(
    base_model: str,
    *,
    lora_rank: int = 32,
    load_checkpoint_path: str | None = None,
    resume_optimizer: bool = False,
    base_url: str | None = None,
    seed: int = 42,
    train_mlp: bool = True,
    train_attn: bool = True,
    train_unembed: bool = True,
    user_metadata: dict[str, Any] | None = None,
) -> tinker.TrainingClient:
    log = get_logger()
    service_client = tinker.ServiceClient(base_url=base_url)

    if load_checkpoint_path and resume_optimizer:
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                load_checkpoint_path
            )
        )
        log.info(f"Resumed training (with optimizer) from {load_checkpoint_path}")
    elif load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            load_checkpoint_path
        )
        log.info(f"Loaded weights from {load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=base_model,
            rank=lora_rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
            user_metadata=user_metadata,
        )

    return training_client
