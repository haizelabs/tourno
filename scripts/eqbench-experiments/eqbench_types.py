from typing import Literal

import pydantic

TaskType = Literal["standard", "analysis"]


class EQBenchSample(pydantic.BaseModel):
    prompt: list[dict[str, str]]
    prompt_id: str
    scenario_id: str
    sub_prompt_index: int
    task_type: TaskType
    scenario_text: str
    row_id: int | None = None
