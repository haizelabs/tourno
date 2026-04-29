import pydantic


class CreativeBenchSample(pydantic.BaseModel):
    prompt: list[dict[str, str]]
    prompt_id: str
    scenario_id: str
    seed_index: int
    category: str
    title: str
    writing_prompt: str
    row_id: int | None = None
