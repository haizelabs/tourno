import pydantic


class Rubric(pydantic.BaseModel):
    criterion: str
    points: int
    tags: list[str]


class HealthBenchSample(pydantic.BaseModel):
    prompt: list[dict[str, str]]
    prompt_id: str
    rubrics: list[Rubric]
    canary: str
    row_id: int | None = None
