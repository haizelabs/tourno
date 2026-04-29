import pydantic


class TruthfulQASample(pydantic.BaseModel):
    prompt: list[dict[str, str]]
    prompt_id: str
    best_answer: str
    correct_answers: list[str]
    incorrect_answers: list[str]
    category: str
    row_id: int | None = None
