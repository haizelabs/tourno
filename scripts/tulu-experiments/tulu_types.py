import pydantic


class TuluSample(pydantic.BaseModel):
    """Schema for a training/val/test prompt drawn from Tulu-3 personas-IF (or WildChat / WildBench)."""

    prompt: list[dict[str, str]]              # [{"role": "user", "content": "..."}]
    prompt_id: str
    user_query: str                            # the raw user message text (for judge)
    history: str = ""                          # multi-turn history (may be empty)
    constraints: list[str] = []                # explicit verifiable constraints (Tulu-3 personas-IF)
    checklist: list[str] = []                  # WildBench-style per-prompt eval checklist (eval-only)
    primary_tag: str = ""                      # category tag (optional)
    row_id: int | None = None
