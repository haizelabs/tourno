from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
POINTWISE_PROMPT_PATH = PROMPTS_DIR / "cwbench_pointwise.jinja"
PAIRWISE_PROMPT_PATH = PROMPTS_DIR / "cwbench_pairwise.jinja"
