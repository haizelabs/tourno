from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
POINTWISE_PROMPT_PATH = PROMPTS_DIR / "healthbench_pointwise_judge.txt"
PAIRWISE_PROMPT_PATH = PROMPTS_DIR / "healthbench_pairwise_judge.txt"
