from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
POINTWISE_PROMPT_PATH = PROMPTS_DIR / "healthbench_rubric_pointwise_score.txt"
POINTWISE_TRAIN_PROMPT_PATH = PROMPTS_DIR / "healthbench_holistic_pointwise_score_0_20.txt"
PAIRWISE_PROMPT_PATH = PROMPTS_DIR / "healthbench_rubric_pairwise_choice.txt"
PAIRWISE_TRAIN_PROMPT_PATH = PROMPTS_DIR / "healthbench_holistic_pairwise_margin.txt"
