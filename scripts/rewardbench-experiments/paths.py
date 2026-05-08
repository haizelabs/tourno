from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
DATASETS_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"

POLICY_POINTWISE_PROMPT_PATH = PROMPTS_DIR / "rewardbench2_policy_pointwise.txt"
META_POINTWISE_PROMPT_PATH = PROMPTS_DIR / "rewardbench2_meta_pointwise.txt"
META_PAIRWISE_PROMPT_PATH = PROMPTS_DIR / "rewardbench2_meta_pairwise.txt"

TRAIN_DATASET_PATH = (DATASETS_DIR / "rewardbench2_train.jsonl").as_posix()
VAL_DATASET_PATH = (DATASETS_DIR / "rewardbench2_val.jsonl").as_posix()
TEST_DATASET_PATH = (DATASETS_DIR / "rewardbench2_test.jsonl").as_posix()
