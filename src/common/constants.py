from pathlib import Path
from typing import Dict
import torch

COMPONENTS = ["APP", "DISCOVERY", "HARDWARE", "KERNEL", "MMCS"]
ALL_LEVELS = ["INFO", "WARNING", "ERROR", "SEVERE", "FATAL"]
NORMAL_LEVELS = ["INFO", "WARNING", "ERROR"]
ERROR_LEVELS = ["SEVERE", "FATAL"]

WINDOW_SIZE = 10
THRESHOLD_SECONDS = 300

component_mapping: Dict[str, int] = {
    "APP": 0,
    "DISCOVERY": 1,
    "HARDWARE": 2,
    "KERNEL": 3,
    "MMCS": 4,
}

levels_mapping: Dict[str, int] = {
    "INFO": 0,
    "WARNING": 1,
    "ERROR": 2,
    "SEVERE": 3,
    "FATAL": 4,
}

cwd = Path(__file__).resolve().parent
PROJECT_ROOT = cwd.parent.parent
DATASET_ROOT = f"{PROJECT_ROOT}/data/synthetic"

ANCHOR_FILE_PATH_TRAIN = Path(f"{DATASET_ROOT}/anchors_train.csv")
POSITIVE_FILE_PATH_TRAIN = Path(f"{DATASET_ROOT}/positives_train.csv")
NEGATIVE_FILE_PATH_TRAIN_0 = Path(f"{DATASET_ROOT}/negatives_train_easy.csv")
NEGATIVE_FILE_PATH_TRAIN_1 = Path(f"{DATASET_ROOT}/negatives_train_medium.csv")
NEGATIVE_FILE_PATH_TRAIN_2 = Path(f"{DATASET_ROOT}/negatives_train_hard.csv")
NEGATIVE_FILE_PATH_TRAIN_3 = Path(f"{DATASET_ROOT}/negatives_train_extreme.csv")

ANCHOR_FILE_PATH_VALID = Path(f"{DATASET_ROOT}/anchors_valid.csv")
POSITIVE_FILE_PATH_VALID = Path(f"{DATASET_ROOT}/positives_valid.csv")
NEGATIVE_FILE_PATH_VALID = Path(f"{DATASET_ROOT}/negatives_valid.csv")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
