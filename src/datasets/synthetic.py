from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torch
import math

from common.constants import (
    WINDOW_SIZE,
    THRESHOLD_SECONDS,
    component_mapping,
    levels_mapping,
    ANCHOR_FILE_PATH_TRAIN,
    POSITIVE_FILE_PATH_TRAIN,
    NEGATIVE_FILE_PATH_TRAIN,
    ANCHOR_FILE_PATH_VALID,
    POSITIVE_FILE_PATH_VALID,
    NEGATIVE_FILE_PATH_VALID,
)


class LogDataset(Dataset):
    def _transform(self, v: pd.Series, prev_ts=None) -> torch.Tensor:
        current_ts = float(v["timestamp"].item())
        c = v["component"].item()
        lv = v["level"].item()

        if prev_ts is None:
            delta_ts = THRESHOLD_SECONDS
        else:
            delta_ts = current_ts - prev_ts

        log_delta = math.log(1 + delta_ts)  # * 10.0  # amplify this signal
        # ensure that the data we are using has monotonically increasing timestamp
        if log_delta < 0:
            assert False

        outs = [
            log_delta,
            component_mapping[c],
            levels_mapping[lv],
        ]
        return torch.tensor(outs)

    def _transform_frames(self, f):
        transformed = []
        for i in range(len(f)):
            if i == 0:
                prev_ts = None
            else:
                prev_ts = float(f.iloc[i - 1]["timestamp"])

            f_t = self._transform(f.iloc[i : i + 1], prev_ts)
            transformed.append(f_t)
        # shape is [Sequence_length, channels]
        # but Conv1d expects [channels, Sequence_length], hence permute
        return torch.stack(transformed).permute(1, 0)


class TrainingDataset(LogDataset):
    def __init__(self, for_validation=False):
        if for_validation:
            anchor_df = pd.read_csv(ANCHOR_FILE_PATH_VALID, comment="#")
            positive_df = pd.read_csv(POSITIVE_FILE_PATH_VALID, comment="#")
            negative_df = pd.read_csv(NEGATIVE_FILE_PATH_VALID, comment="#")
        else:
            anchor_df = pd.read_csv(ANCHOR_FILE_PATH_TRAIN, comment="#")
            positive_df = pd.read_csv(POSITIVE_FILE_PATH_TRAIN, comment="#")
            negative_df = pd.read_csv(NEGATIVE_FILE_PATH_TRAIN, comment="#")

        # each entry is a 2D array of in batches of WINDOW_SIZE log lines
        self.anchors = [
            anchor_df[i : i + WINDOW_SIZE]
            for i in range(0, len(anchor_df), WINDOW_SIZE)
        ]

        self.positives = [
            positive_df[i : i + WINDOW_SIZE]
            for i in range(0, len(positive_df), WINDOW_SIZE)
        ]
        self.negatives = [
            negative_df[i : i + WINDOW_SIZE]
            for i in range(0, len(negative_df), WINDOW_SIZE)
        ]

    def __getitem__(self, index: int):
        a, p, n = (
            self.anchors[index],
            self.positives[index],
            self.negatives[index],
        )

        a_t = self._transform_frames(a)
        p_t = self._transform_frames(p)
        n_t = self._transform_frames(n)
        return a_t, p_t, n_t

    def __len__(self):
        return len(self.anchors)


class InferenceDataset(LogDataset):
    def __init__(self, file_path: Path):
        df = pd.read_csv(file_path, comment="#")
        self.vals = [df[i : i + WINDOW_SIZE] for i in range(0, len(df), WINDOW_SIZE)]

    def __getitem__(self, index: int):
        v = self.vals[index]
        return self._transform_frames(v)

    def __len__(self):
        return len(self.vals)
