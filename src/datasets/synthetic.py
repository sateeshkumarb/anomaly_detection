import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torch

from common.constants import (
    WINDOW_SIZE,
    THRESHOLD_SECONDS,
    component_mapping,
    levels_mapping,
    ANCHOR_FILE_PATH_TRAIN,
    POSITIVE_FILE_PATH_TRAIN,
    NEGATIVE_FILE_PATH_TRAIN_0,
    ANCHOR_FILE_PATH_VALID,
    POSITIVE_FILE_PATH_VALID,
    NEGATIVE_FILE_PATH_VALID,
    NEGATIVE_FILE_PATH_TRAIN_1,
    NEGATIVE_FILE_PATH_TRAIN_2,
    NEGATIVE_FILE_PATH_TRAIN_3,
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

        log_delta = np.log1p(delta_ts)
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
        self._validation_dataset = for_validation
        if for_validation:
            anchor_df = pd.read_csv(ANCHOR_FILE_PATH_VALID, comment="#")
            positive_df = pd.read_csv(POSITIVE_FILE_PATH_VALID, comment="#")
            negative_df_0 = pd.read_csv(NEGATIVE_FILE_PATH_VALID, comment="#")
        else:
            anchor_df = pd.read_csv(ANCHOR_FILE_PATH_TRAIN, comment="#")
            positive_df = pd.read_csv(POSITIVE_FILE_PATH_TRAIN, comment="#")
            negative_df_0 = pd.read_csv(NEGATIVE_FILE_PATH_TRAIN_0, comment="#")
            negative_df_1 = pd.read_csv(NEGATIVE_FILE_PATH_TRAIN_1, comment="#")
            negative_df_2 = pd.read_csv(NEGATIVE_FILE_PATH_TRAIN_2, comment="#")
            negative_df_3 = pd.read_csv(NEGATIVE_FILE_PATH_TRAIN_3, comment="#")

        # each entry is a 2D array of in batches of WINDOW_SIZE log lines
        self.anchors = [
            anchor_df[i : i + WINDOW_SIZE]
            for i in range(0, len(anchor_df), WINDOW_SIZE)
        ]
        self.positives = [
            positive_df[i : i + WINDOW_SIZE]
            for i in range(0, len(positive_df), WINDOW_SIZE)
        ]

        self.negatives_0 = [
            negative_df_0[i : i + WINDOW_SIZE]
            for i in range(0, len(negative_df_0), WINDOW_SIZE)
        ]

        if not for_validation:
            self.negatives_1 = [
                negative_df_1[i : i + WINDOW_SIZE]
                for i in range(0, len(negative_df_1), WINDOW_SIZE)
            ]
            self.negatives_2 = [
                negative_df_2[i : i + WINDOW_SIZE]
                for i in range(0, len(negative_df_2), WINDOW_SIZE)
            ]
            self.negatives_3 = [
                negative_df_3[i : i + WINDOW_SIZE]
                for i in range(0, len(negative_df_3), WINDOW_SIZE)
            ]

    def __getitem__(self, index: int):
        if self._validation_dataset:
            a, p, n = (
                self.anchors[index],
                self.positives[index],
                self.negatives_0[index],
            )
            a_t = self._transform_frames(a)
            p_t = self._transform_frames(p)
            n_t = self._transform_frames(n)
            return a_t, p_t, n_t
        else:
            a, p, n0, n1, n2, n3 = (
                self.anchors[index],
                self.positives[index],
                self.negatives_0[index],
                self.negatives_1[index],
                self.negatives_2[index],
                self.negatives_3[index],
            )

            a_t = self._transform_frames(a)
            p_t = self._transform_frames(p)
            n_t_0 = self._transform_frames(n0)
            n_t_1 = self._transform_frames(n1)
            n_t_2 = self._transform_frames(n2)
            n_t_3 = self._transform_frames(n3)
            return a_t, p_t, n_t_0, n_t_1, n_t_2, n_t_3

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
