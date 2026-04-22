import sys
import typing

from common.state import get_root_directory
from data_generator.log_generator import generate_synthetic_logs
from cnn_siamese import model

import logging

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)


def generate_dummy_logs(batch_count, trian):
    generate_synthetic_logs(batch_count, trian)


def train_model():
    state_root_dir = get_root_directory()
    model.train(state_root_dir)


def infer(file_paths: typing.List):
    state_root_dir = get_root_directory()
    model.infer(state_root_dir, file_paths)


def main():
    # generate_synthetic_logs()
    # train_model()
    fpaths = [
        # "../data/synthetic/negatives_test.csv",
        # "../data/synthetic/positives_test.csv",
        "../data/synthetic/trial.csv",
    ]
    infer(fpaths)


if __name__ == "__main__":
    main()
