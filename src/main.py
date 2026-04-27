import sys
from typing import Annotated

from common.state import get_root_directory
from data_generator.log_generator import generate_synthetic_logs
from cnn_siamese import model

import typer

import logging

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

app = typer.Typer()


@app.command()
def generate_dummy_logs(
    batch_count: Annotated[int, typer.Argument(help="number of batches")],
    for_validation: Annotated[
        bool, typer.Argument(help="logs to be used for validation")
    ] = False,
):
    """Generate synthetic logs. The logs will be generated under <project_root>/synthetic. Three sets of logs: anchors_(train|valid).csv, positives_(train|valid).csv,
    and negatives_(train|valid).csv will be generated. Size of each batch will be 10 and total number of log lines will be 10 * batch_count.
    """
    generate_synthetic_logs(batch_count, for_validation)


@app.command()
def train_model(
    state_root_dir: Annotated[
        str,
        typer.Argument(
            help="root directory under which trained model state and related configuration will be stored"
        ),
    ] = None,
):
    """Train the model using the training and validation log samples available under <project_root>/synthetic. State of the trained model and
    other learned parameters will be saved under specified state_root_dir, if one is specified. If no directory is specified the trained model state
    and related parameters will be saved under <project_root>/saved_states.
    """
    if state_root_dir is None:
        state_root_dir = get_root_directory()
    model.train(state_root_dir)


@app.command()
def infer(
    file_path: Annotated[
        str,
        typer.Argument(
            help="complete path of file which contains logs to be classified"
        ),
    ],
    state_root_dir: Annotated[
        str,
        typer.Argument(
            help="root directory to read the saved model and the related configuration"
        ),
    ] = None,
):
    """Run inference on the batch of logs contained in given file path. The inference will be run on batch of 10 logs at a time, and classification result will be on each batch (of 10) logs.
    State of the trained model and other necessary parameters will be loaded from specified state_root_dir, if one is specified. If no directory is specified the trained model state
    and related parameters will be loaded from <project_root>/saved_states."""
    file_path = file_path.strip()

    if state_root_dir is None:
        state_root_dir = get_root_directory()
    logging.info(f"classifying logs in file:{file_path}")
    model.infer(state_root_dir, [file_path])


@app.command()
def show_plots(
    state_root_dir: Annotated[
        str,
        typer.Argument(
            help="root directory to read the saved model and the related configuration"
        ),
    ] = None,
):
    """Generate and show Kernel Density Estimation (KDE) and box plots using the trained model. These models show distance spread and outliers of samples in validation dataset compared to the
    anchor Centroid."""
    if state_root_dir is None:
        state_root_dir = get_root_directory()
    model.generate_plots(state_root_dir)


if __name__ == "__main__":
    app()
