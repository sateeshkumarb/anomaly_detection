import logging
import json
import os
import typing
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from datasets.synthetic import TrainingDataset, InferenceDataset
from common.constants import component_mapping, levels_mapping, device
from common.training_functions import train_and_save

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

MODEL_STATE_DIR = "cnn_siamese"
GOLDEN_DISTANCES_STATE_FILE = "golden_distances.pth"
GOLDEN_CENTROID_STATE_FILE = "golden_centroid.pth"
GOLDEN_RATIOS_STATE_FILE = "golden_ratios.json"
MODEL_SAVE_FILE = "model.pth"


class EmbeddingNetwork(nn.Module):

    def __init__(self, raw_field_count, embedding_dim=8):
        super(EmbeddingNetwork, self).__init__()

        # 1. Define Embeddings only for Categorical fields
        # num_embedddings => number of possible values the field can take (say similar to size of vocabulary)
        self.component_embedding = nn.Embedding(len(component_mapping), embedding_dim)
        self.level_embedding = nn.Embedding(len(levels_mapping), embedding_dim)

        # 2. The 1D-CNN
        #  Input channels = 1 raw + 8 + 8  from embeddings = 17
        input_channels = raw_field_count + 2 * embedding_dim
        self.cnn = nn.Sequential(
            # layer-1
            # kernel_size: how many lines to look for (convolution)
            # output shape: [batch_size, out_channels, (input_channels-kernel_size+1)] when padding=0
            nn.Conv1d(
                in_channels=input_channels, out_channels=64, kernel_size=5, padding=0
            ),
            nn.ReLU(),
            # layer-2 no pooling (for now)
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            # layer-3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),  # this doesn't change the data (only the format of it)
            nn.Linear(256, 64),
        )

    def forward(self, x):
        raw_features = x[:, 0:1, :].float()
        comp_idx = x[:, 1, :].long()
        level_idx = x[:, 2, :].long()
        comp_vect = self.component_embedding(comp_idx).permute(0, 2, 1)
        level_vect = self.level_embedding(level_idx).permute(0, 2, 1)
        # Concatenate along feature dim → (batch, window, 17)
        combined = torch.cat([raw_features, comp_vect, level_vect], dim=1)
        out = self.cnn(combined)
        return F.normalize(out, p=2, dim=1)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.embedding_network = EmbeddingNetwork(raw_field_count=1)

    def forward(self, *inputs):
        anchor, positive, negative = inputs
        return (
            self.embedding_network(anchor),
            self.embedding_network(positive),
            self.embedding_network(negative),
        )

    def get_embedding(self, x):
        return self.embedding_network(x)


def save_golden_ratios(
    model: torch.nn.Module,
    golden_dataloader: torch.utils.data.DataLoader,
    dir_to_save_state: str,
):
    model.eval()

    running_sum = None
    total_count = 0

    with torch.no_grad():
        for batch in golden_dataloader:
            golden_batch, _, _ = batch
            logs = golden_batch.to(device)
            embeddings = model.get_embedding(logs)  # Shape: [Batch, 512]

            if running_sum is None:
                running_sum = torch.sum(embeddings, dim=0)
            else:
                running_sum += torch.sum(embeddings, dim=0)

            total_count += embeddings.size(0)
    # Final centroid is the total sum divided by the number of windows
    centroid = running_sum / total_count
    centroid_normalized = F.normalize(centroid, dim=0)
    golden_centroid = centroid_normalized.unsqueeze(0)
    torch.save(golden_centroid, f"{dir_to_save_state}/{GOLDEN_CENTROID_STATE_FILE}")

    golden_distances = []
    with torch.no_grad():
        for batch in golden_dataloader:
            _, golden, _ = batch
            emb = model.get_embedding(golden.to(device))
            # Calculate distance to pre-computed golden centroid
            dist = torch.norm(emb - golden_centroid, p=2, dim=1)
            golden_distances.extend(dist.cpu().tolist())

    # Statistical Profile of "Normal"
    mu = np.mean(golden_distances)
    sigma = np.std(golden_distances)
    threshold = mu + (
        2 * sigma
    )  # The 95% confidence boundary, 3 sigma is 99.7% confidence boundary
    maximum = np.max(golden_distances)
    threshold_99 = np.percentile(golden_distances, 99)
    ratios = {
        "mu": mu,
        "sigma": sigma,
        "max": maximum,
        "threshold": threshold,
        "threshold_99": threshold_99,
    }

    with open(f"{dir_to_save_state}/{GOLDEN_RATIOS_STATE_FILE}", "w") as fp:
        json.dump(ratios, fp)
    # saving golden distances too. Useful for debugging or calculating other ratios
    torch.save(golden_distances, f"{dir_to_save_state}/{GOLDEN_DISTANCES_STATE_FILE}")
    logging.info(f"Normal Baseline (Mean): {mu:.4f}, sigma:{sigma:.4f}")
    logging.info(f"Suggested Anomaly Threshold: {threshold_99:.4f}")


def train(state_root_dir: str) -> None:
    model = SiameseNetwork().to(device)
    triplet_loss = nn.TripletMarginLoss(margin=2.0, p=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    train_dataset = TrainingDataset(for_validation=False)
    train_data_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4
    )
    validation_dataset = TrainingDataset(for_validation=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=32)
    save_dir = f"{state_root_dir}/{MODEL_STATE_DIR}"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = f"{save_dir}/{MODEL_SAVE_FILE}"
    trained_model = train_and_save(
        model,
        train_data_loader,
        validation_data_loader,
        triplet_loss,
        optimizer,
        scheduler,
        model_save_path,
        40,
    )
    save_golden_ratios(trained_model, validation_data_loader, save_dir)


def generate_plots(state_root_dir: str):
    load_dir = f"{state_root_dir}/{MODEL_STATE_DIR}"
    model_path = f"{load_dir}/{MODEL_SAVE_FILE}"
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    validation_dataset = TrainingDataset(for_validation=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=32)
    anchor_centroid = torch.load(f"{load_dir}/{GOLDEN_CENTROID_STATE_FILE}")

    positive_distances = []
    negative_distances = []
    with torch.no_grad():
        for batch in validation_data_loader:
            _, p, n = batch
            p_emb = model.get_embedding(p.to(device))
            p_dist = torch.norm(p_emb - anchor_centroid, p=2, dim=1)
            positive_distances.extend(p_dist.cpu().tolist())

            n_emb = model.get_embedding(n.to(device))
            n_dist = torch.norm(n_emb - anchor_centroid, p=2, dim=1)
            negative_distances.extend(n_dist.cpu().tolist())

    p_mu = np.mean(positive_distances)
    p_sigma = np.std(positive_distances)
    p_maximum = np.max(positive_distances)
    p_99 = np.percentile(positive_distances, 99)

    n_mu = np.mean(negative_distances)
    n_sigma = np.std(negative_distances)
    n_minimum = np.min(negative_distances)
    n_maximum = np.max(negative_distances)
    n_99 = np.percentile(negative_distances, 99)

    logging.info(f"+ve profile- mean:{p_mu},sigma:{p_sigma},p99:{p_99},max:{p_maximum}")
    logging.info(
        f"-ve profile- mean:{n_mu},sigma:{n_sigma},p99:{n_99},min:{n_minimum},max:{n_maximum}"
    )

    positive_distances = np.asarray(positive_distances)
    negative_distances = np.asarray(negative_distances)

    # --- Plot 1: KDE overlay (viewing overlap) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.kdeplot(
        positive_distances,
        ax=axes[0],
        fill=True,
        color="steelblue",
        alpha=0.5,
        label="Positive (normal)",
    )
    sns.kdeplot(
        negative_distances,
        ax=axes[0],
        fill=True,
        color="tomato",
        alpha=0.5,
        label="Negative (anomalous)",
    )

    # mark the means
    axes[0].axvline(
        positive_distances.mean(),
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        label=f"+ve mean: {positive_distances.mean():.3f}",
    )
    axes[0].axvline(
        negative_distances.mean(),
        color="tomato",
        linestyle="--",
        linewidth=1.5,
        label=f"-ve mean: {negative_distances.mean():.3f}",
    )

    axes[0].set_title("Distance Distribution from Anchor Centroid")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # --- Plot 2: Box plot (best for viewing spread and outliers) ---
    df = pd.DataFrame(
        {
            "distance": np.concatenate([positive_distances, negative_distances]),
            "type": ["Positive (normal)"] * len(positive_distances)
            + ["Negative (anomalous)"] * len(negative_distances),
        }
    )
    sns.boxplot(
        data=df,
        x="type",
        y="distance",
        palette={"Positive (normal)": "steelblue", "Negative (anomalous)": "tomato"},
        showfliers=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=df,
        x="type",
        y="distance",
        palette={"Positive (normal)": "steelblue", "Negative (anomalous)": "tomato"},
        alpha=0.4,
        size=4,
        jitter=True,
        ax=axes[1],
    )  # overlay raw points

    # TODO: better understand this
    neg_outlier_threshold = np.percentile(
        positive_distances, 95
    )  # negatives below this are outliers
    pos_outlier_threshold = np.percentile(
        negative_distances, 10
    )  # positives above this are outliers

    n_neg_outliers = (negative_distances < neg_outlier_threshold).sum()
    n_pos_outliers = (positive_distances > pos_outlier_threshold).sum()

    for label, distances, x_pos, outlier_count in [
        ("Positive (normal)", positive_distances, 0, n_pos_outliers),
        ("Negative (anomalous)", negative_distances, 1, n_neg_outliers),
    ]:
        axes[1].text(
            x=x_pos,
            y=2.1,
            # s=f"n={len(distances)}\noutliers={outlier_count}",
            # TODO: include outlier count too in the plot
            s=f"n={len(distances)}",
            ha="center",
            fontsize=9,
        )

    axes[1].set_title("Distance Spread and Outliers")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Distance")

    plt.suptitle(
        "Embedding Space Separation: Normal vs Anomalous",
        fontsize=20,
        fontweight="bold",
        y=0.95,
    )
    plt.tight_layout()
    plt.savefig("distance_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()


def infer(state_root_dir: str, file_paths: typing.List[str]) -> None:
    load_dir = f"{state_root_dir}/{MODEL_STATE_DIR}"
    ratio_path = f"{load_dir}/{GOLDEN_RATIOS_STATE_FILE}"
    with open(ratio_path) as fr:
        vals = json.load(fr)

    mu, sigma, threshold = vals["mu"], vals["sigma"], vals["threshold_99"]
    golden_centroid = torch.load(f"{load_dir}/{GOLDEN_CENTROID_STATE_FILE}")
    model_load_path = f"{load_dir}/{MODEL_SAVE_FILE}"
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_load_path, map_location="cpu"))
    model.to(device)
    model.eval()

    for f in file_paths:
        test_dataset = InferenceDataset(Path(f))
        test_data_loader = DataLoader(test_dataset, batch_size=1)

        abnormal_count = 0
        normal_count = 0

        with torch.no_grad():
            for i, batch in enumerate(test_data_loader):
                batch = batch.to(device)
                embeddings = model.get_embedding(batch)
                distances = torch.norm(embeddings - golden_centroid, p=2, dim=1)
                for d in distances:
                    severity = (d - mu) / sigma
                    pred = get_prediction(d, mu, threshold)
                    if pred > 0.5:
                        logging.info(
                            f"Batch:{i}:ABNORMAL, anomaly probability:{pred:.4f},severity:{severity:.4f}"
                        )
                        abnormal_count += 1
                    else:
                        logging.info(
                            f"Batch:{i}:NORMAL,anomaly probability:{pred:.4f},severity:{severity:.4f}"
                        )
                        normal_count += 1

            logging.info(
                f"Abnormal count:{abnormal_count}, Normal count:{normal_count}"
            )


def get_prediction(dist: float, mu: float, threshold: float) -> float:
    # If distance is near or below the mean, probability is near 0
    if dist <= mu:
        return 0.0

    # Calculate how far past the 'Normal' mean we are
    # This scales the probability so that reaching the 'threshold' is ~50%
    # and doubling the threshold is ~99%
    scaled_dist = (dist - mu) / (threshold - mu)
    # TODO: stronger reasoning to use -0.7 or should we use some other val as constant ?
    # 0.7 because 1 - exp(-0.7) ~= 0.5
    # Don't hardcode 0.7 forever. After you calculate your mu and sigma from
    # the 10k golden lines, run one "Known Bad" batch (your negative logs)
    # through the function.  If the probability for the bad logs is lower
    # than 95%, increase the constant (e.g., to 1.0).  If your golden logs
    # are getting scores higher than 10%, decrease the constant (e.g., to
    # 0.5) or check if your mu is too low (TODO)
    prob = 1 - torch.exp(-0.7 * scaled_dist)  # 0.7 is a tuning constant (sensitivity)
    return prob.item()
