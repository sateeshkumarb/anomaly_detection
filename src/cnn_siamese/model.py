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

from common.visualizations import distance_distribution, tsne
from datasets.synthetic import TrainingDataset, InferenceDataset
from common.constants import component_mapping, levels_mapping, device
from common.training_functions import train_and_save, get_anchor_centroid

MODEL_STATE_DIR = "cnn_siamese"
GOLDEN_DISTANCES_STATE_FILE = "golden_distances.pth"
GOLDEN_CENTROID_STATE_FILE = "golden_centroid.pth"
GOLDEN_RATIOS_STATE_FILE = "golden_ratios.json"
MODEL_SAVE_FILE = "model.pth"


class EmbeddingNetwork(nn.Module):

    def __init__(self, raw_field_count, embedding_dim=8, pool_size=1):
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
            # input shape : [batch_size, input_channels, seq_length]
            # output shape: [batch_size, out_channels, (seq_length-kernel_size+1)] when padding=0
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
            nn.AdaptiveMaxPool1d(pool_size),
            nn.Flatten(),  # this doesn't change the data (only the format of it)
            # if there was no pooling in_features: 256*sequence_length after last Conv1d
            nn.Linear(256 * pool_size, 64),
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
    golden_centroid = get_anchor_centroid(model, golden_dataloader)
    torch.save(golden_centroid, f"{dir_to_save_state}/{GOLDEN_CENTROID_STATE_FILE}")

    model.eval()
    golden_distances = []
    with torch.no_grad():
        for batch in golden_dataloader:
            _, golden, _ = batch
            # model already returns normalized embeddings, hence no need to normalize
            # these embeddings while computing L2 distance from the anchor centroid
            emb = model.get_embedding(golden.to(device))
            # guard against any change in the model code and normalization isn't done anymore
            assert torch.allclose(
                torch.linalg.vector_norm(emb, ord=2, dim=1),
                torch.ones(emb.size(0), device=device),
                atol=1e-5,
            ), "Embeddings are not unit normalised — check model forward"

            # Calculate L2 distance to pre-computed golden centroid
            # for each sample in the batch.
            # L2 distance is calculated as:
            # distance = sqrt(sum((emb[i] - centroid) ^ 2))
            dist = torch.linalg.vector_norm(emb - golden_centroid, ord=2, dim=1)
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


def _load_model(state_root_dir: str) -> nn.Module:
    load_dir = f"{state_root_dir}/{MODEL_STATE_DIR}"
    model_path = f"{load_dir}/{MODEL_SAVE_FILE}"
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    return model


def plot_distances(state_root_dir: str):
    load_dir = f"{state_root_dir}/{MODEL_STATE_DIR}"
    model = _load_model(state_root_dir)

    validation_dataset = TrainingDataset(for_validation=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=32)
    anchor_centroid = torch.load(f"{load_dir}/{GOLDEN_CENTROID_STATE_FILE}")

    positive_distances = []
    negative_distances = []
    with torch.no_grad():
        for batch in validation_data_loader:
            _, p, n = batch
            p_emb = model.get_embedding(p.to(device))
            p_dist = torch.linalg.vector_norm(p_emb - anchor_centroid, ord=2, dim=1)
            positive_distances.extend(p_dist.cpu().tolist())

            n_emb = model.get_embedding(n.to(device))
            n_dist = torch.linalg.vector_norm(n_emb - anchor_centroid, ord=2, dim=1)
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
    distance_distribution(positive_distances, negative_distances)


def plot_embeddings(state_root_dir: str):
    model = _load_model(state_root_dir)

    validation_dataset = TrainingDataset(for_validation=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=32)

    embeddings = []
    labels = []
    with torch.no_grad():
        for anchor, positive, negative in validation_data_loader:
            anchor_emb = model.get_embedding(anchor.to(device)).squeeze(-1)
            positive_emb = model.get_embedding(positive.to(device)).squeeze(-1)
            negative_emb = model.get_embedding(negative.to(device)).squeeze(-1)

            batch_size = anchor.size(0)
            embeddings.append(anchor_emb.cpu().numpy())
            embeddings.append(positive_emb.cpu().numpy())
            embeddings.append(negative_emb.cpu().numpy())

            labels.extend(["Anchor (normal)"] * batch_size)
            labels.extend(["Positive (normal)"] * batch_size)
            labels.extend(["Negative (anomalous)"] * batch_size)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)
    tsne(embeddings, labels)


def infer(state_root_dir: str, file_paths: typing.List[str]) -> None:
    load_dir = f"{state_root_dir}/{MODEL_STATE_DIR}"
    ratio_path = f"{load_dir}/{GOLDEN_RATIOS_STATE_FILE}"
    with open(ratio_path) as fr:
        vals = json.load(fr)

    mu, sigma, threshold = vals["mu"], vals["sigma"], vals["threshold_99"]
    golden_centroid = torch.load(f"{load_dir}/{GOLDEN_CENTROID_STATE_FILE}")
    model = _load_model(state_root_dir)

    for f in file_paths:
        test_dataset = InferenceDataset(Path(f))
        test_data_loader = DataLoader(test_dataset, batch_size=1)

        abnormal_count = 0
        normal_count = 0

        with torch.no_grad():
            for i, batch in enumerate(test_data_loader):
                batch = batch.to(device)
                embeddings = model.get_embedding(batch)
                distances = torch.linalg.vector_norm(
                    embeddings - golden_centroid, ord=2, dim=1
                )
                for d in distances:
                    # z-score. How many standard deviations the batch distance is from the normal mean.
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
    # dist: L2 norm of distance of embeddings of given batch of logs from normalized anchor centroid.
    # mu: mean of L2 norm of distance of embeddings of +ve samples (of validation set) from normalized anchor centroid.
    # threshold: 99th percentile of vector norm of distance of the embeddings of +ve samples (of validation set)
    # from normalized anchor centroid.
    # If distance is near or below the mean, probability is near 0
    if dist <= mu:
        return 0.0

    # if dist = threshold. we are outer edge of what can be considered normal. The probability evaluates
    # to ~0.5
    scaled_dist = (dist - mu) / (threshold - mu)
    # 0.7: exp(-0.7) ~= 0.5. So when scaled dist is 1 (which happens when dist = threshold) probability
    # will be 0.5 As the difference increases probability will also scale accordingly
    prob = 1 - torch.exp(-0.7 * scaled_dist)
    return prob.item()
