import logging
import time

import torch
from tqdm import tqdm

from common.constants import device
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def train_and_save(
    model,
    train_loader,
    val_loader,
    loss_fcn,
    optimizer,
    scheduler,
    save_path=None,
    n_epochs=10,
):
    model.to(device)
    best_overlap = float("inf")
    best_separation_ratio = 0.0

    logging.info("Starting Training and Validation")
    start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_samples_processed = 0
        train_progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{n_epochs} [Training]", leave=False
        )

        for data_batch in train_progress_bar:
            anchor, positive, n0, n1, n2, n3 = data_batch
            if epoch < 5:
                negative = n0
            elif epoch < 10:
                negative = n1
            elif epoch < 15:
                negative = n2
            elif epoch < 20:
                negative = n3
            else:
                negative = n3

            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )
            optimizer.zero_grad()
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            loss = loss_fcn(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            # Weight loss by batch size for correct averaging
            batch_size = anchor.size(0)
            running_train_loss += loss.item() * batch_size
            train_samples_processed += batch_size

            # Update the progress bar with the running average loss
            display_loss = running_train_loss / train_samples_processed
            train_progress_bar.set_postfix(loss=f"{display_loss:.4f}")

        # --- Validation Phase ---
        model.eval()
        m = compute_validation_metrics(model, val_loader)
        logging.info(f"Validation metrics:{m}")
        overlap_fraction = m["overlap_fraction"]
        separation_ratio = m["separation_ratio"]
        if scheduler:
            scheduler.step(overlap_fraction)

        # Save the model if it has the best overlap fraction so far
        # lesser the overlap better it is
        if overlap_fraction < best_overlap:
            best_overlap = overlap_fraction
            best_separation_ratio = separation_ratio
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                logging.info(
                    f"  -> New best model saved to '{save_path}' with overlap fraction: {best_overlap:.4f}, separation ratio:{best_separation_ratio:.4f}\n"
                )
        elif (overlap_fraction == best_overlap) and (
            separation_ratio > best_separation_ratio
        ):
            best_overlap = overlap_fraction
            best_separation_ratio = separation_ratio
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                logging.info("using separation ratio as criteria to save")
                logging.info(
                    f"  -> New best model saved to '{save_path}' with overlap fraction: {best_overlap:.4f}, separation ratio:{best_separation_ratio:.4f}\n"
                )

    end = time.time()
    duration = end - start
    logging.info(f"Training and Validation complete. Took {duration} seconds")
    if save_path:
        logging.info(
            f" Best model saved to '{save_path}' with overlap fraction: {best_overlap:.4f}, separation ratio:{best_separation_ratio:.4f}\n"
        )

    return model


def compute_validation_metrics(model, data_loader):
    model.eval()
    pos_distances = []
    neg_distances = []
    anchor_embeddings = []

    with torch.no_grad():
        for a, _, _ in data_loader:
            a = a.to(device)
            a_emb = model.get_embedding(a).squeeze(-1)
            anchor_embeddings.append(a_emb.cpu().numpy())

        centroid = np.mean(np.concatenate(anchor_embeddings, axis=0), axis=0)
        centroid_tensor = torch.tensor(centroid).to(device)

        for _, p, n in data_loader:
            p = p.to(device)
            n = n.to(device)
            p_embeddings = model.get_embedding(p).squeeze(-1)
            n_embeddings = model.get_embedding(n).squeeze(-1)

            pos_dist = torch.norm(p_embeddings - centroid_tensor, dim=1)
            neg_dist = torch.norm(n_embeddings - centroid_tensor, dim=1)

            pos_distances.extend(pos_dist.cpu().numpy())
            neg_distances.extend(neg_dist.cpu().numpy())

        pos_distances = np.array(pos_distances)
        neg_distances = np.array(neg_distances)

        y_true = np.array([0] * len(pos_distances) + [1] * len(neg_distances))
        scores = np.concatenate([pos_distances, neg_distances])

        roc_auc = roc_auc_score(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        sorted_indices = np.argsort(recall)
        pr_auc = auc(recall[sorted_indices], precision[sorted_indices])
        overlap_fraction = (neg_distances < np.percentile(pos_distances, 95)).mean()
        separation_ratio = neg_distances.mean() / (pos_distances.mean() + 1e-8)
        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "overlap_fraction": overlap_fraction,
            "separation_ratio": separation_ratio,
            "pos_mean": pos_distances.mean(),
            "neg_mean": neg_distances.mean(),
        }
