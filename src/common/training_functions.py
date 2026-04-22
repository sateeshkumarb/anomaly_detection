import logging
import torch
from tqdm import tqdm

from common.constants import device


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
    best_val_acc = 0.0

    logging.info("Starting Training and Validation")

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_samples_processed = 0
        train_progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{n_epochs} [Training]", leave=False
        )

        for data_batch in train_progress_bar:
            anchor, positive, negative = data_batch
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
        train_loss = running_train_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        correct_predictions = 0
        total_pairs = 0
        running_val_loss = 0.0
        val_samples_processed = 0
        val_progress_bar = tqdm(
            val_loader, desc=f"Epoch {epoch}/{n_epochs} [Validation]", leave=False
        )

        with torch.no_grad():
            for data_batch in val_progress_bar:
                anchor, positive, negative = data_batch
                anchor, positive, negative = (
                    anchor.to(device),
                    positive.to(device),
                    negative.to(device),
                )

                anchor_out, pos_out, neg_out = model(anchor, positive, negative)
                val_loss_item = loss_fcn(anchor_out, pos_out, neg_out)

                # Weight loss by batch size for correct averaging
                batch_size = anchor.size(0)
                running_val_loss += val_loss_item.item() * batch_size
                val_samples_processed += batch_size

                # Accuracy calculation
                dist_pos = torch.norm(anchor_out - pos_out, p=2, dim=1)
                dist_neg = torch.norm(anchor_out - neg_out, p=2, dim=1)
                correct_predictions += torch.sum(dist_pos < dist_neg).item()
                total_pairs += anchor.size(0)

                # Update running metrics on the progress bar
                current_acc = (
                    correct_predictions / total_pairs if total_pairs > 0 else 0
                )
                display_loss = running_val_loss / val_samples_processed
                val_progress_bar.set_postfix(
                    acc=f"{current_acc:.2%}", loss=f"{display_loss:.4f}"
                )

        val_accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0
        val_loss = running_val_loss / len(val_loader.dataset)

        # Print a summary for the epoch
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Validation Acc: {val_accuracy:.2%} | LR: {current_lr:.6f}"
        )

        # Update the learning rate scheduler, if one is provided
        if scheduler:
            scheduler.step(val_loss)

        # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                logging.info(
                    f"  -> New best model saved to '{save_path}' with validation accuracy: {best_val_acc:.2%}\n"
                )

    logging.info("Training and Validation Complete")
    if save_path:
        logging.info(
            f"Best model saved to '{save_path}' with accuracy {best_val_acc:.2%}"
        )

    return model
