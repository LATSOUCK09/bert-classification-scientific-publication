import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_best_model(model, val_loss, best_val_loss, model_name):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_path = f"best_{model_name}.pth"
        torch.save(model.state_dict(), model_path)
    return best_val_loss


def compute_metrics(preds, labels, threshold=0.5, average="weighted"):
    """Compute accuracy, precision, recall and F1 for binary/multi-label outputs."""
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    preds = np.array(preds)
    labels = np.array(labels)

    if preds.ndim > 1:
        preds = (preds >= threshold).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average=average, zero_division=0)
    recall = recall_score(labels, preds, average=average, zero_division=0)
    f1 = f1_score(labels, preds, average=average, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_training_history(history, figsize=(10, 5), save_path=None):
    """Plot training and validation loss and metrics over epochs."""
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(epochs, history.get("train_loss", []), label="Train Loss", marker="o")
    ax1.plot(epochs, history.get("val_loss", []), label="Val Loss", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    if history.get("train_accuracy") is not None:
        ax2.plot(epochs, history.get("train_accuracy", []), label="Train Acc", linestyle="--", marker="x")
    if history.get("val_accuracy") is not None:
        ax2.plot(epochs, history.get("val_accuracy", []), label="Val Acc", linestyle="--", marker="x")
    if history.get("val_f1") is not None:
        ax2.plot(epochs, history.get("val_f1", []), label="Val F1", linestyle="--", marker="s")
    ax2.set_ylabel("Metrics")
    ax2.tick_params(axis="y")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()
