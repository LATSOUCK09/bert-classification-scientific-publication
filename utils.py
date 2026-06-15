import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import dataset


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



def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return acc, f1

def plot_training_history(history, figsize=(10, 5), save_path=None):
    """Plot training and validation loss and metrics history."""
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

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

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def weight_ponderation(dataset=None):
    if dataset is None:
        raise ValueError(
            "Le paramètre 'dataset' est requis pour weight_ponderation. "
            "Passez un DataFrame pandas déjà chargé."
        )

    drop_columns = [col for col in ["ID", "TITLE", "ABSTRACT"] if col in dataset.columns]
    label_columns = dataset.drop(columns=drop_columns).columns
    n_samples = len(dataset)
    pos_weights = []
    for col in label_columns:
        positive_count = dataset[col].sum()
        negative_count = n_samples - positive_count
        weights = negative_count / positive_count
        pos_weights.append(weights)

    pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
    return pos_weights


def weighted_sampling(dataset=None, indices=None, replacement=True):
    """Build a WeightedRandomSampler from a pandas DataFrame.

    Parameters
    - dataset: pandas.DataFrame containing label columns (excludes ID/TITLE/ABSTRACT)
    - indices: optional list/sequence of row indices to restrict the sampler to a subset
    - replacement: whether sampling is with replacement

    Returns a `torch.utils.data.WeightedRandomSampler` ready to pass to a `DataLoader`.
    """
    if dataset is None:
        raise ValueError(
            "Le paramètre 'dataset' est requis pour weight_ponderation. "
            "Passez un DataFrame pandas déjà chargé."
        )

    drop_columns = [col for col in ["ID", "TITLE", "ABSTRACT"] if col in dataset.columns]
    label_columns = dataset.drop(columns=drop_columns).columns
    n_samples = len(dataset)
    class_weights = {}
    for col in label_columns:
        count = dataset[col].sum()
        class_weights[col] = n_samples / count
    sample_weights = []

    for _, row in dataset.iterrows():
        active_classes = []
        for col in label_columns:
            if row[col] == 1:
                active_classes.append(
                    class_weights[col]
                )
        # If no active classes for this sample, give it a default weight=1.0
        if len(active_classes) == 0:
            sample_weights.append(1.0)
        else:
            sample_weights.append(np.mean(active_classes))

    # Convert to tensor of weights per sample and build a WeightedRandomSampler
    cleaned_weights = []
    for w in sample_weights:
        if np.isnan(w) or np.isinf(w):
            cleaned_weights.append(1.0)
        else:
            cleaned_weights.append(float(w))

    # If indices provided, select only those sample weights (useful when using Subset)
    if indices is not None:
        subset_weights = [cleaned_weights[i] for i in indices]
        weights_tensor = torch.DoubleTensor(subset_weights)
        num_samples = len(subset_weights)
    else:
        weights_tensor = torch.DoubleTensor(cleaned_weights)
        num_samples = len(weights_tensor)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=num_samples,
        replacement=replacement,
    )

    return sampler

