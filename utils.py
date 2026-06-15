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


def weight_ponderation(df):
    label_columns = df.drop(columns=["ID", "TITLE", "ABSTRACT"]).columns
    n_samples = len(df)
    pos_weights = []
    for col in label_columns:
        positive_count = df[col].sum()
        negative_count = n_samples - positive_count
        pos_weights.append(negative_count / positive_count)
    return torch.tensor(pos_weights, dtype=torch.float32)


def weighted_sampling(df):
    label_columns = df.drop(columns=["ID", "TITLE", "ABSTRACT"]).columns
    n_samples = len(df)
    class_weights = {col: n_samples / df[col].sum() for col in label_columns}
    sample_weights = []
    for _, row in df.iterrows():
        active = [class_weights[col] for col in label_columns if row[col] == 1]
        sample_weights.append(np.mean(active))
    return torch.DoubleTensor(sample_weights)