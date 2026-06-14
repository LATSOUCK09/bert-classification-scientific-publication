 #  boucles train_epoch / eval_epoch + main
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from dataset import TextClassificationDataset, create_dataloaders
from model import BertForMultiLabelClassification
from utils import set_seed, save_best_model, compute_metrics, ponderation_loss

#boucle d'entrainement pour une époque
def train_epoch(
        model,
        optimizer, 
        criterion,
        train_loader,
        device="cpu"):
    
    train_loss = 0.0
    all_preds = []
    all_labels = []
    model.train()

    num_batches = len(train_loader)
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"]
        attention_mask = batch["attention_mask"].to(device)
        output = model(inputs, attention_mask=attention_mask)
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = (torch.sigmoid(output) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss = train_loss / num_batches
    train_acc, _ = compute_metrics(all_preds, all_labels)
    return train_loss, train_acc


#boucle d'évaluation pour une époque
def val_epoch(model,
               val_loader, 
                 criterion, 
                  device="cpu"):
        
        valid_loss = 0.0
        correct= 0.0
        total= 0.0
        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            pbar=tqdm(val_loader, desc="Validation", unit="batch")
            for batch in pbar:
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device) # Move labels to device immediately
                attention_mask = batch["attention_mask"].to(device)
                output = model(inputs, attention_mask=attention_mask)
                loss = criterion(output, labels)
                valid_loss += loss.data.item()
                #ici nous allons chaner la fonctions argsmax pour la fonction sigmoid et faire une comparaison avec un seuil de 0.5 pour les problèmes de classification binaire
                preds = (torch.sigmoid(output) > 0.5).float()
                correct += (preds==labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        valid_loss = valid_loss / len(val_loader) # calculer la perte moyenne pour cette époque
        acc, f1 = compute_metrics(all_preds, all_labels)
        return valid_loss, acc, f1




def main():

    MODEL_NAME = (
        "bert-base-uncased"
    )

    DATA_PATH = (
        "data/scientific-publication.csv"
    )

    BATCH_SIZE = 8

    MAX_LENGTH = 256

    EPOCHS = 5

    LR = 2e-5

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device : {device}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    dataset = (
        TextClassificationDataset(
            csv_file=DATA_PATH,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH
        )
    )

    train_loader, val_loader = (
        create_dataloaders(
            dataset,
            batch_size=BATCH_SIZE
        )
    )

    model = (
        BertForMultiLabelClassification(
            MODEL_NAME,
            n_class=6
        )
        .to(device)
    )

    set_seed()
    pos_weights = ponderation_loss()
    criterion = nn.BCEWithLogitsLoss(pos_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_f1_score": [],
        "val_accuracy": []
    }

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            device=device
        )
        valid_loss, valid_accuracy, valid_f1 = val_epoch(
            model,
            val_loader,
            criterion,
            device=device
        )

        best_val_loss = save_best_model(
            model,
            valid_loss,
            best_val_loss,
            "bert_multilabel"
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(valid_loss)
        history["val_f1_score"].append(valid_f1)
        history["val_accuracy"].append(valid_accuracy)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} | "
            f"val_loss={valid_loss:.4f} val_accuracy={valid_accuracy:.4f} | "
            f"val_f1_score={valid_f1:.4f}"
        )


if __name__ == "__main__":
    main()