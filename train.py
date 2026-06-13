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
from utils import set_seed, save_best_model

#boucle d'entrainement pour une époque
def train_epoch(
        model,
        optimizer, 
        criterion,
        train_loader,
        device="cpu"):
    
    train_loss = 0.0
    correct = 0.0
    total = 0.0
    model.train()
    
    num_batches = len(train_loader)
        # Barre de progression
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad() # réinitialiser les gradients pour le prochain batch
        inputs = batch["input_ids"].to(device)
        target = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        output = model(inputs, attention_mask=attention_mask)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()  # mettre à jour les poids du modèle
        
        train_loss += loss.item()   # stocker la valeur de la perte pour cette itération
        #preds = torch.argmax(output, dim=1)
        #ici nous allons chaner la fonctions argsmax pour la fonction sigmoid et faire une comparaison avec un seuil de 0.5 pour les problèmes de classification binaire
        preds = (torch.sigmoid(output) > 0.5).float()
        correct += (preds == target).sum().item()
        total += target.size(0)            
    train_loss = train_loss / num_batches # calculer la perte moyenne pour cette époque
    train_accuracy = correct / total 
    return train_loss, train_accuracy


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
                targets = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                output = model(inputs, attention_mask=attention_mask)
                loss = criterion(output, targets)
                valid_loss += loss.data.item()
                #preds=torch.argmax(output,dim=1)
                #ici nous allons chaner la fonctions argsmax pour la fonction sigmoid et faire une comparaison avec un seuil de 0.5 pour les problèmes de classification binaire
                preds = (torch.sigmoid(output) > 0.5).float()
                correct += (preds==targets).sum().item()
                total += targets.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        valid_loss = valid_loss / len(val_loader) # calculer la perte moyenne pour cette époque
        valid_accuracy = correct / total    
        val_F1_score = f1_score(all_labels, all_preds, average="weighted")
        return valid_loss, valid_accuracy, val_F1_score




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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_f1_score": [],
        "VAL_ACCURACY": []
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
        history["VAL_ACCURACY"].append(valid_accuracy)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} | "
            f"val_loss={valid_loss:.4f} VAL_ACCURACY={valid_accuracy:.4f} | "
            f"val_f1_score={valid_f1:.4f}"
        )


if __name__ == "__main__":
    main()