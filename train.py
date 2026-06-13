 #  boucles train_epoch / eval_epoch + main
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from dataset import TextClassificationDataset, create_dataloaders

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

