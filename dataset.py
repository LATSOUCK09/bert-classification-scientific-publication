from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pandas as pd
import torch
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class TextClassificationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_file)
        self.data = self.data.drop(columns=["ID"])
        self.max_length = max_length
        self.label_columns = [
            col for col in self.data.columns if col not in ["ABSTRACT", "TITLE"]
        ]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        title = self.data.loc[idx, "TITLE"]
        abstract = self.data.loc[idx, "ABSTRACT"]
        text = title+ " [SEP] " + abstract
        labels = self.data.loc[idx, self.label_columns].values.astype(int)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length= self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float32)
        }
    

def create_dataloaders(
    dataset,
    batch_size=8,
    train_ratio=0.8,
    seed=42,
    sampler=None
):

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_sampler = None
    if sampler is not None:
        train_weights = sampler[train_dataset.indices]
        train_sampler = WeightedRandomSampler(train_weights, len(train_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader
    



