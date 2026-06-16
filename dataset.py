from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pandas as pd
import torch
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class TextClassificationDataset(Dataset):
    """
    Dataset custom pour une tâche de classification multilabel de textes.

    Cette classe hérite de torch.utils.data.Dataset et permet de charger un fichier
    CSV contenant des textes (titre et résumé) ainsi que leurs labels associés.
    Les textes sont combinés puis tokenisés à l'aide d'un tokenizer Transformer
    afin d'être utilisés comme entrée d'un modèle de classification NLP.

    Le fichier CSV doit contenir au minimum les colonnes :
        - TITLE : titre du document
        - ABSTRACT : résumé du document
        - Colonnes de labels : variables binaires indiquant l'appartenance à chaque classe

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer):
            Tokenizer utilisé pour convertir les textes en séquences de tokens.

        data (pandas.DataFrame):
            Jeu de données chargé depuis le fichier CSV après suppression de la colonne ID.
max_length (int):
            Longueur maximale des séquences tokenisées. Les séquences plus courtes
            sont complétées par padding et les plus longues sont tronquées.

        label_columns (list[str]):
            Liste des colonnes correspondant aux labels de classification.
            Ces colonnes sont toutes les colonnes du DataFrame sauf TITLE et ABSTRACT.

    Methods:
        __len__():
            Retourne le nombre d'exemples présents dans le dataset.

        __getitem__(idx):
            Retourne un exemple tokenisé avec ses labels associés.

            Args:
                idx (int):
                    Index de l'exemple à récupérer.

            Returns:
                dict:Dictionnaire contenant :
                    - input_ids (torch.Tensor):
                        Identifiants des tokens générés par le tokenizer.
                    - attention_mask (torch.Tensor):
                        Masque indiquant les tokens réels et les tokens de padding.
                    - labels (torch.Tensor):
                        Vecteur binaire représentant les classes associées au texte.
    """
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
    batch_size=16,
    train_ratio=0.8,
    seed=42,
    sampler=None
):
    
    """
    Crée les DataLoaders d'entraînement et de validation.

    La fonction divise le dataset en deux sous-ensembles selon le ratio
    spécifié, puis construit les DataLoaders correspondants. Un
    WeightedRandomSampler peut être utilisé pour rééquilibrer les
    données d'entraînement lorsque les classes sont fortement
    déséquilibrées.

    Args:
        dataset (torch.utils.data.Dataset):
            Dataset complet contenant les exemples d'entraînement.

        batch_size (int, optional):
            Nombre d'échantillons par batch.
            Défaut : 16.

        train_ratio (float, optional):
            Proportion du dataset utilisée pour l'entraînement.
            Défaut : 0.8.

        seed (int, optional):
            Graine utilisée pour garantir la reproductibilité du split.
            Défaut : 42.

        sampler (torch.Tensor, optional):
            Tensor contenant les poids d'échantillonnage calculés pour
            chaque exemple du dataset. Si fourni, un
            WeightedRandomSampler est utilisé pour le DataLoader
            d'entraînement.

    Returns:
        tuple:
            Un tuple contenant :

            - train_loader (DataLoader) :
              DataLoader utilisé pour l'entraînement.

            - val_loader (DataLoader) :
              DataLoader utilisé pour la validation.

    Example:
        >>> sample_weights = weighted_sampling(df)
        >>> train_loader, val_loader = create_dataloaders(
        ...     dataset,
        ...     batch_size=16,
        ...     sampler=sample_weights
        ... )
    """

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
    



