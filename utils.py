<<<<<<< HEAD
=======
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from dataset import TextClassificationDataset
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score 
# %%
def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return acc, f1
>>>>>>> b66bd8eb1959eabf068cdb83d4be8a772d5250c8
