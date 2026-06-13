import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score 
# %%
def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return acc, f1