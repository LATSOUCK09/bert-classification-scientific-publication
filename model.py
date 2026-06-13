from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertForSequenceClassification


class BertForMultiLabelClassification(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()

        self.pretrained = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=n_class,
            problem_type="multi_label_classification"
        )
    def forward(self, input_ids, attention_mask):

        outputs = self.pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits