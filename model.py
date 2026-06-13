from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertForSequenceClassification, BertModel


class BertForMultiLabelClassification(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()

        self.pretrained = BertModel.from_pretrained(
            model_name,)
        self.classifier = nn.Linear(
            self.pretrained.config.hidden_size, n_class
        )
    def forward(self, input_ids, attention_mask):

        outputs = self.pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
   
        return logits
    
