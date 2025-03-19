import torch
import torch.nn as nn
from transformers import AutoModel
from src.config import MODEL_NAME

class PhoBERTClassifier(nn.Module):
    def __init__(self):
        super(PhoBERTClassifier, self).__init__()
        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.phobert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(pooled_output))
