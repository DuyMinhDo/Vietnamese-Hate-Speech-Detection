import torch
import torch.nn as nn
from transformers import AutoModel
from src.config import MODEL_NAME, DROPOUT, PATIENCE

class PhoBERTClassifier(nn.Module):
    def __init__(self):
        super(PhoBERTClassifier, self).__init__()
        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(self.phobert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(pooled_output))

class EarlyStopping:
    def __init__(self, patience=PATIENCE, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True