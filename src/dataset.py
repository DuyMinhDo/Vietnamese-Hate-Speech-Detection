from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import torch
from src.config import MODEL_NAME, MAX_LENGTH, BATCH_SIZE

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class HateSpeechDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data["cleaned_comment"] = self.data["cleaned_comment"].fillna("").astype(str)
        self.texts = self.data["cleaned_comment"].tolist()
        self.labels = self.data["label_id"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def get_data_loader(file_path, batch_size=BATCH_SIZE):
    dataset = HateSpeechDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
