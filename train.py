import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler
from tqdm import tqdm
from src.config import *
from src.dataset import get_data_loader
from src.model import PhoBERTClassifier

train_loader = get_data_loader(TRAIN_FILE, batch_size=BATCH_SIZE)
dev_loader = get_data_loader(DEV_FILE, batch_size=BATCH_SIZE)

model = PhoBERTClassifier().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
num_training_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(WARMUP_RATIO * num_training_steps), num_training_steps=num_training_steps)

def train():
    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            input_ids, attention_mask, labels = batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), MODEL_SAVE_PATH + "/best_model.pth")
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    train()
