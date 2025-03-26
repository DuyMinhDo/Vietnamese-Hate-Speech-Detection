import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler
from tqdm import tqdm
from src.config import *
from src.dataset import get_data_loader
from src.model import PhoBERTClassifier, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

train_loader = get_data_loader(TRAIN_FILE, batch_size=BATCH_SIZE)
dev_loader = get_data_loader(DEV_FILE, batch_size=BATCH_SIZE)

model = PhoBERTClassifier().to(DEVICE)

def get_class_weights(train_file):
    df = pd.read_csv(train_file)
    labels = df['label_id'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    
class_weights = get_class_weights(TRAIN_FILE)
class_weights = torch.tensor([0.5, 3.0, 2.0], dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_training_steps = EPOCHS * len(train_loader)
warmup_steps = int(WARMUP_RATIO * num_training_steps)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

train_losses = []
val_losses = []
val_f1_scores = []

early_stopping = EarlyStopping(patience=3)

def evaluate():
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(DEVICE), 
                batch["attention_mask"].to(DEVICE), 
                batch["label"].to(DEVICE)
            )
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    avg_val_loss = total_val_loss / len(dev_loader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_val_loss, f1

def train():
    model.train()
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        total_train_loss = 0
        loop = tqdm(train_loader, leave=True)
        
        for batch in loop:
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE), 
                batch["label"].to(DEVICE)
            )

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            total_train_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss, val_f1 = evaluate()
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_f1_scores.append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val F1 = {val_f1:.4f}")
        print(f"Class weights applied: {class_weights.tolist()}")
        print(f"Warmup steps: {warmup_steps} (10% of {num_training_steps} total steps)")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH + "/best_model.pth")
            print(f"Best model updated at epoch {epoch+1} with Val Loss = {best_val_loss:.4f}")
        
        if early_stopping(avg_val_loss):
            print("Early stopping triggered. Training stopped.")
            break
        
    loss_data = {"train_loss": train_losses, "val_loss": val_losses, "val_f1": val_f1_scores}
    loss_file_path = os.path.join(LOG_SAVE_PATH, "losses.json")
    with open(loss_file_path, "w") as f:
        json.dump(loss_data, f, indent=4)

    print("Training Complete. Best model saved.")

if __name__ == "__main__":
    train()