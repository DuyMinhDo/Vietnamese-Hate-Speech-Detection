import torch
from src.config import *
from src.dataset import get_data_loader
from src.model import PhoBERTClassifier
from sklearn.metrics import classification_report

test_loader = get_data_loader(TEST_FILE, batch_size=BATCH_SIZE)

model = PhoBERTClassifier()
model.load_state_dict(torch.load(MODEL_SAVE_PATH + "/best_model.pth"))
model.to(DEVICE)
model.eval()

def evaluate():
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()
