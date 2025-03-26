import torch
import json
from src.config import *
from src.model import PhoBERTClassifier
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = PhoBERTClassifier()
model.load_state_dict(torch.load(MODEL_SAVE_PATH + "/best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

label_map = {0: "Clean", 1: "Offensive", 2: "Hate"}

def predict_comment(comment):
    encoded = tokenizer(
        comment,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
    
    return label_map[pred]

with open(TEST_JSON_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)
    
if __name__ == "__main__":
    print("Hate Speech Classifier Demo")
    print(f"Predicting labels for comments from {TEST_JSON_FILE}...\n")

    for item in test_data["test_data"]:
        comment = item["text"]
        prediction = predict_comment(comment)
        print(f"Comment: {comment}")
        print(f"Prediction: {prediction}\n")