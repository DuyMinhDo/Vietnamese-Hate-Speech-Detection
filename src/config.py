import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "processed") 
MODEL_DIR = os.path.join(BASE_DIR, "..", "trained_models")  
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
DEMO_DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "demo_data")

TRAIN_FILE = os.path.join(DATA_DIR, "processed_train.csv")
DEV_FILE = os.path.join(DATA_DIR, "processed_dev.csv")
TEST_FILE = os.path.join(DATA_DIR, "processed_test.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "phobert_model")
LOG_SAVE_PATH = os.path.join(LOG_DIR, "phobert_model")
TEST_JSON_FILE = os.path.join(DEMO_DATA_DIR, "NLP_test.json")

MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 70
BATCH_SIZE = 16
EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
LEARNING_RATE = 2e-5
DROPOUT = 0.3
PATIENCE = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_SAVE_PATH, exist_ok=True)
