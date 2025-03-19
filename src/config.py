import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset", "processed") 
MODEL_DIR = os.path.join(BASE_DIR, "..", "trained_models")  

TRAIN_FILE = os.path.join(DATA_DIR, "processed_train.csv")
DEV_FILE = os.path.join(DATA_DIR, "processed_dev.csv")
TEST_FILE = os.path.join(DATA_DIR, "processed_test.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "phobert_model")

MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 50
BATCH_SIZE = 32 
EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1 
WEIGHT_DECAY = 0.01  
FP16 = True 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
