import torch
import os

MODEL = "google/vit-base-patch16-224" # Path to HF model
IN_DIM = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join(os.getcwd(), "data", "resized_images")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

CSV_DIR = os.path.join(os.getcwd(), 'data')
TRAIN_CSV = os.path.join(CSV_DIR, 'train.csv')
VAL_CSV = os.path.join(CSV_DIR, 'val.csv')
TEST_CSV = os.path.join(CSV_DIR, 'test.csv')