import os, sys

MODEL = "google/vit-base-patch16-224" # Path to HF model
TRAINED_MODEL_PATH = "models/demo_run.pth"
IN_DIM = 224
DEVICE = "cuda"
CSV_PATH = os.path.join(os.getcwd(), 'data', 'data.csv')
RESIZED_PATH = os.path.join(os.getcwd(), 'data', 'resized_images')
BATCH_SIZE = 4
LR = 0.001