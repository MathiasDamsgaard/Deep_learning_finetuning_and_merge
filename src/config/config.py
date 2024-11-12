import torch

MODEL = "google/vit-base-patch16-224" # Path to HF model
IN_DIM = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")