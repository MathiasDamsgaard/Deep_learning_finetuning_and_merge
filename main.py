# Python native modules
import os
import sys
import argparse
import faulthandler
from typing import Union

# Computational modules
import numpy as np
import torch.backends
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTModel, ViTForImageClassification

# Logging
from loguru import logger

# Custom modules
from src.config.config import (MODEL, IN_DIM, DEVICE, CSV_PATH, RESIZED_PATH, BATCH_SIZE, LR, TRAINED_MODEL_PATH)
from src.model.lora_model import LoraModel
from src.data.dataset import PlaneTypeDataset
from src.utils.train import ModelTrainer

def main(step: Union[str, None] = None, epochs: Union[int, None] = None) -> None:
    """Main function for training and inference for the LoRA model tasks.

    Args:
        step (str, optional): Step of the main function to execute. Defaults to None.
    """
    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE=="cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    if step == "demo":
        logger.info("Running demo step.")
        # Initialize model
        model = LoraModel(model_str=MODEL, in_dim=IN_DIM, device=device)
        logger.info(f"Initialized model: \n{model}")
        
    elif step == "train":
        logger.info("Running training.")
        # Initialize model
        # model = LoraModel(model_str=MODEL, in_dim=IN_DIM, device=DEVICE)
        # Baseline for now
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        dataset = PlaneTypeDataset(CSV_PATH,
                                   RESIZED_PATH,
                                   processor=processor)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        trainer = ModelTrainer(model=model, dataloader=dataloader, criterion=nn.CrossEntropyLoss(),
                               optimizer=optim.Adam(model.parameters(), lr=LR), epochs=epochs)
        logger.info(f"Initialized training setup. Training for {epochs} epochs.")
        trainer.train_model()
        trainer.save_model(path=TRAINED_MODEL_PATH)
    
if __name__ == "__main__":
    # Initialize faulthandler
    faulthandler.enable()
    
    # Initialize logger
    # logger.remove()
    # logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="LoRA model for training and inference.")
    parser.add_argument("--step", type=str, default=None, help="Step of the main function to execute.")
    parser.add_argument("--epochs", help="Number of epochs", default=1, type=int)
    args = parser.parse_args()
    
    assert args.step in ["demo", "train"], "Invalid step argument. Please choose from ['demo', 'train']"
    
    logger.info(f"Executing main function with step: {args.step}")
    
    # Run main function
    main(step=args.step, epochs=args.epochs)