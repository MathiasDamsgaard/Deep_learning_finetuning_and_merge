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

# Logging
from loguru import logger

# Custom modules
from src.model.baseline_model import train_model
from src.model.lora_model import LoraModel
from src.config.config import MODEL, IN_DIM, DEVICE

def main(step: Union[str, None] = None, bm: Union[str, None] = None, epochs: Union[int, None] = None) -> None:
    """Main function for training and inference for the LoRA model tasks.

    Args:
        step (str, optional): Step of the main function to execute. Defaults to None.
    """
    if step == "demo":
        logger.info("Running demo step.")
        # Initialize model
        model = LoraModel(model_str=MODEL, in_dim=IN_DIM, device=DEVICE)
        logger.info(f"Initialized model: \n{model}")
        
    elif bm == "baseline_model":
        logger.info("Running baseline model step.")
        train_model(epochs)
    
if __name__ == "__main__":
    # Initialize faulthandler
    faulthandler.enable()
    
    # Initialize logger
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="LoRA model for training and inference.")
    parser.add_argument("--step", type=str, default=None, help="Step of the main function to execute.")
    parser.add_argument("BM", help="Baseline model")
    parser.add_argument("--epochs", help="Number of epochs", default=1, type=int)
    args = parser.parse_args()
    
    logger.info(f"Executing main function with step: {args.step}")
    
    # Run main function
    main(step=args.step, bm=args.BM, epochs=args.epochs)