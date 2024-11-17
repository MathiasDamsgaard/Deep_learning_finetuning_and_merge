# Python native modules
import os
import sys
import argparse
import faulthandler
from typing import Union, List, Tuple, Dict, Any, Optional

# Computational modules
import numpy as np
import torch.backends
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Logging
from loguru import logger

# Custom modules
from src.model.baseline_model import train_model, save_model, load_model, model_infer, eval_predictions
from src.model.lora_model import LoraModel
from src.config.config import *

def main(step: Union[str, None] = None, BM: bool = False, epochs: Union[int, None] = None, c: bool = False, i_only: bool = False) -> None:
    """Main function for training and inference for the LoRA model tasks.

    Args:
        step (str, optional): Step of the main function to execute. Defaults to None.
    """
    if step == "demo":
        logger.info("Running demo step.")
        # Initialize model
        model = LoraModel(model_str=MODEL, in_dim=IN_DIM, device=DEVICE)
        logger.info(f"Initialized model: \n{model}")
        
    elif BM:
        logger.info("Running baseline model.")
        SAVE_PATH = os.path.join(os.getcwd(), "models", "baseline_model")
        
        model = load_model(c, SAVE_PATH)
        if c:
            logger.info(f"Continuing training model: \n{model}")
        else:
            logger.info(f"Initialized model: \n{model}")
        
        if not i_only:
            model = train_model(model, epochs)
            logger.info("Model trained successfully.")
            
            save_model(model, SAVE_PATH)
            logger.info("Model saved.")
        
        logger.info("Running inference.")
        preds = model_infer(model)
        #logger.info(f"Predicitions: {preds}")

        acc = eval_predictions()
        logger.info(f"Accuracy: {acc}")

if __name__ == "__main__":
    # Initialize faulthandler
    faulthandler.enable()
    
    # Initialize logger
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="LoRA model for training and inference.")
    parser.add_argument("--step", type=str, default=None, help="Step of the main function to execute.")
    parser.add_argument("BM", help="Baseline model", default=False, type=bool)
    parser.add_argument("--epochs", help="Number of epochs", default=1, type=int)
    parser.add_argument("--c", help="Use this if you want to continue training", action="store_true")
    parser.add_argument("--i_only", help="Use this if you want to only infer with the model", action="store_true")
    args = parser.parse_args()
    
    logger.info(f"Executing main function with step: {args.step}")
    
    # Run main function
    main(step=args.step, BM=args.BM, epochs=args.epochs, c=args.c, i_only=args.i_only)