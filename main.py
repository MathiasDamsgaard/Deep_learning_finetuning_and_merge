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
import wandb

# Custom modules
from src.model.baseline_model import train_model, save_model, load_model, model_infer, eval_predictions
from src.model.lora_model import LoraModel
from src.model.lora import lora_loop
from src.config.config import *

os.environ['HF_HOME'] = '.cache/hf'
os.environ["WANDB_PROJECT"]="LoRA_model"
wandb.init(project="LoRA_model", mode="online")

def main(step: Union[str, None] = None, epochs: int = 1, r: int = 1, c: bool = False, i_only: bool = False) -> None:
    """Main function for training and inference for the LoRA model tasks.

    Args:
        step (str, optional): Step of the main function to execute. Defaults to None.
        epochs (int, optional): Number of epochs. Defaults to 1.
        r (int, optional): Rank to use. Defaults to 1.
        c (bool, optional): Use this if you want to continue training. Defaults to False.
        i_only (bool, optional): Use this if you want to only infer with the model. Defaults to False.
    """

    if step == "demo":
        logger.info("Running demo step.")
        # Initialize model
        model = LoraModel(model_str=MODEL, in_dim=IN_DIM, device=DEVICE)
        logger.info(f"Initialized model: \n{model}")
    
    elif step == "baseline":
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
        
    elif step == "lora":
        logger.info("Running the lora_config step.")
        score, profiler_data = lora_loop(type_=step, epochs=epochs, r=r)
        logger.info(f"Test Accuracy: {score * 100:.2f}%")
        logger.info(f"Profiler data: \n {profiler_data}")
    
    elif step == "Q_lora":
        logger.info("Running the Q_lora_config step.")
        score, profiler_data = lora_loop(type_=step, epochs=epochs, r=r)
        logger.info(f"Test Accuracy: {score * 100:.2f}%")
        logger.info(f"Profiler data: \n {profiler_data}")
    
    elif step == "lora_plus":
        logger.info("Running the lora_plus_config step.")
        score, profiler_data = lora_loop(type_=step, epochs=epochs, r=r)
        logger.info(f"Test Accuracy: {score * 100:.2f}%")
        logger.info(f"Profiler data: \n {profiler_data}")
    
    elif step == "Q_lora_plus":
        logger.info("Running the Q_lora_plus_config step.")
        score, profiler_data = lora_loop(type_=step, epochs=epochs, r=r)
        logger.info(f"Test Accuracy: {score * 100:.2f}%")
        logger.info(f"Profiler data: \n {profiler_data}")
    
    else:
        logger.error(f"Invalid step: {step}.")

if __name__ == "__main__":
    # Initialize faulthandler
    faulthandler.enable()

    # Initialize logger
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")

    # Parse arguments
    parser = argparse.ArgumentParser(description="LoRA model for training and inference.")
    parser.add_argument(
        "--step",
        type=str,
        default="lora",
        help="Step of the main function to execute.",
    )
    parser.add_argument("--epochs", help="Number of epochs", default=1, type=int)
    parser.add_argument("--r", help="Rank to use", default=1, type=int)
    parser.add_argument("--c", help="Use this if you want to continue training", action="store_true")
    parser.add_argument("--i_only", help="Use this if you want to only infer with the model", action="store_true")
    args = parser.parse_args()

    logger.info(f"Executing main function with step: {args.step}")

    # Run main function
    main(step=args.step, epochs=args.epochs, r=args.r, c=args.c, i_only=args.i_only)
