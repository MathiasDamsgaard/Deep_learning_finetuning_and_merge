# Python native modules
import os
import sys
import argparse
import faulthandler
from typing import Union, List, Tuple, Dict, Any, Optional

os.environ['HF_HOME'] = os.path.join(os.getcwd(), ".cache/hf")
os.environ["WANDB_PROJECT"]="LoRA_model"

# Computational modules
import numpy as np
import torch.backends
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import ViTForImageClassification

# Logging
from loguru import logger
import wandb
# from thop import profile, clever_format
from ptflops import get_model_complexity_info

# Custom modules
from src.model.baseline_model import train_model, save_model, load_model, model_infer, eval_predictions, load_dataset
from src.model.lora_model import LoraModel
# from src.model.lora import lora_loop, train_model_lora_wandb
from src.model.lora_manual import lora_loop, LoRATrainer, get_lora_config
from src.config.config import *

# wandb.init(project="LoRA_model", mode="online")

def main(step: Union[str, None] = None,
         epochs: Union[int, None] = None, 
         c: bool = False, 
         i_only: bool = False, 
         type_: str = "lora",
         num_folds: int = 1,
         config: dict = None,
         checkpoint: str = None) -> None:
    
    """Main function for training and inference for the LoRA model tasks.

    Args:
        step (str): Step of the main function to execute.
        BM (bool): Baseline model.
        epochs (int): Number of epochs.
        c (bool): Use this if you want to continue training.
        i_only (bool): Use this if you want to only infer with the model.
        type_ (str): Type of model.
        config (dict): Configuration dictionary for the model.
        checkpoint (str): Path to checkpoint.
        
    Returns:
        None
    """
        
    if step == "train":
        logger.info(f"Running train step for model type {type_}.")
        score, trainer = lora_loop(type_, epochs, num_folds=num_folds, config=config)
        trainable_params, all_params = trainer.get_trainable_parameters()
        macs, params = get_model_complexity_info(trainer.model, (3, 224, 224), as_strings=True, backend='pytorch',
                                           print_per_layer_stat=True, verbose=True)
        _, col_name = trainer.model_infer(load_dataset(TEST_CSV, TEST_DIR))
        acc = trainer.eval_predictions(col_name)
        logger.success(f"Model trained on entire training set. Inference results: {acc:.2f}")
        logger.info(f"Trainable parameters: {trainable_params}, All parameters: {all_params}, MACs: {macs}, Parameters (ptflops): {params}")
        trainer.visualize_misclassifications(col_name=col_name,
                                             images_or_hist="histograms")
        trainer.visualize_misclassifications(col_name=col_name,
                                             image_root_dir=TEST_DIR,
                                             num_examples=5,
                                             saliency=True,
                                             images_or_hist="images")
        
    elif step == "get_lora_config":
        logger.info("Running get_lora_config step.")
        score, profiler_data = lora_loop(type_=type_, epochs=epochs)
        logger.info(f"Test Accuracy: {score * 100:.2f}%")
        logger.info(f"Profiler data: \n {profiler_data}")
        
    elif step == "sweep_manual":
        logger.info("Running sweep_manual step.")
        lora_loop(type_=type_, do_sweep=True)
        
    elif step == "plot":
        trainer = LoRATrainer(None, 
                            None, 
                            load_dataset(TRAIN_CSV, TRAIN_DIR), 
                            load_dataset(VAL_CSV, VAL_DIR), 
                            epochs=30, 
                            batch_size=8,
                            type_=type_)
        logger.info("Running plot step.")
        trainer.model = trainer.load_model(path="google/vit-base-patch16-224")
        # logger.debug(trainer.model)
        if checkpoint is not None:
            trainer.load_trained_model(checkpoint)
            col_name = checkpoint.split("/")[-2] + "_" + checkpoint.split("/")[-1].split("_")[0]
        else:
            trainer.load_trained_model("/dtu/blackhole/15/155381/Deep_learning_finetuning_and_merge/models/lora/19109_53.045")
            col_name = "lora_19109"
        sorted_summary, _ = trainer.analyze_misclassifications(col_name=col_name, verbose=False)
        trainer.visualize_misclassifications(col_name=col_name,
                                             image_root_dir=TEST_DIR,
                                             num_examples=5,
                                             saliency=True,
                                             images_or_hist="images")
        # print(sorted_summary)
        
    elif step == "print_params":
        trainer = LoRATrainer(None,
                            None,
                            load_dataset(TRAIN_CSV, TRAIN_DIR),
                            load_dataset(VAL_CSV, VAL_DIR),
                            epochs=30,
                            batch_size=8,
                            type_=type_)
        trainer.model, _, _ = get_lora_config(type_, r = config["r"], lora_alpha = config["lora_alpha"], loraplus_lr_ratio = config["loraplus_lr_ratio"], dropout = config["dropout"])
        trainer.print_trainable_parameters()
        macs, params = get_model_complexity_info(trainer.model, (3, 224, 224), as_strings=True, backend='pytorch',
                                           print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        # print(trainer.model.state_dict().keys())
        # rand_input = torch.randn(1, 3, 224, 224).to(trainer.device)
        # macs, params = profile(trainer.model, inputs=(rand_input, ), verbose=False)
        # macs, params = clever_format([macs, params], "%.3f")
        # print(f"Parameters for model type: {type_}MACs: {macs}, Parameters: {params}")
        
    if step == "BM":
        logger.info("Running baseline model.")
        SAVE_PATH = os.path.join(os.getcwd(), "models", "baseline_model")
        
        model = load_model(c, SAVE_PATH, BM=True)
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
    # logger.remove()
    # logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")

    # Parse arguments
    parser = argparse.ArgumentParser(description="LoRA model for training and inference.")
    parser.add_argument(
        "--step",
        type=str,
        default="lora",
        help="Step of the main function to execute.",
    )
    parser.add_argument("--epochs", help="Number of epochs", default=15, type=int)
    parser.add_argument("--num_folds", help="Number of folds for cross validation", default=5, type=int)
    parser.add_argument("--c", help="Use this if you want to continue training", action="store_true")
    parser.add_argument("--i_only", help="Use this if you want to only infer with the model", action="store_true")
    parser.add_argument("--type", help="Type of model", default="lora", type=str)
    parser.add_argument("--learning_rate", help="Learning rate", default=0.0015, type=float)
    parser.add_argument("--batch_size", help="Batch size", default=16, type=int)
    parser.add_argument("--r", help="Rank", default=32, type=int)
    parser.add_argument("--lora_alpha", help="lora_alpha", default=4.0, type=float)
    parser.add_argument("--loraplus_lr_ratio", help="loraplus_lr_ratio", default=32, type=int)
    parser.add_argument("--dropout", help="Dropout", default=0.2, type=float)
    parser.add_argument("--checkpoint", help="Path to checkpoint", default=None, type=str)
    args = parser.parse_args()
    
    assert args.type in ["baseline", "lora", "Q_lora", "lora_plus", "Q_lora_plus"], "Invalid model type."
    
    # Create a configuration dict from the arguments as  config = {"learning_rate": 1.5e-4, 
                #   "batch_size": 8, 
                #   "r": 32, 
                #   "lora_alpha": 4.0, 
                #   "loraplus_lr_ratio": 32,
                #   "dropout": 0.2}
                
    config = {key: value for key, value in args.__dict__.items() if key not in ["step", "BM", "epochs", "c", "i_only", "type"]}

    logger.info(f"Executing main function with step: {args.step}")

    # Run main function
    main(step=args.step, 
         epochs=args.epochs, 
         c=args.c, 
         i_only=args.i_only, 
         type_=args.type, 
         num_folds=args.num_folds, 
         config=config,
         checkpoint=args.checkpoint)
