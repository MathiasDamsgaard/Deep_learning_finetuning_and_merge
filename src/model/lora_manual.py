import os
import random
from typing import Tuple, Dict

import bitsandbytes as bnb
import numpy as np
import torch
import wandb
from loguru import logger
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from src.config.config import *
from src.model.baseline_model import load_dataset, load_model


class CustomTrainer:
    def __init__(self, model: ViTForImageClassification, optimizer: torch.optim.Optimizer, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset, epochs: int, batch_size: int = 8):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.epoch_accuracies = []
        self.epoch_losses = []

    def train(self) -> None:
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(pixel_values)
                loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * pixel_values.size(0)
                preds = outputs.logits.argmax(dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

            epoch_loss = running_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples
            self.epoch_losses.append(epoch_loss)
            self.epoch_accuracies.append(epoch_accuracy)

            # logger.info(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")
            wandb.log({"epoch_loss": epoch_loss, "epoch_accuracy": epoch_accuracy})

    def evaluate(self) -> Dict[str, float]:
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        running_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(pixel_values)
                loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
                running_loss += loss.item() * pixel_values.size(0)

                preds = outputs.logits.argmax(dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

        eval_loss = running_loss / total_samples
        eval_accuracy = correct_predictions / total_samples
        logger.info(f"Evaluation: Loss = {eval_loss:.4f}, Accuracy = {eval_accuracy:.4f}")
        wandb.log({"eval_loss": eval_loss, "eval_accuracy": eval_accuracy})
        return {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}

def get_lora_config(type_: str, r: int = 8, lora_alpha: float = 1.0) -> Tuple[ViTForImageClassification, torch.optim.Optimizer]:
    base_model = load_model()
    if type_ == "lora":
        lora_config = LoraConfig(
            r = r,
            lora_alpha = lora_alpha,
            init_lora_weights="gaussian", 
            target_modules=["query", "value"])
        return get_peft_model(base_model, lora_config), None
    elif type_ == "Q_lora":
        Q_lora_config = LoraConfig(
            r = r,
            lora_alpha = lora_alpha,
            init_lora_weights="gaussian", 
            target_modules="all-linear")
        return get_peft_model(base_model, Q_lora_config), None
    elif type_ == "lora_plus":
        lora_config = LoraConfig(
            r = r,
            lora_alpha = lora_alpha,
            init_lora_weights="gaussian", 
            target_modules=["query", "value"])
        optimizer = create_loraplus_optimizer(
            model=get_peft_model(base_model, lora_config),
            optimizer_cls=bnb.optim.Adam8bit,
            lr=5e-5,
            loraplus_lr_ratio=16,
        )
        return get_peft_model(base_model, lora_config), optimizer
    elif type_ == "Q_lora_plus":
        Q_lora_config = LoraConfig(
            r = r,
            lora_alpha = lora_alpha,
            init_lora_weights="gaussian", 
            target_modules="all-linear")
        optimizer = create_loraplus_optimizer(
            model=get_peft_model(base_model, Q_lora_config),
            optimizer_cls=bnb.optim.Adam8bit,
            lr=5e-5,
            loraplus_lr_ratio=16,
        )
        return get_peft_model(base_model, Q_lora_config), optimizer
    else:
        raise ValueError(f"Invalid type: {type_}")

def train_model_lora(epochs: int, type_: str, train_indices: np.ndarray = None, val_indices: np.ndarray = None) -> CustomTrainer:
    model, optimizer = get_lora_config(type_)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if train_indices is None or val_indices is None:
        train_subset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
        val_subset = load_dataset(csv_file=VAL_CSV, root_dir=VAL_DIR)
    else:
        train_dataset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
        val_dataset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

    trainer = CustomTrainer(
        model=model,
        optimizer=optimizer,
        train_dataset=train_subset,
        val_dataset=val_subset,
        epochs=epochs,
        batch_size=8
    )

    logger.info("Training model...")
    trainer.train()
    logger.success("Training complete")
    return trainer

def run_sweep(type_: str):
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "eval_accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"values": [1e-3, 1e-4, 1e-5]},
            "batch_size": {"values": [8, 16, 32]},
            "epochs": {"values": [5, 10, 15]},
            "type": {"values": [type_]},
            "r": {"values": [2, 4, 8, 16]}, # Rank of the LORA matrix
            "lora_alpha": {"values": [0.1, 0.5, 1.0, 2.0]}, # Alpha parameter for LORA
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="lora-sweep")

    def train_model_lora_wandb():
        wandb.init(reinit=True)
        config = wandb.config
        model, optimizer = get_lora_config(config.type, config.r, config.lora_alpha)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        train_subset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
        val_subset = load_dataset(csv_file=VAL_CSV, root_dir=VAL_DIR)

        trainer = CustomTrainer(
            model=model,
            optimizer=optimizer,
            train_dataset=train_subset,
            val_dataset=val_subset,
            epochs=config.epochs,
            batch_size=config.batch_size
        )

        logger.info("Training model...")
        trainer.train()
        logger.success("Training complete")
        eval_results = trainer.evaluate()
        wandb.log(eval_results)

    wandb.agent(sweep_id, function=train_model_lora_wandb, count=30)
    
    # Get the best configuration from wandb
    best_run = wandb.runs(wandb.sweep(sweep_id), order="desc", per_page=1)[0]
    best_config = best_run.config
    logger.info(f"Best configuration: {best_config}")
    torch.save(best_config, f"data/best_config_{type_}.pth")

def five_fold_cross_validation(epochs: int, type_: str) -> float:
    dataset = load_dataset(csv_file=TRAIN_CSV, root_dir=TRAIN_DIR)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        logger.info(f"Training fold {fold + 1}...")
        trainer = train_model_lora(epochs, type_, train_indices, val_indices)
        eval_results = trainer.evaluate()
        logger.success(f"Fold {fold + 1} results: {eval_results}")
        fold_results.append(eval_results)

    avg_accuracy = np.mean([result['eval_accuracy'] for result in fold_results])
    logger.info(f"Average cross-validated accuracy: {avg_accuracy}")
    return avg_accuracy

def lora_loop(type_: str, epochs: int = None, do_sweep: bool = False) -> float:
    if do_sweep:
        run_sweep(type_)
        logger.success("Sweep complete")
    else:
        config_path = f"data/best_config_{type_}.pth"
        if os.path.exists(config_path):
            config = torch.load(config_path)
            logger.info(f"Loaded best configuration: {config}")
            epochs = config.epochs
            batch_size = config.batch_size
        else:
            batch_size = 8
            epochs = epochs if epochs is not None else 3
            
        avg_accuracy = five_fold_cross_validation(epochs, type_)
        logger.info(f"Final model trained with optimal configuration: {config_path}")
        return avg_accuracy