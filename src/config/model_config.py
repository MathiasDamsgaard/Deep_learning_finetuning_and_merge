import torch
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb
from torch.optim.lr_scheduler import StepLR
from loguru import logger
from typing import Tuple, Optional

def get_lora_config(base_model, type_: str, r: int = 8, lora_alpha: float = 1.0, learning_rate: float = 1e-5, loraplus_lr_ratio: int = 16, dropout: float = 0.01, step_size: int = 10, gamma: float = 0.1, **kwargs) -> Tuple[ViTForImageClassification, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Configures and returns a LoRA model, an optimizer, and a learning rate scheduler based on the type specified.

    Args:
        type_ (str): Type of LoRA model to configure.
        r (int): Rank of the LoRA adaptation.
        lora_alpha (float): Scaling factor for LoRA.
        learning_rate (float): Learning rate for the optimizer.
        loraplus_lr_ratio (int): Ratio for LoRA+ optimizer learning rate.
        dropout (float): Dropout rate for LoRA.
        step_size (int): Step size for the learning rate scheduler.
        gamma (float): Multiplicative factor for the learning rate scheduler.

    Returns:
        Tuple[ViTForImageClassification, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]: Configured model, optimizer, and scheduler.
    """
    # base_model = load_model()
    logger.info(f"Loading {type_} model with parameters: r={r}, lora_alpha={lora_alpha}, learning_rate={learning_rate}, loraplus_lr_ratio={loraplus_lr_ratio}, step_size={step_size}, gamma={gamma}")

    if type_ == "lora":
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["query", "value"],
            lora_dropout=dropout
        )
        model = get_peft_model(base_model, lora_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return model, optimizer, scheduler

    elif type_ == "Q_lora":
        Q_lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules="all-linear",
            lora_dropout=dropout
        )
        model = get_peft_model(base_model, Q_lora_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return model, optimizer, scheduler

    elif type_ == "lora_plus":
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["query", "value"],
            lora_dropout=dropout
        )
        model = get_peft_model(base_model, lora_config)
        optimizer = create_loraplus_optimizer(
            model=model,
            optimizer_cls=bnb.optim.Adam8bit,
            lr=learning_rate,
            loraplus_lr_ratio=loraplus_lr_ratio
        )
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return model, optimizer, scheduler

    elif type_ == "Q_lora_plus":
        Q_lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules="all-linear",
            lora_dropout=dropout
        )
        model = get_peft_model(base_model, Q_lora_config)
        optimizer = create_loraplus_optimizer(
            model=model,
            optimizer_cls=bnb.optim.Adam8bit,
            lr=learning_rate,
            loraplus_lr_ratio=loraplus_lr_ratio
        )
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return model, optimizer, scheduler

    elif type_ == "baseline":
        optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return base_model, optimizer, scheduler

    else:
        raise ValueError(f"Invalid type: {type_}")