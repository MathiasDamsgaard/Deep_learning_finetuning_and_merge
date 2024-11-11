import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoImageProcessor, AutoModelForImageClassification

class LoraModel(nn.Module):
    """
    LoRA model for training and inference.
    
    This class defines the model architecture as well 
    as the training and inference methods.
    """
    def __init__(self, model_str: str = "google/vit-base-patch16-224", in_dim: int = 224, device: str = 'cuda') -> None:
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_str, use_fast=True) # Load fast processor
        self.model = AutoModelForImageClassification.from_pretrained(model_str)
        logger.info(f"Initialized model on device: {self.device}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x